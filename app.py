"""
app.py — Графическое веб-приложение Spinel Studio (на базе Streamlit).

Интуитивный интерфейс для геологов, кристаллохимиков и материаловедов:
- Моделирование реальных многокомпонентных шпинелей, бинарных растворов A–B,
  модельных систем ABCDE и любых пользовательских составов с произвольными элементами.
- Загрузка собственных файлов структур (.docx, .txt, .gin, .out, .cif) или генерация
  идеальных суперъячеек шпинели Fd-3m любой размерности (2×2×4, 2×2×2 и др.).
- Строго раздельная оптимизация тетраэдров (A, CN=4) и октаэдров (B, CN=6) по критерию χ².
- Выгрузка готовых пакетов для GULP (.gin), VESTA (.cif), Word (.docx), графиков 300 dpi (.png) и CSV/JSON.
"""

from __future__ import annotations

import io
import math
import re
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine import (
    SimulationResult,
    StructureTemplate,
    ValidationReport,
    analyze_coordination_shells,
    build_ideal_spinel_supercell,
    build_summary_dataframes,
    build_zip_package,
    parse_structure_file,
    run_simulation,
    to_coordinate_docx,
    to_coordinate_txt,
    to_gulp_gin,
    to_vesta_cif,
    validate_setup,
)
from presets import (
    ATOMIC_MASSES,
    LATTICE_TEMPLATES,
    CompositionConfig,
    SpeciesSpec,
    get_builtin_presets,
)

GAS_CONSTANT_R: float = 8.314462618  # Дж/(моль·К)


# =============================================================================
# Чистые вспомогательные функции бизнес-логики интерфейса (без привязки к UI)
# =============================================================================

def sanitize_prefix(raw_prefix: str) -> str:
    """Очищает имя префикса файлов от пробелов и спецсимволов."""
    cleaned = re.sub(r"[^\w\-]+", "_", str(raw_prefix).strip(), flags=re.UNICODE)
    cleaned = cleaned.strip("_")
    return cleaned or "spinel_model"


def species_to_dataframe(specs: Sequence[SpeciesSpec]) -> pd.DataFrame:
    """Преобразует список SpeciesSpec в аккуратную таблицу для редактора."""
    if not specs:
        return pd.DataFrame({
            "Элемент": pd.Series(dtype="str"),
            "Количество": pd.Series(dtype="int64"),
            "Заряд (e)": pd.Series(dtype="float64"),
        })
    return pd.DataFrame([
        {
            "Элемент": str(s.element),
            "Количество": int(s.count),
            "Заряд (e)": float(s.charge),
        }
        for s in specs
    ])


def dataframe_to_species(df: pd.DataFrame, sublattice_label: str) -> Tuple[List[SpeciesSpec], List[str]]:
    """
    Безопасно преобразует таблицу из st.data_editor (включая новые строки с None/NaN/pd.NA)
    в список SpeciesSpec. Возвращает (список_компонентов, список_ошибок_ввода).
    """
    specs: List[SpeciesSpec] = []
    errors: List[str] = []

    if df is None or df.empty:
        return specs, [f"Таблица «{sublattice_label}» пуста — добавьте хотя бы один элемент."]

    for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
        raw_el = row.get("Элемент", None)
        raw_cnt = row.get("Количество", None)
        raw_chg = row.get("Заряд (e)", None)

        el_empty = pd.isna(raw_el) or str(raw_el).strip() == ""
        cnt_empty = pd.isna(raw_cnt) or str(raw_cnt).strip() == ""
        chg_empty = pd.isna(raw_chg) or str(raw_chg).strip() == ""

        # Полностью пустую строку (или строку с 0 атомов без символа) тихо пропускаем
        if el_empty and cnt_empty:
            continue
        if el_empty and not cnt_empty:
            try:
                if int(round(float(raw_cnt))) == 0:
                    continue
            except Exception:
                pass
            errors.append(f"{sublattice_label}, строка {row_idx}: не указан символ элемента.")
            continue

        el_str = str(raw_el).strip()
        if not re.fullmatch(r"[A-Za-zА-Яа-я0-9_+\-]+", el_str):
            errors.append(f"{sublattice_label}, строка {row_idx}: недопустимый символ элемента «{el_str}».")
            continue

        if cnt_empty:
            cnt_val = 0
        else:
            try:
                cnt_float = float(raw_cnt)
                if cnt_float < 0:
                    errors.append(f"{sublattice_label}, элемент «{el_str}»: количество не может быть отрицательным.")
                    continue
                cnt_val = int(round(cnt_float))
            except Exception:
                errors.append(f"{sublattice_label}, элемент «{el_str}»: некорректное количество атомов.")
                continue

        if chg_empty:
            chg_val = 0.0
        else:
            try:
                chg_val = float(raw_chg)
            except Exception:
                errors.append(f"{sublattice_label}, элемент «{el_str}»: некорректное число в поле заряда.")
                continue

        if cnt_val > 0:
            specs.append(SpeciesSpec(element=el_str, count=cnt_val, charge=chg_val))

    if not specs and not errors:
        errors.append(f"В таблице «{sublattice_label}» сумма количеств равна 0 — укажите ненулевое число атомов.")

    return specs, errors


def scale_counts_proportionally(df: pd.DataFrame, target_total: int) -> pd.DataFrame:
    """
    Пересчитывает введённые числа, доли или проценты в точные целые числа атомов
    по методу наибольших остатков (метод Гамильтона) так, чтобы сумма строго равнялась target_total.
    """
    if df is None or df.empty or target_total <= 0:
        return df

    out = df.copy().reset_index(drop=True)
    valid_indices: List[int] = []
    weights: List[float] = []

    for idx, row in out.iterrows():
        raw_el = row.get("Элемент", None)
        raw_cnt = row.get("Количество", None)
        if pd.isna(raw_el) or str(raw_el).strip() == "":
            continue
        w = 0.0
        if not pd.isna(raw_cnt):
            try:
                w = max(float(raw_cnt), 0.0)
            except Exception:
                w = 0.0
        valid_indices.append(idx)
        weights.append(w)

    if not valid_indices:
        return out

    total_w = sum(weights)
    if total_w <= 0:
        # Если все количества были 0, делим поровну между всеми вписанными элементами
        weights = [1.0] * len(valid_indices)
        total_w = float(len(valid_indices))

    exact = [w / total_w * target_total for w in weights]
    floors = [int(math.floor(x)) for x in exact]
    remainder = target_total - sum(floors)

    order = sorted(range(len(exact)), key=lambda i: (exact[i] - floors[i]), reverse=True)
    for k in range(remainder):
        floors[order[k]] += 1

    for pos, df_idx in enumerate(valid_indices):
        out.loc[df_idx, "Количество"] = int(floors[pos])

    return out


def compute_balanced_charge(
    tet_specs: Sequence[SpeciesSpec],
    oct_specs: Sequence[SpeciesSpec],
    n_anion: int,
    anion_charge: float,
    target_mode: str,
    target_element: Optional[str] = None,
) -> Tuple[float, str]:
    """
    Вычисляет точный заряд выбранного иона для достижения полной электронейтральности ячейки (Q_total = 0).
    target_mode:
      - "TET"  : менять заряд target_element только в тетраэдрах A
      - "OCT"  : менять заряд target_element только в октаэдрах B
      - "BOTH" : установить единый заряд target_element сразу в обеих подрешётках A и B (как для Mn в вариантах 1–3)
      - "ANION": подогнать заряд аниона
    """
    if target_mode == "ANION":
        if n_anion <= 0:
            raise ValueError("В структуре нет анионов для балансировки.")
        q_cations = sum(s.count * s.charge for s in tet_specs) + sum(s.count * s.charge for s in oct_specs)
        new_q = -q_cations / n_anion
        return new_q, f"Новый заряд аниона: {new_q:+.9f} e (на {n_anion} узлов)"

    if not target_element:
        raise ValueError("Не выбран элемент для балансировки заряда.")

    q_rest = n_anion * anion_charge
    n_target = 0

    for s in tet_specs:
        if s.element == target_element and target_mode in {"TET", "BOTH"}:
            n_target += s.count
        else:
            q_rest += s.count * s.charge

    for s in oct_specs:
        if s.element == target_element and target_mode in {"OCT", "BOTH"}:
            n_target += s.count
        else:
            q_rest += s.count * s.charge

    if n_target <= 0:
        raise ValueError(f"Элемент «{target_element}» с ненулевым числом атомов не найден в выбранной подрешётке.")

    new_q = -q_rest / n_target
    scope_txt = {
        "TET": "в тетраэдрах A",
        "OCT": "в октаэдрах B",
        "BOTH": "в обеих подрешётках A + B",
    }[target_mode]
    return new_q, f"Точный заряд {target_element} ({scope_txt}, всего {n_target} атомов): {new_q:+.9f} e"


def apply_balanced_charge_to_dfs(
    tet_df: pd.DataFrame,
    oct_df: pd.DataFrame,
    anion_charge: float,
    new_q: float,
    target_mode: str,
    target_element: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Применяет вычисленный компенсирующий заряд к таблицам подрешёток или аниону."""
    new_tet = tet_df.copy()
    new_oct = oct_df.copy()
    new_an_q = float(anion_charge)
    rounded_q = round(float(new_q), 9)

    if target_mode == "ANION":
        new_an_q = rounded_q
    else:
        if target_mode in {"TET", "BOTH"} and not new_tet.empty:
            mask_t = new_tet["Элемент"].astype(str).str.strip() == str(target_element)
            new_tet.loc[mask_t, "Заряд (e)"] = rounded_q
        if target_mode in {"OCT", "BOTH"} and not new_oct.empty:
            mask_o = new_oct["Элемент"].astype(str).str.strip() == str(target_element)
            new_oct.loc[mask_o, "Заряд (e)"] = rounded_q

    return new_tet, new_oct, new_an_q


def build_binary_ab_tables(
    n_tet: int,
    n_oct: int,
    mix_in: str,
    fixed_el: str,
    fixed_q: float,
    comp1_el: str,
    comp1_q: float,
    comp2_el: str,
    comp2_q: float,
    comp1_percent: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Формирует готовые таблицы A и B для произвольной точки бинарного твёрдого раствора."""
    frac = min(max(float(comp1_percent) / 100.0, 0.0), 1.0)
    if mix_in == "OCT":
        c1 = int(round(frac * n_oct))
        c2 = n_oct - c1
        tet_specs = [SpeciesSpec(fixed_el, n_tet, fixed_q)]
        oct_specs = []
        if c1 > 0:
            oct_specs.append(SpeciesSpec(comp1_el, c1, comp1_q))
        if c2 > 0:
            oct_specs.append(SpeciesSpec(comp2_el, c2, comp2_q))
        key = sanitize_prefix(f"bin_{comp1_el}{c1}_{comp2_el}{c2}")
        title = f"Бинарный раствор в октаэдрах: {c1} {comp1_el} / {c2} {comp2_el} ({100*c1/max(n_oct,1):.1f}% / {100*c2/max(n_oct,1):.1f}%)"
    else:
        c1 = int(round(frac * n_tet))
        c2 = n_tet - c1
        tet_specs = []
        if c1 > 0:
            tet_specs.append(SpeciesSpec(comp1_el, c1, comp1_q))
        if c2 > 0:
            tet_specs.append(SpeciesSpec(comp2_el, c2, comp2_q))
        oct_specs = [SpeciesSpec(fixed_el, n_oct, fixed_q)]
        key = sanitize_prefix(f"bin_tet_{comp1_el}{c1}_{comp2_el}{c2}")
        title = f"Бинарный раствор в тетраэдрах: {c1} {comp1_el} / {c2} {comp2_el} ({100*c1/max(n_tet,1):.1f}% / {100*c2/max(n_tet,1):.1f}%)"

    return species_to_dataframe(tet_specs), species_to_dataframe(oct_specs), key, title


def compute_configurational_entropy(sim: SimulationResult) -> Tuple[float, float]:
    """
    Вычисляет идеальную конфигурационную энтропию смешения на 1 формульную единицу AB2X4:
    ΔS_conf / R = - sum_i x_{i,A} ln(x_{i,A}) - 2 * sum_j x_{j,B} ln(x_{j,B}).
    Возвращает (S_в_единицах_R, S_в_Дж_моль_К).
    """
    def _sub_entropy(specs: Sequence[SpeciesSpec], n_sites: int) -> float:
        if n_sites <= 0:
            return 0.0
        counts: Dict[str, int] = {}
        for s in specs:
            if s.count > 0:
                counts[s.element] = counts.get(s.element, 0) + s.count
        val = 0.0
        for c in counts.values():
            p = c / n_sites
            if p > 1e-12:
                val -= p * math.log(p)
        return val

    s_a = _sub_entropy(sim.config.active_tet(), sim.validation.n_tet_expected)
    s_b = _sub_entropy(sim.config.active_oct(), sim.validation.n_oct_expected)
    s_in_r = s_a + 2.0 * s_b
    return s_in_r, s_in_r * GAS_CONSTANT_R


def package_single_result(
    sim: SimulationResult,
    prefix: str,
    gulp_header: str = "single",
) -> Dict[str, Any]:
    """
    Формирует все артефакты одного расчёта в оперативной памяти (и сразу закрывает Figure Matplotlib,
    чтобы исключить утечки памяти при долгой работе веб-сервера).
    """
    clean_pref = sanitize_prefix(prefix)
    fig, zip_bytes = build_zip_package(sim, prefix=clean_pref)
    plt.close(fig)

    gin_text = to_gulp_gin(sim, keyword_header=gulp_header.strip() or "single")
    cif_text = to_vesta_cif(sim)
    txt_table = to_coordinate_txt(sim)

    # Извлекаем PNG и DOCX из готового архива (и при необходимости обновляем заголовок .gin)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf_in:
        png_bytes = zf_in.read(f"{clean_pref}_histograms.png")
        docx_bytes = zf_in.read(f"{clean_pref}_table.docx")
        if gulp_header.strip() != "single":
            updated_buf = io.BytesIO()
            with zipfile.ZipFile(updated_buf, "w", zipfile.ZIP_DEFLATED) as zf_out:
                for member in zf_in.namelist():
                    if member == f"{clean_pref}.gin":
                        zf_out.writestr(member, gin_text)
                    else:
                        zf_out.writestr(member, zf_in.read(member))
            zip_bytes = updated_buf.getvalue()

    dfs = build_summary_dataframes(sim)
    s_r, s_j = compute_configurational_entropy(sim)

    return {
        "sim": sim,
        "prefix": clean_pref,
        "gulp_header": gulp_header.strip() or "single",
        "zip_bytes": zip_bytes,
        "png_bytes": png_bytes,
        "docx_bytes": docx_bytes,
        "gin_text": gin_text,
        "cif_text": cif_text,
        "txt_table": txt_table,
        "dfs": dfs,
        "entropy_R": s_r,
        "entropy_J": s_j,
    }


def run_batch_series(
    base_structure: StructureTemplate,
    configs: Sequence[CompositionConfig],
    max_tries: int = 15000,
    target_chi2: float = 0.45,
    seed: Optional[int] = None,
    gulp_header: str = "single",
    use_uploaded_lattice: bool = False,
) -> Dict[str, Any]:
    """
    Пакетно рассчитывает серию составов и собирает единый мастер-архив ZIP
    с отдельной папкой под каждый вариант и общей сводной таблицей series_summary.csv.
    """
    master_buf = io.BytesIO()
    summary_rows: List[Dict[str, Any]] = []
    packages: List[Dict[str, Any]] = []

    with zipfile.ZipFile(master_buf, "w", zipfile.ZIP_DEFLATED) as master_zf:
        for idx, orig_cfg in enumerate(configs):
            cfg = CompositionConfig.from_dict(orig_cfg.to_dict())
            if use_uploaded_lattice:
                cfg.lattice = tuple(float(v) for v in base_structure.lattice.tolist())
                cfg.angles = tuple(float(v) for v in base_structure.angles.tolist())
                tet_sh = analyze_coordination_shells(base_structure, "TET", max_r=9.0)
                oct_sh = analyze_coordination_shells(base_structure, "OCT", max_r=9.0)
                cfg.r_tet = float(tet_sh["suggested_cutoff"])
                cfg.r_oct = float(oct_sh["suggested_cutoff"])

            run_seed = None if seed is None else int(seed) + idx * 1009
            sim = run_simulation(
                base_structure,
                cfg,
                max_tries=max_tries,
                target_chi2=target_chi2,
                seed=run_seed,
                strict_neutrality=False,
            )
            pkg = package_single_result(sim, prefix=cfg.key, gulp_header=gulp_header)
            packages.append(pkg)

            with zipfile.ZipFile(io.BytesIO(pkg["zip_bytes"]), "r") as sub_zf:
                for name in sub_zf.namelist():
                    master_zf.writestr(f"{pkg['prefix']}/{name}", sub_zf.read(name))

            summary_rows.append({
                "Код": cfg.key,
                "Название": cfg.title,
                "Формула суперъячейки": sim.validation.formula_summary,
                "χ² (Тетраэдры A)": round(sim.tet_result.chi2_total, 4),
                "χ² (Октаэдры B)": round(sim.oct_result.chi2_total, 4),
                "Суммарный заряд Q (e)": round(sim.validation.q_total, 6),
                "Объём V (Å³)": round(sim.validation.volume_ang3, 2),
                "Плотность ρ (г/см³)": (
                    round(sim.validation.density_g_cm3, 4)
                    if sim.validation.density_g_cm3 is not None
                    else None
                ),
                "ΔS_conf / R": round(pkg["entropy_R"], 4),
            })

        summary_df = pd.DataFrame(summary_rows)
        master_zf.writestr("series_summary.csv", summary_df.to_csv(index=False).encode("utf-8-sig"))

    master_buf.seek(0)
    return {
        "summary_df": summary_df,
        "zip_bytes": master_buf.getvalue(),
        "packages": packages,
    }


# =============================================================================
# Веб-интерфейс Streamlit
# =============================================================================

def run_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Spinel Studio — Генератор суперъячеек шпинелей",
        page_icon="💎",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    def ui_full_width(fn, *args, **kwargs):
        """Обеспечивает растягивание виджета на всю ширину без предупреждений в любых версиях Streamlit."""
        ver_parts = [int(x) for x in st.__version__.split(".")[:2] if x.isdigit()]
        if tuple(ver_parts) >= (1, 49):
            try:
                return fn(*args, width="stretch", **kwargs)
            except TypeError:
                pass
        try:
            return fn(*args, use_container_width=True, **kwargs)
        except TypeError:
            return fn(*args, **kwargs)

    # Ненавязчивое научное оформление карточек и заголовков (совместимо со светлой и тёмной темой)
    st.markdown(
        """
        <style>
        .step-box {
            border: 1px solid rgba(128, 128, 128, 0.28);
            border-radius: 10px;
            padding: 0.85rem 1.1rem;
            margin-bottom: 0.75rem;
            background-color: rgba(128, 128, 128, 0.04);
        }
        .step-badge {
            display: inline-block;
            background-color: #1f5aa6;
            color: #ffffff;
            font-weight: 600;
            font-size: 0.82rem;
            padding: 0.18rem 0.62rem;
            border-radius: 6px;
            margin-bottom: 0.35rem;
        }
        .hint-muted {
            font-size: 0.90rem;
            opacity: 0.85;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    presets_map = get_builtin_presets()

    # Инициализация состояния сессии при первом открытии
    def load_config_into_state(cfg: CompositionConfig) -> None:
        st.session_state["cfg_version"] = st.session_state.get("cfg_version", 0) + 1
        st.session_state["active_key"] = cfg.key
        st.session_state["cfg_title"] = cfg.title
        st.session_state["cfg_desc"] = cfg.description
        st.session_state["tet_df"] = species_to_dataframe(cfg.tet_species)
        st.session_state["oct_df"] = species_to_dataframe(cfg.oct_species)
        st.session_state["anion_el"] = cfg.anion_element
        st.session_state["anion_q"] = float(cfg.anion_charge)
        st.session_state["lat_a"] = float(cfg.lattice[0])
        st.session_state["lat_b"] = float(cfg.lattice[1])
        st.session_state["lat_c"] = float(cfg.lattice[2])
        st.session_state["ang_al"] = float(cfg.angles[0])
        st.session_state["ang_be"] = float(cfg.angles[1])
        st.session_state["ang_ga"] = float(cfg.angles[2])
        st.session_state["sc_nx"] = int(cfg.supercell_dims[0])
        st.session_state["sc_ny"] = int(cfg.supercell_dims[1])
        st.session_state["sc_nz"] = int(cfg.supercell_dims[2])
        st.session_state["r_tet"] = float(cfg.r_tet)
        st.session_state["r_oct"] = float(cfg.r_oct)
        st.session_state["is_model_labels"] = bool(cfg.is_model_labels)
        st.session_state["file_prefix"] = sanitize_prefix(cfg.key)

    if "cfg_version" not in st.session_state:
        load_config_into_state(presets_map["variant_1"])
        st.session_state["last_loaded_preset"] = "variant_1"
        st.session_state["uploaded_struct"] = None
        st.session_state["uploaded_struct_name"] = None
        st.session_state["last_result"] = None
        st.session_state["batch_result"] = None

    # Шапка приложения
    st.title("💎 Spinel Studio — Конструктор неупорядоченных суперъячеек шпинелей")
    st.markdown(
        "Генерация статистически равномерных катионных раскладок в структуре шпинели ($AB_2X_4$) **строго раздельно "
        "по тетраэдрической ($A$, $CN=4$) и октаэдрической ($B$, $CN=6$) подрешёткам** по критерию $\\chi^2$. "
        "Готовые выходные файлы для **GULP** (`.gin`), **VESTA** (`.cif`), **Word** (`.docx`) и публикации (`300 dpi .png`, `.csv`)."
    )

    tab_single, tab_batch, tab_guide = st.tabs([
        "🔬 1. Конструктор и расчёт одного состава",
        "📦 2. Серийный расчёт (несколько составов сразу)",
        "📖 3. Инструкция для геологов (Как пользоваться)",
    ])

    # =========================================================================
    # ВКЛАДКА 1: ОСНОВНОЙ ПОШАГОВЫЙ КОНСТРУКТОР
    # =========================================================================
    with tab_single:
        # ---------------------------------------------------------------------
        # ШАГ 1. ВЫБОР ТИПА ЗАДАЧИ И ШАБЛОНА СОСТАВА
        # ---------------------------------------------------------------------
        st.markdown(
            '<div class="step-box"><span class="step-badge">ШАГ 1 из 4</span> '
            '<b>Выберите тип задачи (стартовый шаблон состава)</b><br>'
            '<span class="hint-muted">Любой выбранный шаблон заполняет таблицы ниже — после этого вы можете '
            'свободно менять элементы, их количества и заряды вручную.</span></div>',
            unsafe_allow_html=True,
        )

        TASK_MODES = [
            "🧪 Реальные многокомпонентные шпинели (химические элементы: Co, Ni, Mn, Fe, Cr и др.)",
            "⚖️ Бинарный твёрдый раствор (смешение двух компонентов типа A–B в одной подрешётке)",
            "🧩 Модельная многокомпонентная система (условные катионы A, B, C, D, E)",
            "✨ Свой состав с нуля / Загрузка сохранённого рецепта (.json)",
        ]
        selected_mode = st.radio(
            "Какую кристаллохимическую систему вы моделируете?",
            options=TASK_MODES,
            index=0,
        )

        if selected_mode == TASK_MODES[0]:
            real_keys = ["variant_1", "variant_2", "variant_3"]
            c_sel, c_btn = st.columns([3, 1])
            with c_sel:
                chosen_real = st.selectbox(
                    "Готовые варианты расселения высокоэнтропийной шпинели Co77 Ni77 Mn77 Fe77 Cr76 O512:",
                    options=real_keys,
                    format_func=lambda k: presets_map[k].title,
                )
                st.caption(f"ℹ️ {presets_map[chosen_real].description}")
            with c_btn:
                st.write("")
                if ui_full_width(st.button, "📥 Загрузить этот вариант", key="btn_load_real"):
                    load_config_into_state(presets_map[chosen_real])
                    st.session_state["last_loaded_preset"] = chosen_real
                    st.rerun()
            if st.session_state.get("last_loaded_preset") != chosen_real and st.session_state.get("auto_mode_prev") == selected_mode:
                load_config_into_state(presets_map[chosen_real])
                st.session_state["last_loaded_preset"] = chosen_real
                st.rerun()

        elif selected_mode == TASK_MODES[1]:
            st.info(
                "**Что такое бинарный режим (тип A–B)?** Одна подрешётка целиком занята одним элементом "
                "(например, все 128 тетраэдров заняты `Mg`), а во второй подрешётке смешиваются два катиона "
                "(условные `A` и `B` либо реальные металлы, например `Cr` и `Al`) в заданной пропорции."
            )
            ab_keys = ["ab_32_224", "ab_64_192", "ab_85_171", "ab_128_128"]
            c_ab_sel, c_ab_btn = st.columns([3, 1])
            with c_ab_sel:
                chosen_ab = st.selectbox(
                    "Выберите стандартную точку бинарной серии AB:",
                    options=ab_keys,
                    format_func=lambda k: presets_map[k].title,
                )
            with c_ab_btn:
                st.write("")
                if ui_full_width(st.button, "📥 Загрузить точку AB", key="btn_load_ab"):
                    load_config_into_state(presets_map[chosen_ab])
                    st.session_state["last_loaded_preset"] = chosen_ab
                    st.rerun()
            if st.session_state.get("last_loaded_preset") != chosen_ab and st.session_state.get("auto_mode_prev") == selected_mode:
                load_config_into_state(presets_map[chosen_ab])
                st.session_state["last_loaded_preset"] = chosen_ab
                st.rerun()

            with st.expander("🛠️ Или задайте любое своё соотношение A : B в процентах (конструктор бинарной точки)"):
                bc1, bc2, bc3, bc4 = st.columns(4)
                with bc1:
                    mix_target = st.selectbox(
                        "Где смешиваем 2 компонента?",
                        options=["OCT", "TET"],
                        format_func=lambda v: "В октаэдрах B (тетраэдры фиксированы)" if v == "OCT" else "В тетраэдрах A (октаэдры фиксированы)",
                    )
                    fixed_sym = st.text_input("Фиксированный катион", value="Mg")
                    fixed_chg = st.number_input("Заряд фиксированного катиона", value=1.2 if mix_target == "OCT" else 1.8, format="%.6f")
                with bc2:
                    c1_sym = st.text_input("Символ Компонента 1", value="A")
                    c1_chg = st.number_input("Заряд Компонента 1", value=1.8 if mix_target == "OCT" else 1.2, format="%.6f")
                with bc3:
                    c2_sym = st.text_input("Символ Компонента 2", value="B")
                    c2_chg = st.number_input("Заряд Компонента 2", value=1.8 if mix_target == "OCT" else 1.2, format="%.6f")
                with bc4:
                    c1_pct = st.slider("Доля Компонента 1 (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.5)
                    st.write("")
                    if ui_full_width(st.button, "✨ Применить пропорцию в таблицы", key="btn_apply_custom_ab"):
                        cur_ntet = st.session_state.get("current_n_tet", 128)
                        cur_noct = st.session_state.get("current_n_oct", 256)
                        t_df, o_df, b_key, b_title = build_binary_ab_tables(
                            cur_ntet, cur_noct, mix_target,
                            fixed_sym.strip() or "Mg", fixed_chg,
                            c1_sym.strip() or "A", c1_chg,
                            c2_sym.strip() or "B", c2_chg,
                            c1_pct,
                        )
                        st.session_state["tet_df"] = t_df
                        st.session_state["oct_df"] = o_df
                        st.session_state["cfg_title"] = b_title
                        st.session_state["file_prefix"] = b_key
                        all_syms = {fixed_sym.strip(), c1_sym.strip(), c2_sym.strip()}
                        st.session_state["is_model_labels"] = not all(s in ATOMIC_MASSES for s in all_syms)
                        st.session_state["cfg_version"] += 1
                        st.rerun()

        elif selected_mode == TASK_MODES[2]:
            c_m1, c_m2 = st.columns([3, 1])
            with c_m1:
                st.info(
                    f"**{presets_map['abcde_5comp'].title}** — {presets_map['abcde_5comp'].description} "
                    "Вы можете добавлять любые новые буквенные обозначения (F, G...) в таблицах ниже."
                )
            with c_m2:
                st.write("")
                if ui_full_width(st.button, "📥 Загрузить шаблон ABCDE", key="btn_load_abcde"):
                    load_config_into_state(presets_map["abcde_5comp"])
                    st.session_state["last_loaded_preset"] = "abcde_5comp"
                    st.rerun()
            if st.session_state.get("auto_mode_prev") != selected_mode:
                load_config_into_state(presets_map["abcde_5comp"])
                st.session_state["last_loaded_preset"] = "abcde_5comp"
                st.rerun()

        else:
            cc1, cc2 = st.columns([1, 2])
            with cc1:
                st.write("Начать с чистого шаблона нормальной шпинели:")
                if ui_full_width(st.button, "🧹 Сбросить к базовой шпинели MgCr2O4", key="btn_reset_custom"):
                    load_config_into_state(presets_map["custom"])
                    st.session_state["last_loaded_preset"] = "custom"
                    st.rerun()
            with cc2:
                uploaded_json = st.file_uploader(
                    "Или загрузите ранее сохранённый файл рецепта состава (*_config.json):",
                    type=["json"],
                    key="json_cfg_uploader",
                )
                if uploaded_json is not None:
                    json_sig = f"{uploaded_json.name}_{uploaded_json.size}"
                    if st.session_state.get("last_json_sig") != json_sig:
                        try:
                            loaded_cfg = CompositionConfig.from_json(uploaded_json.getvalue().decode("utf-8"))
                            load_config_into_state(loaded_cfg)
                            st.session_state["last_json_sig"] = json_sig
                            st.success(f"Конфигурация «{loaded_cfg.title}» успешно загружена!")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Не удалось прочитать JSON-конфиг: {exc}")

        st.session_state["auto_mode_prev"] = selected_mode
        st.divider()

        # ---------------------------------------------------------------------
        # ШАГ 2. КРИСТАЛЛИЧЕСКИЙ КАРКАС И ПАРАМЕТРЫ ЯЧЕЙКИ
        # ---------------------------------------------------------------------
        st.markdown(
            '<div class="step-box"><span class="step-badge">ШАГ 2 из 4</span> '
            '<b>Кристаллический каркас и параметры суперъячейки</b><br>'
            '<span class="hint-muted">Используйте встроенную идеальную решётку шпинели (внешний файл не требуется) '
            'или загрузите собственный файл координат (.docx, .txt, .gin, .out, .cif).</span></div>',
            unsafe_allow_html=True,
        )

        struct_source = st.radio(
            "Источник координат узлов кристаллической решётки:",
            options=[
                "📐 Встроенная идеальная решётка шпинели (пр. гр. Fd-3m)",
                "📂 Свой файл структуры от геологов (.docx, .txt, .gin, .out, .cif)",
            ],
            horizontal=True,
        )

        base_struct: Optional[StructureTemplate] = None

        if struct_source.startswith("📐"):
            sc_col1, sc_col2 = st.columns([1.2, 1.8])
            with sc_col1:
                dim_preset = st.selectbox(
                    "Размерность суперъячейки (nx × ny × nz):",
                    options=[
                        "2×2×4 (стандартная: 896 атомов = 128 A + 256 B + 512 O)",
                        "2×2×2 (средняя: 448 атомов = 64 A + 128 B + 256 O)",
                        "1×1×1 (элементарная ячейка: 56 атомов = 8 A + 16 B + 32 O)",
                        "3×3×3 (крупная: 1512 атомов = 216 A + 432 B + 864 O)",
                        "Своя размерность (nx, ny, nz)...",
                    ],
                    index=0,
                )
                if dim_preset.startswith("2×2×4"):
                    nx, ny, nz = 2, 2, 4
                elif dim_preset.startswith("2×2×2"):
                    nx, ny, nz = 2, 2, 2
                elif dim_preset.startswith("1×1×1"):
                    nx, ny, nz = 1, 1, 1
                elif dim_preset.startswith("3×3×3"):
                    nx, ny, nz = 3, 3, 3
                else:
                    d1, d2, d3 = st.columns(3)
                    nx = int(d1.number_input("nx", min_value=1, max_value=8, value=int(st.session_state["sc_nx"]), step=1))
                    ny = int(d2.number_input("ny", min_value=1, max_value=8, value=int(st.session_state["sc_ny"]), step=1))
                    nz = int(d3.number_input("nz", min_value=1, max_value=8, value=int(st.session_state["sc_nz"]), step=1))

                if (nx, ny, nz) != (st.session_state["sc_nx"], st.session_state["sc_ny"], st.session_state["sc_nz"]):
                    # При смене кратности пересчитываем линейные размеры пропорционально a0 ~ 8.299 Å
                    a0_est = st.session_state["lat_a"] / max(st.session_state["sc_nx"], 1)
                    st.session_state["sc_nx"], st.session_state["sc_ny"], st.session_state["sc_nz"] = nx, ny, nz
                    st.session_state["lat_a"] = round(nx * a0_est, 4)
                    st.session_state["lat_b"] = round(ny * a0_est, 4)
                    st.session_state["lat_c"] = round(nz * a0_est, 4)

            with sc_col2:
                if (nx, ny, nz) == (2, 2, 4):
                    b_t1, b_t2, _ = st.columns([1, 1, 0.6])
                    if b_t1.button("Шаблон NiCr2O4 (16.598 Å)", key="lat_btn_ni"):
                        st.session_state["lat_a"], st.session_state["lat_b"], st.session_state["lat_c"] = (16.598, 16.598, 33.196)
                        st.session_state["r_tet"], st.session_state["r_oct"] = 3.90, 3.25
                        st.rerun()
                    if b_t2.button("Шаблон MgCr2O4 (16.674 Å)", key="lat_btn_mg"):
                        st.session_state["lat_a"], st.session_state["lat_b"], st.session_state["lat_c"] = (16.674, 16.674, 33.348)
                        st.session_state["r_tet"], st.session_state["r_oct"] = 3.95, 3.25
                        st.rerun()

                lc1, lc2, lc3 = st.columns(3)
                lat_a = lc1.number_input("Параметр a (Å)", min_value=3.0, max_value=200.0, value=float(st.session_state["lat_a"]), step=0.01, format="%.4f")
                lat_b = lc2.number_input("Параметр b (Å)", min_value=3.0, max_value=200.0, value=float(st.session_state["lat_b"]), step=0.01, format="%.4f")
                lat_c = lc3.number_input("Параметр c (Å)", min_value=3.0, max_value=200.0, value=float(st.session_state["lat_c"]), step=0.01, format="%.4f")
                st.session_state["lat_a"], st.session_state["lat_b"], st.session_state["lat_c"] = lat_a, lat_b, lat_c

            base_struct = build_ideal_spinel_supercell(
                nx=nx, ny=ny, nz=nz,
                lattice=(lat_a, lat_b, lat_c),
                angles=(st.session_state["ang_al"], st.session_state["ang_be"], st.session_state["ang_ga"]),
            )
            st.caption(
                f"📐 Сформирован идеальный каркас **Fd-3m ({nx}×{ny}×{nz})**: всего **{base_struct.n_total}** узлов — "
                f"**{base_struct.n_tet}** тетраэдров (A), **{base_struct.n_oct}** октаэдров (B), **{base_struct.n_anion}** анионов."
            )

        else:
            up_col1, up_col2 = st.columns([1.6, 1.4])
            with up_col1:
                up_file = st.file_uploader(
                    "Перетащите сюда файл структуры (.docx, .txt, .gin, .out, .cif):",
                    type=["docx", "txt", "gin", "out", "cif"],
                    key="struct_file_uploader",
                )
                with st.expander("Дополнительные подсказки символов (только если в файле нет анионов или структура сильно искажена)"):
                    hc1, hc2, hc3 = st.columns(3)
                    hint_tet_raw = hc1.text_input("Символы TET через запятую", value="", placeholder="например: Ni, Mg")
                    hint_oct_raw = hc2.text_input("Символы OCT через запятую", value="", placeholder="например: Cr, Al")
                    hint_an_raw = hc3.text_input("Символы анионов", value="", placeholder="например: O, S")

            if up_file is not None:
                tet_h = {s.strip() for s in hint_tet_raw.split(",") if s.strip()} or None
                oct_h = {s.strip() for s in hint_oct_raw.split(",") if s.strip()} or None
                an_h = {s.strip() for s in hint_an_raw.split(",") if s.strip()} or None
                file_sig = f"{up_file.name}_{up_file.size}_{hint_tet_raw}_{hint_oct_raw}_{hint_an_raw}"
                try:
                    if st.session_state.get("uploaded_struct_sig") != file_sig:
                        parsed = parse_structure_file(
                            up_file.getvalue(), up_file.name,
                            tet_hint=tet_h, oct_hint=oct_h, anion_hint=an_h,
                        )
                        st.session_state["uploaded_struct"] = parsed
                        st.session_state["uploaded_struct_sig"] = file_sig
                        # Автоматически подхватываем параметры решётки из загруженного файла
                        st.session_state["lat_a"] = float(parsed.lattice[0])
                        st.session_state["lat_b"] = float(parsed.lattice[1])
                        st.session_state["lat_c"] = float(parsed.lattice[2])
                        st.session_state["ang_al"] = float(parsed.angles[0])
                        st.session_state["ang_be"] = float(parsed.angles[1])
                        st.session_state["ang_ga"] = float(parsed.angles[2])
                    base_struct = st.session_state["uploaded_struct"]
                except Exception as exc:
                    st.error(f"Ошибка разбора файла «{up_file.name}»: {exc}")
                    base_struct = None

            with up_col2:
                if base_struct is not None:
                    st.success(
                        f"✅ Файл **{base_struct.source_name}** прочитан!\n\n"
                        f"- Всего атомов: **{base_struct.n_total}**\n"
                        f"- Позиций A (тетраэдры, CN=4): **{base_struct.n_tet}**\n"
                        f"- Позиций B (октаэдры, CN=6): **{base_struct.n_oct}**\n"
                        f"- Позиций анионов: **{base_struct.n_anion}**"
                    )
                    ul1, ul2, ul3 = st.columns(3)
                    lat_a = ul1.number_input("a (Å)", value=float(st.session_state["lat_a"]), format="%.4f", key="u_lat_a")
                    lat_b = ul2.number_input("b (Å)", value=float(st.session_state["lat_b"]), format="%.4f", key="u_lat_b")
                    lat_c = ul3.number_input("c (Å)", value=float(st.session_state["lat_c"]), format="%.4f", key="u_lat_c")
                    st.session_state["lat_a"], st.session_state["lat_b"], st.session_state["lat_c"] = lat_a, lat_b, lat_c
                    base_struct = base_struct.with_lattice(
                        (lat_a, lat_b, lat_c),
                        (st.session_state["ang_al"], st.session_state["ang_be"], st.session_state["ang_ga"]),
                    )
                else:
                    st.warning("Пожалуйста, загрузите файл структуры слева (или переключитесь на встроенную идеальную решётку).")

        # Если файл ещё не загружен в режиме загрузки, подставляем временный каркас для отображения таблиц
        preview_struct = base_struct if base_struct is not None else build_ideal_spinel_supercell(2, 2, 4)
        st.session_state["current_n_tet"] = preview_struct.n_tet
        st.session_state["current_n_oct"] = preview_struct.n_oct

        st.divider()

        # ---------------------------------------------------------------------
        # ШАГ 3. ЗАСЕЛЕНИЕ ПОДРЕШЁТОК A И B, АНИОН И ЗАРЯДЫ
        # ---------------------------------------------------------------------
        st.markdown(
            '<div class="step-box"><span class="step-badge">ШАГ 3 из 4</span> '
            '<b>Заселение тетраэдрической (A) и октаэдрической (B) подрешёток и заряды ионов</b><br>'
            '<span class="hint-muted">Подрешётки A и B перемешиваются <b>независимо друг от друга</b>. '
            'Можно вводить любые символы из таблицы Менделеева (Co, Ni, Zn, Ti, Al, Ga...) или модельные метки (A, B, C...). '
            'Для добавления нового элемента кликните на пустую строку со значком «+» внизу нужной таблицы.</span></div>',
            unsafe_allow_html=True,
        )

        ver = st.session_state["cfg_version"]
        col_tet, col_oct = st.columns(2)

        col_cfg = {
            "Элемент": st.column_config.TextColumn(
                "Элемент / Метка",
                help="Химический символ (Co, Fe, Mg, Zn...) или модельная метка (A, B, C...)",
                required=True,
            ),
            "Количество": st.column_config.NumberColumn(
                "Количество атомов",
                help="Целое число узлов в этой подрешётке (либо доли/проценты с последующим нажатием кнопки пересчёта ниже)",
                min_value=0,
                step=1,
                format="%d",
                required=True,
            ),
            "Заряд (e)": st.column_config.NumberColumn(
                "Заряд иона (e)",
                help="Эффективный или формальный заряд (например: 1.2, 1.8 или 1.802597403)",
                required=True,
            ),
        }

        with col_tet:
            st.subheader(f"🔺 Подрешётка A — Тетраэдры (CN = 4, мест: {preview_struct.n_tet})")
            edited_tet_df = ui_full_width(
                st.data_editor,
                st.session_state["tet_df"],
                num_rows="dynamic",
                column_config=col_cfg,
                hide_index=True,
                key=f"tet_editor_{ver}",
            )
            tet_specs_now, tet_errs = dataframe_to_species(edited_tet_df, "Тетраэдры (A)")
            sum_tet = sum(s.count for s in tet_specs_now)
            diff_tet = preview_struct.n_tet - sum_tet
            if sum_tet == preview_struct.n_tet and not tet_errs:
                st.success(f"✅ Заполнено точно: **{sum_tet} из {preview_struct.n_tet}** мест")
            else:
                hint_d = f"добавьте ещё {diff_tet}" if diff_tet > 0 else f"уберите лишние {-diff_tet}"
                st.warning(f"⚠️ Сейчас в таблице **{sum_tet} из {preview_struct.n_tet}** атомов ({hint_d}).")
            norm_tet_clicked = ui_full_width(
                st.button,
                f"⚖️ Пересчитать введённые числа/проценты пропорционально на {preview_struct.n_tet} мест",
                key="btn_norm_tet",
            )

        with col_oct:
            st.subheader(f"🔷 Подрешётка B — Октаэдры (CN = 6, мест: {preview_struct.n_oct})")
            edited_oct_df = ui_full_width(
                st.data_editor,
                st.session_state["oct_df"],
                num_rows="dynamic",
                column_config=col_cfg,
                hide_index=True,
                key=f"oct_editor_{ver}",
            )
            oct_specs_now, oct_errs = dataframe_to_species(edited_oct_df, "Октаэдры (B)")
            sum_oct = sum(s.count for s in oct_specs_now)
            diff_oct = preview_struct.n_oct - sum_oct
            if sum_oct == preview_struct.n_oct and not oct_errs:
                st.success(f"✅ Заполнено точно: **{sum_oct} из {preview_struct.n_oct}** мест")
            else:
                hint_o = f"добавьте ещё {diff_oct}" if diff_oct > 0 else f"уберите лишние {-diff_oct}"
                st.warning(f"⚠️ Сейчас в таблице **{sum_oct} из {preview_struct.n_oct}** атомов ({hint_o}).")
            norm_oct_clicked = ui_full_width(
                st.button,
                f"⚖️ Пересчитать введённые числа/проценты пропорционально на {preview_struct.n_oct} мест",
                key="btn_norm_oct",
            )

        if norm_tet_clicked:
            st.session_state["tet_df"] = scale_counts_proportionally(edited_tet_df, preview_struct.n_tet)
            st.session_state["oct_df"] = edited_oct_df
            st.session_state["cfg_version"] += 1
            st.rerun()

        if norm_oct_clicked:
            st.session_state["tet_df"] = edited_tet_df
            st.session_state["oct_df"] = scale_counts_proportionally(edited_oct_df, preview_struct.n_oct)
            st.session_state["cfg_version"] += 1
            st.rerun()

        # Анионная подрешётка и метаданные расчёта
        st.markdown("#### Анионная подрешётка и подпись файлов")
        an_c1, an_c2, an_c3, an_c4 = st.columns([1.0, 1.1, 1.9, 1.3])
        with an_c1:
            anion_el = st.text_input(
                f"Анион ({preview_struct.n_anion} мест)",
                value=st.session_state["anion_el"],
                help="По умолчанию кислород O; для тиошпинелей и селеношпинелей можно указать S, Se и т.д.",
            ).strip() or "O"
            st.session_state["anion_el"] = anion_el
        with an_c2:
            anion_q = st.number_input(
                "Заряд аниона (e)",
                value=float(st.session_state["anion_q"]),
                step=0.05,
                format="%.6f",
            )
            st.session_state["anion_q"] = float(anion_q)
        with an_c3:
            cfg_title = st.text_input(
                "Заголовок модели (для отчёта Word)",
                value=st.session_state["cfg_title"],
            )
            st.session_state["cfg_title"] = cfg_title
        with an_c4:
            file_prefix = st.text_input(
                "Имя выходных файлов (латиницей)",
                value=st.session_state["file_prefix"],
                help="Например, variant_1 -> создаст файлы variant_1.gin, variant_1.cif, variant_1_table.docx и т.д.",
            )
            st.session_state["file_prefix"] = sanitize_prefix(file_prefix)

        all_used_symbols = {s.element for s in tet_specs_now} | {s.element for s in oct_specs_now} | {anion_el}
        auto_is_model = not all(sym in ATOMIC_MASSES for sym in all_used_symbols)

        # Собираем живой конфиг и предрасчётный отчёт валидации
        live_cfg = CompositionConfig(
            key=sanitize_prefix(file_prefix),
            title=cfg_title.strip() or "Spinel Supercell",
            description=st.session_state.get("cfg_desc", ""),
            lattice=(float(st.session_state["lat_a"]), float(st.session_state["lat_b"]), float(st.session_state["lat_c"])),
            angles=(float(st.session_state["ang_al"]), float(st.session_state["ang_be"]), float(st.session_state["ang_ga"])),
            supercell_dims=(int(st.session_state["sc_nx"]), int(st.session_state["sc_ny"]), int(st.session_state["sc_nz"])),
            tet_species=tet_specs_now,
            oct_species=oct_specs_now,
            anion_element=anion_el,
            anion_charge=float(anion_q),
            r_tet=float(st.session_state["r_tet"]),
            r_oct=float(st.session_state["r_oct"]),
            is_model_labels=auto_is_model,
        )

        val_report: ValidationReport = validate_setup(preview_struct, live_cfg)

        # Наглядная панель предрасчётной проверки
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Валовая формула ячейки", val_report.formula_summary)
        m2.metric(
            "Сумма катионов (A / B)",
            f"{val_report.n_tet_configured}/{val_report.n_tet_expected} | {val_report.n_oct_configured}/{val_report.n_oct_expected}",
            delta="Норма" if val_report.is_valid_counts else "Не сходится",
            delta_color="normal" if val_report.is_valid_counts else "inverse",
        )
        m3.metric(
            "Суммарный заряд ячейки",
            f"{val_report.q_total:+.6f} e",
            delta="Электронейтральна" if val_report.is_electroneutral else "Дисбаланс заряда",
            delta_color="normal" if val_report.is_electroneutral else "off",
        )
        dens_label = f"{val_report.density_g_cm3:.4f} г/см³" if val_report.density_g_cm3 is not None else "Модельные метки"
        m4.metric("Плотность ρ (d_min узлов)", dens_label, delta=f"d_min = {val_report.min_interatomic_distance:.2f} Å", delta_color="off")

        for err_msg in tet_errs + oct_errs:
            st.error(err_msg)

        # Помощник автоматического выравнивания заряда в 0 (раскрыт автоматически, если заряд не сходится)
        with st.expander(
            "⚡ Помощник электронейтральности: автоматически подогнать заряд выбранного иона под Q = 0",
            expanded=(not val_report.is_electroneutral and val_report.is_valid_counts),
        ):
            st.caption(
                "Если при вводе нового состава суммарный заряд ячейки отличается от нуля, программа может "
                "автоматически вычислить компенсирующий дробный заряд для выбранного катиона (в одной или сразу "
                "в обеих подрешётках, как для марганца Mn = 1.802597403) либо для аниона."
            )
            tet_els = list(dict.fromkeys(s.element for s in tet_specs_now))
            oct_els = list(dict.fromkeys(s.element for s in oct_specs_now))
            both_els = [el for el in tet_els if el in oct_els]

            balance_options: List[Tuple[str, str, Optional[str]]] = []
            for el in both_els:
                balance_options.append((f"Катион {el} сразу в обеих подрешётках (A + B)", "BOTH", el))
            for el in tet_els:
                balance_options.append((f"Катион {el} только в тетраэдрах (A)", "TET", el))
            for el in oct_els:
                balance_options.append((f"Катион {el} только в октаэдрах (B)", "OCT", el))
            balance_options.append((f"Анион {anion_el} ({preview_struct.n_anion} позиций)", "ANION", None))

            bc_sel, bc_info, bc_apply = st.columns([1.6, 1.6, 1.0])
            with bc_sel:
                chosen_bal_idx = st.selectbox(
                    "Чей заряд скорректировать?",
                    options=list(range(len(balance_options))),
                    format_func=lambda i: balance_options[i][0],
                )
            _, b_mode, b_el = balance_options[chosen_bal_idx]
            try:
                calc_q, calc_msg = compute_balanced_charge(
                    tet_specs_now, oct_specs_now, preview_struct.n_anion, float(anion_q), b_mode, b_el
                )
                with bc_info:
                    st.info(calc_msg)
                with bc_apply:
                    st.write("")
                    if ui_full_width(st.button, "✅ Подставить этот заряд", key="btn_apply_charge_balance"):
                        nt_df, no_df, nan_q = apply_balanced_charge_to_dfs(
                            edited_tet_df, edited_oct_df, float(anion_q), calc_q, b_mode, b_el
                        )
                        st.session_state["tet_df"] = nt_df
                        st.session_state["oct_df"] = no_df
                        st.session_state["anion_q"] = nan_q
                        st.session_state["cfg_version"] += 1
                        st.rerun()
            except Exception as exc:
                bc_info.warning(str(exc))

        st.divider()

        # ---------------------------------------------------------------------
        # ШАГ 4. ПАРАМЕТРЫ ПЕРЕМЕШИВАНИЯ И ЗАПУСК РАСЧЁТА
        # ---------------------------------------------------------------------
        st.markdown(
            '<div class="step-box"><span class="step-badge">ШАГ 4 из 4</span> '
            '<b>Параметры 1-й координационной сферы и запуск Монте-Карло оптимизации</b></div>',
            unsafe_allow_html=True,
        )

        tet_shell_info = analyze_coordination_shells(preview_struct, "TET", max_r=9.0)
        oct_shell_info = analyze_coordination_shells(preview_struct, "OCT", max_r=9.0)

        rc1, rc2, rc3 = st.columns([1.2, 1.2, 1.1])
        with rc1:
            r_tet = st.number_input(
                "Радиус 1-й сферы тетраэдров R_tet (Å)",
                min_value=1.5, max_value=15.0, value=float(st.session_state["r_tet"]), step=0.05, format="%.2f",
            )
            st.session_state["r_tet"] = float(r_tet)
            st.caption(
                f"В текущей ячейке: 1-я сфера TET = **{tet_shell_info['first_shell_range'][0]:.2f}–{tet_shell_info['first_shell_range'][1]:.2f} Å** "
                f"(CN={tet_shell_info['expected_cn']}), 2-я сфера от {tet_shell_info['second_shell_start']} Å."
            )
        with rc2:
            r_oct = st.number_input(
                "Радиус 1-й сферы октаэдров R_oct (Å)",
                min_value=1.5, max_value=15.0, value=float(st.session_state["r_oct"]), step=0.05, format="%.2f",
            )
            st.session_state["r_oct"] = float(r_oct)
            st.caption(
                f"В текущей ячейке: 1-я сфера OCT = **{oct_shell_info['first_shell_range'][0]:.2f}–{oct_shell_info['first_shell_range'][1]:.2f} Å** "
                f"(CN={oct_shell_info['expected_cn']}), 2-я сфера от {oct_shell_info['second_shell_start']} Å."
            )
        with rc3:
            st.write("")
            if ui_full_width(st.button, "🎯 Автоподбор радиусов под текущую ячейку", key="btn_auto_radii"):
                st.session_state["r_tet"] = float(tet_shell_info["suggested_cutoff"])
                st.session_state["r_oct"] = float(oct_shell_info["suggested_cutoff"])
                st.rerun()

        with st.expander("⚙️ Тонкие настройки алгоритма Монте-Карло и заголовка GULP (.gin)"):
            adv1, adv2, adv3, adv4 = st.columns(4)
            max_tries = int(adv1.number_input("Максимум попыток Монте-Карло", min_value=500, max_value=200000, value=15000, step=2500))
            target_chi2 = float(adv2.number_input("Порог χ² досрочной остановки", min_value=0.01, max_value=5.0, value=0.45, step=0.05, format="%.2f"))
            use_fixed_seed = adv3.checkbox("Фиксировать случайное зерно (seed)", value=False, help="Включите, чтобы получать идентичную раскладку при повторных запусках")
            seed_val = int(adv3.number_input("Значение seed", min_value=1, max_value=9999999, value=42, step=1, disabled=not use_fixed_seed))
            gulp_mode = adv4.selectbox(
                "Режим расчёта в шапке .gin (GULP)",
                options=[
                    "single",
                    "opti conp",
                    "opti conv",
                ],
                format_func=lambda s: {
                    "single": "single (расчёт энергии без релаксации)",
                    "opti conp": "opti conp (релаксация при P = const)",
                    "opti conv": "opti conv (релаксация при V = const)",
                }[s],
            )
            strict_q = adv4.checkbox("Блокировать расчёт при Q ≠ 0", value=False)

        live_cfg.r_tet = float(st.session_state["r_tet"])
        live_cfg.r_oct = float(st.session_state["r_oct"])

        can_run = (base_struct is not None) and val_report.is_valid_counts and val_report.has_no_overlaps and not (tet_errs or oct_errs)
        run_clicked = ui_full_width(
            st.button,
            "🚀 Запустить раздельное перемешивание A и B и сформировать все файлы",
            type="primary",
            disabled=not can_run,
            key="btn_run_main",
        )

        if run_clicked and base_struct is not None:
            with st.spinner("Идёт статистический подбор конфигураций по критерию χ² и сборка файлов..."):
                try:
                    sim_out = run_simulation(
                        base_structure=base_struct,
                        config=live_cfg,
                        max_tries=max_tries,
                        target_chi2=target_chi2,
                        seed=seed_val if use_fixed_seed else None,
                        strict_neutrality=strict_q,
                    )
                    st.session_state["last_result"] = package_single_result(
                        sim_out,
                        prefix=live_cfg.key,
                        gulp_header=gulp_mode,
                    )
                except Exception as exc:
                    st.error(f"Не удалось выполнить расчёт: {exc}")

        # ---------------------------------------------------------------------
        # БЛОК ОТОБРАЖЕНИЯ И ВЫГРУЗКИ РЕЗУЛЬТАТОВ
        # ---------------------------------------------------------------------
        res_pkg = st.session_state.get("last_result")
        if res_pkg is not None:
            sim_r: SimulationResult = res_pkg["sim"]
            pref_r: str = res_pkg["prefix"]

            st.divider()
            st.subheader(f"✅ Результаты моделирования: {sim_r.config.title}")
            if not sim_r.validation.is_electroneutral:
                st.warning(
                    f"Обратите внимание: суммарный заряд сгенерированной ячейки равен {sim_r.validation.q_total:+.6f} e "
                    "(перед запуском в GULP убедитесь в электронейтральности или воспользуйтесь помощником зарядов выше)."
                )

            rm1, rm2, rm3, rm4 = st.columns(4)
            rm1.metric(
                f"χ² Тетраэдры A (CN={sim_r.tet_result.cn})",
                f"{sim_r.tet_result.chi2_total:.4f}",
                delta=f"попытка {sim_r.tet_result.attempts_used}",
                delta_color="off",
            )
            rm2.metric(
                f"χ² Октаэдры B (CN={sim_r.oct_result.cn})",
                f"{sim_r.oct_result.chi2_total:.4f}",
                delta=f"попытка {sim_r.oct_result.attempts_used}",
                delta_color="off",
            )
            rm3.metric(
                "Конфиг. энтропия ΔS_conf",
                f"{res_pkg['entropy_R']:.3f} R",
                delta=f"{res_pkg['entropy_J']:.2f} Дж/(моль·К)",
                delta_color="off",
            )
            dens_res = f"{sim_r.validation.density_g_cm3:.4f} г/см³" if sim_r.validation.density_g_cm3 is not None else "—"
            rm4.metric(
                "Рентгеновская плотность ρ",
                dens_res,
                delta=f"V = {sim_r.validation.volume_ang3:.1f} Å³",
                delta_color="off",
            )

            # Кнопки скачивания (полный архив + быстрые отдельные файлы)
            ui_full_width(
                st.download_button,
                f"📦 Скачать полный ZIP-архив результатов ({pref_r}.zip — все 9 файлов: GULP, VESTA, Word, PNG, CSV, JSON)",
                data=res_pkg["zip_bytes"],
                file_name=f"{pref_r}.zip",
                mime="application/zip",
                type="primary",
                key="dl_zip_main",
            )

            dl1, dl2, dl3, dl4 = st.columns(4)
            with dl1:
                ui_full_width(
                    st.download_button,
                    f"⬇️ {pref_r}.gin (GULP)",
                    data=res_pkg["gin_text"].encode("utf-8"),
                    file_name=f"{pref_r}.gin",
                    mime="text/plain",
                    key="dl_gin_single",
                )
            with dl2:
                ui_full_width(
                    st.download_button,
                    f"⬇️ {pref_r}.cif (VESTA)",
                    data=res_pkg["cif_text"].encode("utf-8"),
                    file_name=f"{pref_r}.cif",
                    mime="chemical/x-cif",
                    key="dl_cif_single",
                )
            with dl3:
                ui_full_width(
                    st.download_button,
                    f"⬇️ {pref_r}_table.docx (Word)",
                    data=res_pkg["docx_bytes"],
                    file_name=f"{pref_r}_table.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="dl_docx_single",
                )
            with dl4:
                ui_full_width(
                    st.download_button,
                    f"⬇️ {pref_r}_histograms.png (300 dpi)",
                    data=res_pkg["png_bytes"],
                    file_name=f"{pref_r}_histograms.png",
                    mime="image/png",
                    key="dl_png_single",
                )

            # Детальные аналитические вкладки результатов
            rtab_hist, rtab_pairs, rtab_stoich, rtab_files = st.tabs([
                "📊 Гистограммы и критерий χ²",
                "🔗 Парные контакты (Уоррен–Каули α)",
                "🧪 Сводка состава и зарядов",
                "📄 Координаты и предпросмотр .gin / .cif",
            ])

            with rtab_hist:
                ui_full_width(st.image, res_pkg["png_bytes"])
                hc_l, hc_r = st.columns(2)
                with hc_l:
                    st.markdown(f"**Распределение однородных контактов в тетраэдрах A ($\\chi^2 = {sim_r.tet_result.chi2_total:.4f}$)**")
                    ui_full_width(st.dataframe, res_pkg["dfs"]["tet_hist"], hide_index=True)
                with hc_r:
                    st.markdown(f"**Распределение однородных контактов в октаэдрах B ($\\chi^2 = {sim_r.oct_result.chi2_total:.4f}$)**")
                    ui_full_width(st.dataframe, res_pkg["dfs"]["oct_hist"], hide_index=True)

            with rtab_pairs:
                st.caption(
                    "Параметр ближнего порядка Уоррена–Каули $\\alpha_{ij} = 1 - N_{\\text{факт}}/N_{\\text{теор}}$: "
                    "значения около **0** означают идеальное хаотическое перемешивание без слипания катионов."
                )
                pc_l, pc_r = st.columns(2)
                with pc_l:
                    st.markdown("**Парные связи катион–катион в 1-й сфере тетраэдров (A)**")
                    ui_full_width(st.dataframe, res_pkg["dfs"]["tet_pairs"], hide_index=True)
                with pc_r:
                    st.markdown("**Парные связи катион–катион в 1-й сфере октаэдров (B)**")
                    ui_full_width(st.dataframe, res_pkg["dfs"]["oct_pairs"], hide_index=True)

            with rtab_stoich:
                ui_full_width(st.dataframe, res_pkg["dfs"]["stoichiometry"], hide_index=True)
                ui_full_width(
                    st.download_button,
                    f"💾 Скачать рецепт этого состава ({pref_r}_config.json) для повторной загрузки",
                    data=sim_r.config.to_json().encode("utf-8"),
                    file_name=f"{pref_r}_config.json",
                    mime="application/json",
                    key="dl_json_res",
                )

            with rtab_files:
                f_filter = st.radio(
                    "Фильтр таблицы координат по подрешётке:",
                    options=["Все узлы", "TET (Тетраэдры A)", "OCT (Октаэдры B)", "ANION (Анионы)"],
                    horizontal=True,
                )
                coord_view = res_pkg["dfs"]["coordinates"]
                if f_filter.startswith("TET"):
                    coord_view = coord_view[coord_view["Подрешётка"] == "TET"]
                elif f_filter.startswith("OCT"):
                    coord_view = coord_view[coord_view["Подрешётка"] == "OCT"]
                elif f_filter.startswith("ANION"):
                    coord_view = coord_view[coord_view["Подрешётка"] == "ANION"]
                ui_full_width(st.dataframe, coord_view, hide_index=True, height=320)

                pv1, pv2 = st.columns(2)
                with pv1:
                    st.markdown("**Первые 35 строк входного файла GULP (`.gin`):**")
                    st.code("\n".join(res_pkg["gin_text"].splitlines()[:35]), language="text")
                with pv2:
                    st.markdown("**Первые 35 строк отчётной таблицы координат (`.txt` / `.docx`):**")
                    st.code("\n".join(res_pkg["txt_table"].splitlines()[:35]), language="text")

    # =========================================================================
    # ВКЛАДКА 2: ПАКЕТНЫЙ РАСЧЁТ СЕРИИ СОСТАВОВ
    # =========================================================================
    with tab_batch:
        st.subheader("📦 Серийный расчёт сразу нескольких вариантов")
        st.markdown(
            "Если вам нужно прогнать **целую серию составов** (например, все 3 варианта высокоэнтропийной шпинели "
            "`Co-Ni-Mn-Fe-Cr` или все 4 точки бинарного раствора `AB`) и скачать всё одним общим архивом со сводной "
            "таблицей характеристик — отметьте нужные варианты ниже."
        )

        all_batch_keys = [k for k in presets_map.keys() if k != "custom"]
        qb1, qb2, qb3 = st.columns(3)
        if qb1.button("Выбрать Варианты 1–3 (Co-Ni-Mn-Fe-Cr)", key="b_sel_v123"):
            st.session_state["batch_keys"] = ["variant_1", "variant_2", "variant_3"]
        if qb2.button("Выбрать всю бинарную серию AB (4 точки)", key="b_sel_ab"):
            st.session_state["batch_keys"] = ["ab_32_224", "ab_64_192", "ab_85_171", "ab_128_128"]
        if qb3.button("Выбрать все 8 встроенных задач", key="b_sel_all8"):
            st.session_state["batch_keys"] = all_batch_keys

        selected_batch = st.multiselect(
            "Отметьте составы для пакетного расчёта:",
            options=all_batch_keys,
            default=st.session_state.get("batch_keys", ["variant_1", "variant_2", "variant_3"]),
            format_func=lambda k: f"{k} — {presets_map[k].title}",
        )
        st.session_state["batch_keys"] = selected_batch

        bs1, bs2, bs3 = st.columns(3)
        b_tries = int(bs1.number_input("Попыток Монте-Карло на каждый вариант", min_value=1000, max_value=100000, value=15000, step=2500))
        b_gulp = bs2.selectbox("Шапка .gin файлов", options=["single", "opti conp", "opti conv"], key="b_gulp_sel")
        b_seed_on = bs3.checkbox("Фиксировать seed серии (42)", value=True)

        use_up_in_batch = False
        batch_base = build_ideal_spinel_supercell(2, 2, 4)
        if st.session_state.get("uploaded_struct") is not None:
            up_s: StructureTemplate = st.session_state["uploaded_struct"]
            if (up_s.n_tet, up_s.n_oct) == (128, 256):
                use_up_in_batch = st.checkbox(
                    f"Использовать загруженный во вкладке 1 каркас «{up_s.source_name}» и его параметры ячейки",
                    value=True,
                )
                if use_up_in_batch:
                    batch_base = up_s

        if ui_full_width(
            st.button,
            f"🚀 Рассчитать выбранную серию ({len(selected_batch)} шт.) и собрать общий архив",
            type="primary",
            disabled=(len(selected_batch) == 0),
            key="btn_run_batch",
        ):
            with st.spinner("Выполняется пакетная оптимизация всех выбранных составов..."):
                cfgs_to_run = [presets_map[k] for k in selected_batch]
                st.session_state["batch_result"] = run_batch_series(
                    base_structure=batch_base,
                    configs=cfgs_to_run,
                    max_tries=b_tries,
                    target_chi2=0.45,
                    seed=42 if b_seed_on else None,
                    gulp_header=b_gulp,
                    use_uploaded_lattice=use_up_in_batch,
                )

        b_res = st.session_state.get("batch_result")
        if b_res is not None:
            st.success("Серия успешно рассчитана!")
            ui_full_width(st.dataframe, b_res["summary_df"], hide_index=True)
            ui_full_width(
                st.download_button,
                "📦 Скачать единый мастер-архив всей серии (spinel_series_bundle.zip)",
                data=b_res["zip_bytes"],
                file_name="spinel_series_bundle.zip",
                mime="application/zip",
                type="primary",
                key="dl_batch_master",
            )
            with st.expander("Посмотреть гистограммы всех рассчитанных вариантов серии"):
                for item in b_res["packages"]:
                    st.markdown(f"**{item['sim'].config.title}**")
                    ui_full_width(st.image, item["png_bytes"])

    # =========================================================================
    # ВКЛАДКА 3: ПОНЯТНАЯ ИНСТРУКЦИЯ ДЛЯ ГЕОЛОГОВ
    # =========================================================================
    with tab_guide:
        st.subheader("📖 Краткая памятка для геологов и кристаллохимиков")
        st.markdown(
            """
            ### 1. Что вообще делает эта программа?
            В структуре шпинели $AB_2X_4$ есть **два типа катионных позиций** (плюс каркас из анионов $X$, обычно кислорода $\\mathrm{O}$):
            * **Позиции $A$ (тетраэдры)** — окружены 4 анионами; у каждого тетраэдра в 1-й катионной сфере ровно **$CN = 4$** соседних тетраэдра.
            * **Позиции $B$ (октаэдры)** — окружены 6 анионами; у каждого октаэдра в 1-й катионной сфере ровно **$CN = 6$** соседних октаэдров.

            В стандартной суперъячейке $2 \\times 2 \\times 4$ содержится **896 атомов**: **128 тетраэдров ($A$)**, **256 октаэдров ($B$)** и **512 атомов кислорода**.
            Программа расселяет заданные вами катионы **строго раздельно внутри $A$ и внутри $B$** и перебирает тысячи случайных раскладок методом Монте-Карло, выбирая ту, где распределение одноимённых соседей ближе всего к идеальному случайному хаосу (минимальный критерий $\\chi^2$ — без искусственного слипания одинаковых атомов в кластеры).

            ---

            ### 2. Какой режим на Шаге 1 выбрать?
            * **«Реальные многокомпонентные шпинели»** — выбирайте, когда работаете с конкретными химическими элементами (`Co`, `Ni`, `Mn`, `Fe`, `Cr`, `Mg`, `Al`, `Zn`, `Ti` и любыми другими). Программа посчитает не только координаты и заряды, но и рентгеновскую плотность кристалла $\\rho$ ($\\text{г/см}^3$).
            * **«Бинарный твёрдый раствор (тип A–B)»** — выбирайте, когда изучаете бинарное смешение двух компонентов в одной подрешётке (например, в тетраэдрах всегда сидит `Mg`, а в октаэдрах меняется доля `A : B` от 0% до 100%, либо смешиваются два реальных металла вроде `Al` и `Cr`).
            * **«Модельная система ABCDE»** — для абстрактных расчётов многокомпонентного разупорядочения с буквенными метками `A, B, C, D, E`.
            * **«Свой состав с нуля / .json»** — если хотите собрать полностью новую систему или загрузить сохранённый ранее файл `*_config.json`.

            ---

            ### 3. Как ввести совершенно новый состав со своими элементами и зарядами?
            1. На **Шаге 2** выберите либо встроенную решётку $2 \\times 2 \\times 4$ (никакой файл не нужен), либо перетащите свой `.docx` / `.txt` / `.cif` файл с координатами.
            2. На **Шаге 3** в левой таблице (**Тетраэдры A**) и правой таблице (**Октаэдры B**) дважды кликните на ячейки и впишите нужные символы элементов, их количества и заряды.
               * Чтобы **добавить новую строку**, кликните на пустую нижнюю строчку со значком `+`.
               * Чтобы **удалить строку**, отметьте галочку слева от строки и нажмите значок корзины (или клавишу `Delete`).
               * Один и тот же элемент **можно** вписывать и в $A$, и в $B$ даже с разными зарядами (например, `Fe` с зарядом `+1.2` в тетраэдрах и `Fe` с зарядом `+1.8` в октаэдрах).
               * Если вы знаете состав только в процентах или долях (например, `20, 20, 20, 20, 20`), впишите эти проценты в столбец «Количество» и нажмите кнопку **«Пересчитать введённые числа/проценты пропорционально»** под таблицей — программа сама округлит их ровно до 128 или 256 целых атомов!
            3. Если суммарный заряд ячейки не сходится к нулю, откройте спойлер **«Помощник электронейтральности»** под таблицами и нажмите **«Подставить этот заряд»** — программа сама рассчитает точный компенсирующий заряд до 9-го знака после запятой.
            4. Нажмите **«Запустить раздельное перемешивание»** и скачайте готовый ZIP-архив.

            ---

            ### 4. Какие файлы лежат внутри скачанного ZIP-архива и куда их открывать?
            * **`*.gin`** — готовый входной файл для расчёта энергии решётки и релаксации структуры в пакете **GULP** (остаётся только дописать блок ваших межатомных потенциалов в конец файла).
            * **`*.cif`** — кристаллографический файл для 3D-просмотра полиэдров и проверки межатомных расстояний в программе **VESTA**.
            * **`*_table.docx` и `*_table.txt`** — готовый документ **Microsoft Word** с шапкой параметров ячейки, статистикой $\\chi^2$ и полной таблицей координат/зарядов с флагами `c` и `*`.
            * **`*_histograms.png`** — чёткие чёрно-белые гистограммы (`300 dpi`) «Факт против Биномиальной теории» для вставки в статью, отчёт или презентацию.
            * **`*_statistics.csv`, `*_pair_contacts.csv`, `*_stoichiometry.csv`** — численные таблицы распределения контактов и параметров ближнего порядка Уоррена–Каули $\\alpha_{ij}$ (открываются в **Excel** и **Origin**).
            * **`*_config.json`** — компактный файл-рецепт вашего состава: перетащите его на Шаге 1 в следующий раз, чтобы не вбивать числа заново.
            """
        )

    # =========================================================================
    # БОКОВАЯ ПАНЕЛЬ (СТАТУС И СОХРАНЕНИЕ РЕЦЕПТА)
    # =========================================================================
    with st.sidebar:
        st.header("🧭 Сводка проекта")
        st.markdown(f"**Активный префикс:** `{live_cfg.key}`")
        st.markdown(f"**Каркас:** `{preview_struct.source_name}` ({preview_struct.n_total} ат.)")
        st.markdown(
            f"**Тетраэдры A:** {'✅' if val_report.n_tet_configured == val_report.n_tet_expected else '⚠️'} "
            f"`{val_report.n_tet_configured} / {val_report.n_tet_expected}`"
        )
        st.markdown(
            f"**Октаэдры B:** {'✅' if val_report.n_oct_configured == val_report.n_oct_expected else '⚠️'} "
            f"`{val_report.n_oct_configured} / {val_report.n_oct_expected}`"
        )
        st.markdown(
            f"**Заряд ячейки:** {'✅' if val_report.is_electroneutral else '⚠️'} "
            f"`{val_report.q_total:+.6f} e`"
        )
        st.divider()
        ui_full_width(
            st.download_button,
            "💾 Сохранить текущий рецепт (.json)",
            data=live_cfg.to_json().encode("utf-8"),
            file_name=f"{live_cfg.key}_config.json",
            mime="application/json",
            key="sidebar_dl_json",
        )
        if ui_full_width(st.button, "🔄 Сбросить всё к Варианту 1", key="sidebar_reset_v1"):
            load_config_into_state(presets_map["variant_1"])
            st.session_state["last_loaded_preset"] = "variant_1"
            st.rerun()


if __name__ == "__main__":
    run_streamlit_app()
