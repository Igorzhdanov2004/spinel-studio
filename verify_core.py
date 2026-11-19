"""
verify_core.py — Автоматическая верификация и CLI-интерфейс вычислительного ядра Spinel Studio.
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from engine import (
    analyze_coordination_shells,
    build_ideal_spinel_supercell,
    build_summary_dataframes,
    build_zip_package,
    parse_structure_file,
    run_simulation,
    save_package_to_directory,
    to_coordinate_docx,
    to_coordinate_txt,
    to_gulp_gin,
    to_vesta_cif,
)
from presets import CompositionConfig, SpeciesSpec, get_builtin_presets


def run_self_tests(export_demo_dir: Optional[str] = None) -> None:
    print("=" * 78)
    print("SPINEL STUDIO — АВТОМАТИЧЕСКАЯ ПРОВЕРКА ВЫЧИСЛИТЕЛЬНОГО ЯДРА")
    print("=" * 78)

    base_224 = build_ideal_spinel_supercell(nx=2, ny=2, nz=4, lattice=(16.598, 16.598, 33.196))
    assert base_224.n_total == 896
    assert (base_224.n_tet, base_224.n_oct, base_224.n_anion) == (128, 256, 512)

    tet_shell = analyze_coordination_shells(base_224, "TET")
    oct_shell = analyze_coordination_shells(base_224, "OCT")
    assert tet_shell["expected_cn"] == 4
    assert oct_shell["expected_cn"] == 6
    print(f"[OK] Шаг 1. Базовая решётка Fd-3m (2×2×4): {base_224.n_total} атомов.")

    presets = get_builtin_presets()
    sim_v1 = None
    for key, cfg in presets.items():
        if key == "custom":
            continue
        sim = run_simulation(base_224, cfg, max_tries=8000, target_chi2=0.45, seed=42, strict_neutrality=True)
        if key == "variant_1":
            sim_v1 = sim
        assert abs(sim.validation.q_total) < 1e-5
    print("[OK] Шаг 2. Все встроенные пресеты успешно оптимизированы и электронейтральны.")

    assert sim_v1 is not None
    for fmt_name, payload in [
        ("variant_1_table.txt", to_coordinate_txt(sim_v1).encode("utf-8")),
        ("variant_1_table.docx", to_coordinate_docx(sim_v1, title_prefix="variant_1")),
        ("variant_1.gin", to_gulp_gin(sim_v1).encode("utf-8")),
        ("variant_1.cif", to_vesta_cif(sim_v1).encode("utf-8")),
    ]:
        parsed = parse_structure_file(payload, fmt_name)
        assert (parsed.n_total, parsed.n_tet, parsed.n_oct, parsed.n_anion) == (896, 128, 256, 512)
    print("[OK] Шаг 3. Круговой тест парсера (.docx, .txt, .gin, .cif) пройден.")

    fig, zip_bytes = build_zip_package(sim_v1, prefix="variant_1")
    plt.close(fig)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        names = zf.namelist()
    assert len(names) == 9
    dfs = build_summary_dataframes(sim_v1)
    assert set(dfs.keys()) == {"tet_hist", "oct_hist", "tet_pairs", "oct_pairs", "stoichiometry", "coordinates"}
    print(f"[OK] Шаг 4. Сборка ZIP-архива проверена ({len(names)} файлов).")

    if export_demo_dir:
        save_package_to_directory(sim_v1, export_demo_dir, prefix="variant_1")
        print(f"[ВЫГРУЗКА] Демо-пакет сохранён в '{export_demo_dir}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI и самопроверка ядра Spinel Studio")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--all-presets", action="store_true")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--tries", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not args.preset and not args.all_presets:
        run_self_tests(export_demo_dir=args.out)
        return

    presets = get_builtin_presets()
    if args.input:
        with open(args.input, "rb") as f:
            base_struct = parse_structure_file(f.read(), os.path.basename(args.input))
    else:
        base_struct = build_ideal_spinel_supercell(nx=2, ny=2, nz=4)

    out_root = args.out or "./spinel_output"
    keys_to_run = [k for k in presets if k != "custom"] if args.all_presets else [args.preset]

    for key in keys_to_run:
        if key not in presets:
            raise SystemExit(f"Неизвестный пресет '{key}'.")
        cfg = presets[key]
        sim = run_simulation(base_struct, cfg, max_tries=args.tries, seed=args.seed)
        target_dir = os.path.join(out_root, key) if args.all_presets else out_root
        files = save_package_to_directory(sim, target_dir, prefix=key)
        print(f"[ГОТОВО] {key} -> {target_dir} ({len(files)} файлов)")


if __name__ == "__main__":
    main()
