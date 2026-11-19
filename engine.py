"""
engine.py — Вычислительное ядро Spinel Studio.

Отвечает за:
1. Генерацию идеальных суперъячеек шпинели (Fd-3m, Origin Choice 2) произвольной размерности.
2. Универсальный парсинг файлов структур (.docx, .txt, .out, .gin, .cif) без жёсткой привязки к элементам.
3. Автоматическую геометрическую классификацию позиций (TET: CN_O=4, OCT: CN_O=6, ANION) с учётом PBC.
4. Строгий контроль геометрии (отсутствие наложений координат, проверка координационных сфер) и электронейтральности.
5. Раздельную оптимизацию подрешёток A и B методом Монте-Карло по критерию хи-квадрат (χ²).
6. Экспорт полного комплекта результатов (.gin, .cif, .docx, .txt, .png 300 dpi, сводные таблицы, .json и единый .zip).
"""

from __future__ import annotations

import io
import math
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt

from presets import (
    ATOMIC_MASSES,
    DEFAULT_ANION_SYMBOLS,
    CompositionConfig,
    SpeciesSpec,
)

AVOGADRO: float = 6.02214076e23  # моль^-1


@dataclass
class AtomRecord:
    """Единичная атомная позиция в суперъячейке (хранит собственный заряд и заселённость)."""
    index: int
    site_type: str      # "TET", "OCT" или "ANION"
    element: str        # Химический или модельный символ
    x: float
    y: float
    z: float
    charge: float = 0.0
    occupancy: float = 1.0


@dataclass
class StructureTemplate:
    """Исходный кристаллографический каркас суперъячейки."""
    atoms: List[AtomRecord]
    lattice: np.ndarray          # [a, b, c] в Å
    angles: np.ndarray           # [alpha, beta, gamma] в градусах
    source_name: str = "ideal"

    @property
    def n_total(self) -> int:
        return len(self.atoms)

    @property
    def n_tet(self) -> int:
        return sum(1 for a in self.atoms if a.site_type == "TET")

    @property
    def n_oct(self) -> int:
        return sum(1 for a in self.atoms if a.site_type == "OCT")

    @property
    def n_anion(self) -> int:
        return sum(1 for a in self.atoms if a.site_type == "ANION")

    @property
    def cell_matrix(self) -> np.ndarray:
        """Матрица перехода из дробных координат в декартовы (Å), строки — векторы a, b, c."""
        a, b, c = [float(v) for v in self.lattice]
        al, be, ga = [math.radians(float(v)) for v in self.angles]
        cos_a, cos_b, cos_g = math.cos(al), math.cos(be), math.cos(ga)
        sin_g = math.sin(ga)
        cx = c * cos_b
        cy = c * (cos_a - cos_b * cos_g) / sin_g
        cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
        return np.array([
            [a, 0.0, 0.0],
            [b * cos_g, b * sin_g, 0.0],
            [cx, cy, cz],
        ], dtype=float)

    @property
    def volume(self) -> float:
        """Объём суперъячейки в Å³."""
        return float(abs(np.linalg.det(self.cell_matrix)))

    def with_lattice(
        self,
        lattice: Sequence[float],
        angles: Sequence[float] = (90.0, 90.0, 90.0),
    ) -> "StructureTemplate":
        copied_atoms = [
            AtomRecord(a.index, a.site_type, a.element, a.x, a.y, a.z, a.charge, a.occupancy)
            for a in self.atoms
        ]
        return StructureTemplate(
            atoms=copied_atoms,
            lattice=np.array(lattice, dtype=float),
            angles=np.array(angles, dtype=float),
            source_name=self.source_name,
        )


@dataclass
class ValidationReport:
    """Сводка предрасчётной проверки геометрии, стехиометрии и баланса зарядов."""
    is_valid_counts: bool
    is_electroneutral: bool
    has_no_overlaps: bool
    min_interatomic_distance: float
    n_tet_expected: int
    n_tet_configured: int
    n_oct_expected: int
    n_oct_configured: int
    n_anion: int
    q_tet: float
    q_oct: float
    q_anion: float
    q_total: float
    volume_ang3: float
    density_g_cm3: Optional[float]
    formula_summary: str
    messages: List[str]


@dataclass
class SublatticeOptResult:
    """Результат оптимизации одной подрешётки (TET или OCT)."""
    site_type: str
    indices: List[int]
    assigned_elements: List[str]
    assigned_charges: List[float]
    cn: int
    cutoff: float
    shell_d_min: float
    shell_d_max: float
    next_shell_d_min: Optional[float]
    obs_counts: np.ndarray
    theor_counts: np.ndarray
    chi2_per_bin: np.ndarray
    chi2_total: float
    attempts_used: int
    seed_used: int
    species_labels: List[str]
    pair_bonds_obs: np.ndarray
    pair_bonds_theor: np.ndarray


@dataclass
class SimulationResult:
    """Полный результат генерации и анализа неупорядоченной суперъячейки."""
    config: CompositionConfig
    structure: StructureTemplate
    tet_result: SublatticeOptResult
    oct_result: SublatticeOptResult
    validation: ValidationReport


def build_ideal_spinel_supercell(
    nx: int = 2,
    ny: int = 2,
    nz: int = 4,
    lattice: Optional[Sequence[float]] = None,
    angles: Sequence[float] = (90.0, 90.0, 90.0),
    a0: float = 8.299,
    u_param: float = 0.260,
    tet_element: str = "Ni",
    oct_element: str = "Cr",
    anion_element: str = "O",
    tet_charge: float = 1.2,
    oct_charge: float = 1.8,
    anion_charge: float = -1.2,
) -> StructureTemplate:
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("Кратности суперъячейки (nx, ny, nz) должны быть >= 1.")

    fcc_shifts = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ])

    tet_base = np.array([
        [0.125, 0.125, 0.125],
        [0.875, 0.875, 0.875],
    ])

    oct_base = np.array([
        [0.50, 0.50, 0.50],
        [0.50, 0.25, 0.25],
        [0.25, 0.50, 0.25],
        [0.25, 0.25, 0.50],
    ])

    d = float(u_param) - 0.125
    tet_vecs = np.array([
        [ d,  d,  d],
        [ d, -d, -d],
        [-d,  d, -d],
        [-d, -d,  d],
    ])
    ox_pos = np.array([0.125, 0.125, 0.125]) + tet_vecs
    ox_base = np.vstack([ox_pos, -ox_pos]) % 1.0

    def _expand_unit(base: np.ndarray) -> np.ndarray:
        return (base[:, None, :] + fcc_shifts[None, :, :]).reshape(-1, 3) % 1.0

    def _tile(unit_pts: np.ndarray) -> np.ndarray:
        dims = np.array([nx, ny, nz], dtype=float)
        blocks = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    blocks.append((unit_pts + np.array([ix, iy, iz])) / dims)
        coords = np.vstack(blocks) % 1.0
        coords = np.where(np.abs(coords - 1.0) < 1e-12, 0.0, coords)
        return coords

    tet_sc = _tile(_expand_unit(tet_base))
    oct_sc = _tile(_expand_unit(oct_base))
    ox_sc = _tile(_expand_unit(ox_base))

    atoms: List[AtomRecord] = []
    idx = 1
    for pt in tet_sc:
        atoms.append(AtomRecord(idx, "TET", tet_element, float(pt[0]), float(pt[1]), float(pt[2]), tet_charge))
        idx += 1
    for pt in oct_sc:
        atoms.append(AtomRecord(idx, "OCT", oct_element, float(pt[0]), float(pt[1]), float(pt[2]), oct_charge))
        idx += 1
    for pt in ox_sc:
        atoms.append(AtomRecord(idx, "ANION", anion_element, float(pt[0]), float(pt[1]), float(pt[2]), anion_charge))
        idx += 1

    if lattice is None:
        lat_arr = np.array([nx * a0, ny * a0, nz * a0], dtype=float)
    else:
        lat_arr = np.array(lattice, dtype=float)

    return StructureTemplate(
        atoms=atoms,
        lattice=lat_arr,
        angles=np.array(angles, dtype=float),
        source_name=f"ideal_Fd3m_{nx}x{ny}x{nz}",
    )


def compute_pbc_distances(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    cell_matrix: np.ndarray,
) -> np.ndarray:
    diff = (coords_a[:, None, :] % 1.0) - (coords_b[None, :, :] % 1.0)
    diff -= np.round(diff)
    cart = diff @ cell_matrix
    return np.linalg.norm(cart, axis=-1)


def _clean_float_token(tok: str) -> float:
    cleaned = re.sub(r"\(\d+\)$", "", tok.strip())
    return float(cleaned)


def _strip_atom_label(label: str) -> str:
    m = re.match(r"^([A-Za-z]{1,2})", label.strip())
    if not m:
        return label.strip()
    raw = m.group(1)
    if len(raw) == 2 and raw[1].islower():
        return raw[0].upper() + raw[1].lower()
    return raw[0].upper() if len(raw) == 1 else raw


def classify_spinel_sites(
    raw_entries: List[Tuple[str, float, float, float, float]],
    lattice: np.ndarray,
    angles: np.ndarray = np.array([90.0, 90.0, 90.0]),
    tet_hint: Optional[Set[str]] = None,
    oct_hint: Optional[Set[str]] = None,
    anion_hint: Optional[Set[str]] = None,
) -> List[AtomRecord]:
    if not raw_entries:
        raise ValueError("В файле не найдено ни одной атомной координаты.")

    n_atoms = len(raw_entries)
    elements = [e[0] for e in raw_entries]
    coords = np.array([[e[1], e[2], e[3]] for e in raw_entries], dtype=float)
    charges = [e[4] for e in raw_entries]

    anion_set = set(anion_hint) if anion_hint else set(DEFAULT_ANION_SYMBOLS)
    is_anion = np.array([el in anion_set for el in elements], dtype=bool)

    if not np.any(is_anion) and n_atoms % 7 == 0:
        counts_by_el: Dict[str, int] = {}
        for el in elements:
            counts_by_el[el] = counts_by_el.get(el, 0) + 1
        target_anion_count = (n_atoms * 4) // 7
        for el, cnt in counts_by_el.items():
            if cnt == target_anion_count:
                is_anion = np.array([x == el for x in elements], dtype=bool)
                break

    site_types = [""] * n_atoms
    for i in np.where(is_anion)[0]:
        site_types[i] = "ANION"

    cat_indices = np.where(~is_anion)[0]
    if len(cat_indices) == 0:
        raise ValueError("Не удалось выделить катионные позиции (все атомы определены как анионы).")

    if tet_hint and oct_hint:
        cat_els = {elements[i] for i in cat_indices}
        if cat_els.issubset(tet_hint | oct_hint):
            for i in cat_indices:
                site_types[i] = "TET" if elements[i] in tet_hint else "OCT"
            return [
                AtomRecord(i + 1, site_types[i], elements[i], float(coords[i, 0]),
                           float(coords[i, 1]), float(coords[i, 2]), float(charges[i]))
                for i in range(n_atoms)
            ]

    dummy_struct = StructureTemplate([], lattice, angles)
    cell_mat = dummy_struct.cell_matrix
    classified_geometrically = False

    if np.any(is_anion):
        anion_indices = np.where(is_anion)[0]
        d_cat_an = compute_pbc_distances(coords[cat_indices], coords[anion_indices], cell_mat)
        nearest_per_cat = np.min(d_cat_an, axis=1)
        d0 = float(np.min(nearest_per_cat))
        bond_cutoff = 1.38 * d0
        cn_anion = np.sum(d_cat_an <= bond_cutoff, axis=1)

        if set(cn_anion.tolist()).issubset({4, 6}) and (4 in cn_anion) and (6 in cn_anion):
            for local_idx, global_idx in enumerate(cat_indices):
                site_types[global_idx] = "TET" if cn_anion[local_idx] == 4 else "OCT"
            classified_geometrically = True

    if not classified_geometrically:
        cat_counts: Dict[str, int] = {}
        for i in cat_indices:
            cat_counts[elements[i]] = cat_counts.get(elements[i], 0) + 1
        if len(cat_counts) == 2:
            sorted_els = sorted(cat_counts.items(), key=lambda kv: kv[1])
            tet_el_auto = sorted_els[0][0]
            for i in cat_indices:
                site_types[i] = "TET" if elements[i] == tet_el_auto else "OCT"
        else:
            raise ValueError(
                "Не удалось автоматически разделить катионы на TET и OCT. "
                "Проверьте корректность координат анионов или укажите символы подрешёток явно."
            )

    return [
        AtomRecord(
            index=i + 1,
            site_type=site_types[i],
            element=elements[i],
            x=float(coords[i, 0]),
            y=float(coords[i, 1]),
            z=float(coords[i, 2]),
            charge=float(charges[i]),
        )
        for i in range(n_atoms)
    ]


def _extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    if filename.lower().endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        chunks = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_txt = " ".join(c.text.strip() for c in row.cells if c.text.strip())
                if row_txt:
                    chunks.append(row_txt)
        return "\n".join(chunks)
    return file_bytes.decode("utf-8", errors="ignore")


def _parse_cif_text(text: str) -> Tuple[List[Tuple[str, float, float, float, float]], np.ndarray, np.ndarray]:
    lattice = np.array([16.598, 16.598, 33.196], dtype=float)
    angles = np.array([90.0, 90.0, 90.0], dtype=float)
    lines = text.splitlines()

    for line in lines:
        s = line.strip()
        if s.startswith("_cell_length_a"):
            lattice[0] = _clean_float_token(s.split()[1])
        elif s.startswith("_cell_length_b"):
            lattice[1] = _clean_float_token(s.split()[1])
        elif s.startswith("_cell_length_c"):
            lattice[2] = _clean_float_token(s.split()[1])
        elif s.startswith("_cell_angle_alpha"):
            angles[0] = _clean_float_token(s.split()[1])
        elif s.startswith("_cell_angle_beta"):
            angles[1] = _clean_float_token(s.split()[1])
        elif s.startswith("_cell_angle_gamma"):
            angles[2] = _clean_float_token(s.split()[1])

    entries: List[Tuple[str, float, float, float, float]] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().lower() == "loop_":
            i += 1
            headers: List[str] = []
            while i < len(lines) and lines[i].strip().startswith("_"):
                headers.append(lines[i].strip().lower().split()[0])
                i += 1
            if any("_atom_site_fract_x" in h for h in headers):
                idx_x = headers.index("_atom_site_fract_x")
                idx_y = headers.index("_atom_site_fract_y")
                idx_z = headers.index("_atom_site_fract_z")
                idx_sym = (
                    headers.index("_atom_site_type_symbol")
                    if "_atom_site_type_symbol" in headers
                    else headers.index("_atom_site_label")
                )
                idx_chg = (
                    headers.index("_atom_site_charge")
                    if "_atom_site_charge" in headers
                    else None
                )
                while i < len(lines):
                    row = lines[i].strip()
                    if not row or row.startswith("#") or row.lower() == "loop_" or row.startswith("_") or row.startswith("data_"):
                        break
                    parts = row.split()
                    if len(parts) >= max(idx_x, idx_y, idx_z, idx_sym) + 1:
                        el = _strip_atom_label(parts[idx_sym])
                        x = _clean_float_token(parts[idx_x])
                        y = _clean_float_token(parts[idx_y])
                        z = _clean_float_token(parts[idx_z])
                        q = _clean_float_token(parts[idx_chg]) if (idx_chg is not None and len(parts) > idx_chg) else 0.0
                        entries.append((el, x, y, z, q))
                    i += 1
                continue
        i += 1
    return entries, lattice, angles


def _parse_gulp_or_table_text(text: str) -> Tuple[List[Tuple[str, float, float, float, float]], np.ndarray, np.ndarray]:
    lattice = np.array([16.598, 16.598, 33.196], dtype=float)
    angles = np.array([90.0, 90.0, 90.0], dtype=float)

    lines = text.splitlines()
    for idx, line in enumerate(lines):
        s = line.strip()
        low = s.lower()

        m_a = re.search(r"\ba\s*=\s*([0-9.]+)", low)
        m_al = re.search(r"\balpha\s*=\s*([0-9.]+)", low)
        if m_a and m_al:
            lattice[0] = float(m_a.group(1))
            angles[0] = float(m_al.group(1))
            continue

        m_b = re.search(r"\bb\s*=\s*([0-9.]+)", low)
        m_be = re.search(r"\bbeta\s*=\s*([0-9.]+)", low)
        if m_b and m_be:
            lattice[1] = float(m_b.group(1))
            angles[1] = float(m_be.group(1))
            continue

        m_c = re.search(r"\bc\s*=\s*([0-9.]+)", low)
        m_ga = re.search(r"\bgamma\s*=\s*([0-9.]+)", low)
        if m_c and m_ga:
            lattice[2] = float(m_c.group(1))
            angles[2] = float(m_ga.group(1))
            continue

        if low == "cell" and idx + 1 < len(lines):
            nxt = lines[idx + 1].split()
            try:
                nums = [float(v) for v in nxt]
                if len(nums) >= 3:
                    lattice[:3] = nums[:3]
                if len(nums) >= 6:
                    angles[:3] = nums[3:6]
            except ValueError:
                pass

    stop_keywords = {
        "single", "opti", "conp", "conv", "cell", "fractional", "cartesian",
        "space", "species", "buck", "lennard", "spring", "total", "energy",
        "no.", "atomic", "loop_", "data_", "dimensionality",
    }

    blocks: List[List[Tuple[str, float, float, float, float]]] = [[]]
    last_serial: int = 0

    for line in lines:
        clean = line.replace("*", " ").strip()
        if not clean or clean.startswith(("#", "-", "=", "!")):
            continue
        parts = clean.split()
        first_low = parts[0].lower()
        if first_low in stop_keywords:
            continue

        serial_num: Optional[int] = None
        pos = 0
        if re.fullmatch(r"\d+", parts[0]):
            serial_num = int(parts[0])
            pos = 1
        if pos >= len(parts):
            continue

        cand_el = parts[pos]
        if not re.fullmatch(r"[A-Za-z]{1,2}\d*[+-]?", cand_el):
            continue
        if cand_el.lower() in stop_keywords:
            continue

        el = _strip_atom_label(cand_el)
        pos += 1

        if pos < len(parts):
            tag_low = parts[pos].lower()
            if tag_low in {"s", "shel", "shell"}:
                continue
            if tag_low in {"c", "core"}:
                pos += 1

        floats: List[float] = []
        ok_floats = True
        for tok in parts[pos:]:
            try:
                floats.append(float(tok))
            except ValueError:
                ok_floats = False
                break

        if not ok_floats or len(floats) < 3:
            continue

        x, y, z = floats[0], floats[1], floats[2]
        if max(abs(x), abs(y), abs(z)) > 2.0:
            continue
        charge = floats[3] if len(floats) >= 4 else 0.0

        if serial_num == 1 and last_serial > 10 and len(blocks[-1]) > 10:
            blocks.append([])
        if serial_num is not None:
            last_serial = serial_num

        blocks[-1].append((el, x, y, z, charge))

    non_empty = [b for b in blocks if len(b) > 0]
    if not non_empty:
        return [], lattice, angles
    max_len = max(len(b) for b in non_empty)
    best_blocks = [b for b in non_empty if len(b) == max_len]
    return best_blocks[-1], lattice, angles


def parse_structure_file(
    file_bytes: bytes,
    filename: str,
    tet_hint: Optional[Set[str]] = None,
    oct_hint: Optional[Set[str]] = None,
    anion_hint: Optional[Set[str]] = None,
) -> StructureTemplate:
    text = _extract_text_from_bytes(file_bytes, filename)
    is_cif = filename.lower().endswith(".cif") or ("_atom_site_fract_x" in text)

    if is_cif:
        entries, lattice, angles = _parse_cif_text(text)
    else:
        entries, lattice, angles = _parse_gulp_or_table_text(text)

    atoms = classify_spinel_sites(
        entries,
        lattice=lattice,
        angles=angles,
        tet_hint=tet_hint,
        oct_hint=oct_hint,
        anion_hint=anion_hint,
    )
    return StructureTemplate(atoms=atoms, lattice=lattice, angles=angles, source_name=filename)


def analyze_coordination_shells(
    structure: StructureTemplate,
    site_type: str,
    max_r: float = 7.0,
    round_decimals: int = 2,
) -> Dict[str, object]:
    sub_coords = np.array(
        [[a.x, a.y, a.z] for a in structure.atoms if a.site_type == site_type],
        dtype=float,
    )
    if len(sub_coords) < 2:
        raise ValueError(f"Недостаточно позиций в подрешётке {site_type} ({len(sub_coords)}).")

    dist = compute_pbc_distances(sub_coords, sub_coords, structure.cell_matrix)
    mask = (dist > 1e-5) & (dist <= max_r)
    rounded = np.round(dist[mask], round_decimals)
    unique_d = np.unique(rounded)
    if len(unique_d) == 0:
        raise ValueError(f"В радиусе {max_r} Å не найдено соседей для {site_type}.")

    shells: List[List[float]] = [[float(unique_d[0])]]
    for d_val in unique_d[1:]:
        if float(d_val) - shells[-1][-1] > 0.35:
            shells.append([float(d_val)])
        else:
            shells[-1].append(float(d_val))

    s1_max = shells[0][-1]
    s2_min = shells[1][0] if len(shells) > 1 else s1_max + 1.0
    suggested_cutoff = round(0.5 * (s1_max + s2_min), 2)
    cn_1st = int(np.median(np.sum((dist > 1e-5) & (dist <= suggested_cutoff), axis=1)))

    return {
        "site_type": site_type,
        "first_shell_range": (shells[0][0], s1_max),
        "second_shell_start": s2_min if len(shells) > 1 else None,
        "suggested_cutoff": suggested_cutoff,
        "expected_cn": cn_1st,
    }


def validate_setup(
    structure: StructureTemplate,
    config: CompositionConfig,
    min_dist_tol: float = 1.40,
    charge_tol: float = 1e-4,
) -> ValidationReport:
    all_coords = np.array([[a.x, a.y, a.z] for a in structure.atoms], dtype=float)
    dist_all = compute_pbc_distances(all_coords, all_coords, structure.cell_matrix)
    np.fill_diagonal(dist_all, np.inf)
    min_d = float(np.min(dist_all))
    has_no_overlaps = min_d >= min_dist_tol

    n_tet_exp = structure.n_tet
    n_oct_exp = structure.n_oct
    n_anion = structure.n_anion

    n_tet_cfg = sum(s.count for s in config.active_tet())
    n_oct_cfg = sum(s.count for s in config.active_oct())
    is_valid_counts = (n_tet_cfg == n_tet_exp) and (n_oct_cfg == n_oct_exp)

    q_tet = sum(s.count * s.charge for s in config.active_tet())
    q_oct = sum(s.count * s.charge for s in config.active_oct())
    q_anion = n_anion * config.anion_charge
    q_total = q_tet + q_oct + q_anion
    is_electroneutral = abs(q_total) <= charge_tol

    totals_by_el: Dict[str, int] = {}
    for s in config.active_tet():
        totals_by_el[s.element] = totals_by_el.get(s.element, 0) + s.count
    for s in config.active_oct():
        totals_by_el[s.element] = totals_by_el.get(s.element, 0) + s.count
    totals_by_el[config.anion_element] = totals_by_el.get(config.anion_element, 0) + n_anion

    formula_parts = [f"{el}{cnt}" for el, cnt in totals_by_el.items()]
    formula_summary = " ".join(formula_parts)

    vol = structure.volume
    density: Optional[float] = None
    if not config.is_model_labels and all(el in ATOMIC_MASSES for el in totals_by_el):
        mass_amu = sum(ATOMIC_MASSES[el] * cnt for el, cnt in totals_by_el.items())
        density = float(mass_amu / (AVOGADRO * vol * 1e-24))

    messages: List[str] = []
    if not has_no_overlaps:
        messages.append(f"Обнаружено критическое сближение/наложение атомов: d_min = {min_d:.4f} Å (< {min_dist_tol} Å)!")
    if n_tet_cfg != n_tet_exp:
        messages.append(f"Сумма атомов в тетраэдрах ({n_tet_cfg}) не совпадает с числом позиций TET ({n_tet_exp}).")
    if n_oct_cfg != n_oct_exp:
        messages.append(f"Сумма атомов в октаэдрах ({n_oct_cfg}) не совпадает с числом позиций OCT ({n_oct_exp}).")
    if not is_electroneutral:
        messages.append(f"Ячейка не электронейтральна: суммарный заряд Q = {q_total:+.6f} e.")

    return ValidationReport(
        is_valid_counts=is_valid_counts,
        is_electroneutral=is_electroneutral,
        has_no_overlaps=has_no_overlaps,
        min_interatomic_distance=min_d,
        n_tet_expected=n_tet_exp,
        n_tet_configured=n_tet_cfg,
        n_oct_expected=n_oct_exp,
        n_oct_configured=n_oct_cfg,
        n_anion=n_anion,
        q_tet=q_tet,
        q_oct=q_oct,
        q_anion=q_anion,
        q_total=q_total,
        volume_ang3=vol,
        density_g_cm3=density,
        formula_summary=formula_summary,
        messages=messages,
    )


def optimize_sublattice(
    structure: StructureTemplate,
    site_type: str,
    species_specs: Sequence[SpeciesSpec],
    cutoff: float,
    max_tries: int = 15000,
    target_chi2: float = 0.50,
    seed: Optional[int] = None,
) -> SublatticeOptResult:
    active = [s for s in species_specs if s.count > 0]
    idxs = [i for i, a in enumerate(structure.atoms) if a.site_type == site_type]
    n_sites = len(idxs)

    if sum(s.count for s in active) != n_sites:
        raise ValueError(
            f"Сумма атомов для {site_type} ({sum(s.count for s in active)}) != числу позиций ({n_sites})."
        )

    coords = np.array([[structure.atoms[i].x, structure.atoms[i].y, structure.atoms[i].z] for i in idxs], dtype=float)
    dist = compute_pbc_distances(coords, coords, structure.cell_matrix)

    in_shell = (dist > 1e-5) & (dist <= cutoff)
    cn_per_site = np.sum(in_shell, axis=1)
    unique_cns = sorted(set(cn_per_site.tolist()))
    if len(unique_cns) != 1 or unique_cns[0] == 0:
        hint = analyze_coordination_shells(structure, site_type)
        raise ValueError(
            f"Некорректный радиус 1-й сферы R={cutoff} Å для {site_type}: найдено соседей {unique_cns}. "
            f"Рекомендуемый радиус для данной ячейки: R = {hint['suggested_cutoff']} Å (1-я сфера: {hint['first_shell_range']} Å)."
        )

    cn = unique_cns[0]
    nbrs = np.array([np.where(in_shell[row_i])[0] for row_i in range(n_sites)], dtype=int)

    shell_dists = dist[in_shell]
    d_min = float(np.min(shell_dists))
    d_max = float(np.max(shell_dists))
    outside_dists = dist[dist > cutoff]
    next_d_min = float(np.min(outside_dists)) if len(outside_dists) > 0 else None

    unique_elements: List[str] = []
    elem_counts: Dict[str, int] = {}
    slot_elem_ids: List[int] = []
    slot_charges: List[float] = []

    for spec in active:
        if spec.element not in elem_counts:
            unique_elements.append(spec.element)
            elem_counts[spec.element] = 0
        el_id = unique_elements.index(spec.element)
        elem_counts[spec.element] += spec.count
        slot_elem_ids.extend([el_id] * spec.count)
        slot_charges.extend([spec.charge] * spec.count)

    slot_elem_arr = np.array(slot_elem_ids, dtype=int)
    slot_charge_arr = np.array(slot_charges, dtype=float)
    n_species = len(unique_elements)

    theor = np.zeros(cn + 1, dtype=float)
    for el in unique_elements:
        c = elem_counts[el]
        p = c / n_sites
        for k in range(cn + 1):
            theor[k] += c * math.comb(cn, k) * (p ** k) * ((1.0 - p) ** (cn - k))
    valid_bins = theor > 1e-6

    rng_init = np.random.default_rng(seed)
    actual_seed = int(seed) if seed is not None else int(rng_init.integers(1, 2**31 - 1))
    rng = np.random.default_rng(actual_seed)

    effective_tries = 1 if n_species == 1 else max(1, int(max_tries))

    best_chi2 = float("inf")
    best_order: Optional[np.ndarray] = None
    best_obs: Optional[np.ndarray] = None
    best_attempt = 1

    base_order = np.arange(n_sites, dtype=int)
    for att in range(1, effective_tries + 1):
        order = base_order.copy() if n_species == 1 else rng.permutation(base_order)
        perm_el = slot_elem_arr[order]
        same_neighbors = np.sum(perm_el[nbrs] == perm_el[:, None], axis=1)
        obs = np.bincount(same_neighbors, minlength=cn + 1)
        chi2 = float(np.sum(((obs[valid_bins] - theor[valid_bins]) ** 2) / theor[valid_bins]))

        if chi2 < best_chi2:
            best_chi2 = chi2
            best_order = order.copy()
            best_obs = obs.copy()
            best_attempt = att
            if best_chi2 <= target_chi2:
                break

    assert best_order is not None and best_obs is not None
    final_el_ids = slot_elem_arr[best_order]
    final_charges = slot_charge_arr[best_order]

    chi2_bins = np.zeros(cn + 1, dtype=float)
    chi2_bins[valid_bins] = ((best_obs[valid_bins] - theor[valid_bins]) ** 2) / theor[valid_bins]

    pair_obs_dir = np.zeros((n_species, n_species), dtype=float)
    src_el = np.repeat(final_el_ids, cn)
    dst_el = final_el_ids[nbrs.ravel()]
    np.add.at(pair_obs_dir, (src_el, dst_el), 1.0)

    pair_bonds_obs = np.zeros((n_species, n_species), dtype=float)
    pair_bonds_theor = np.zeros((n_species, n_species), dtype=float)
    total_bonds = 0.5 * n_sites * cn

    for i in range(n_species):
        pi = elem_counts[unique_elements[i]] / n_sites
        for j in range(i, n_species):
            pj = elem_counts[unique_elements[j]] / n_sites
            if i == j:
                bonds_ij = 0.5 * pair_obs_dir[i, i]
                theor_ij = total_bonds * (pi * pi)
            else:
                bonds_ij = 0.5 * (pair_obs_dir[i, j] + pair_obs_dir[j, i])
                theor_ij = total_bonds * (2.0 * pi * pj)
            pair_bonds_obs[i, j] = pair_bonds_obs[j, i] = bonds_ij
            pair_bonds_theor[i, j] = pair_bonds_theor[j, i] = theor_ij

    return SublatticeOptResult(
        site_type=site_type,
        indices=idxs,
        assigned_elements=[unique_elements[k] for k in final_el_ids],
        assigned_charges=[float(q) for q in final_charges],
        cn=cn,
        cutoff=cutoff,
        shell_d_min=d_min,
        shell_d_max=d_max,
        next_shell_d_min=next_d_min,
        obs_counts=best_obs,
        theor_counts=theor,
        chi2_per_bin=chi2_bins,
        chi2_total=best_chi2,
        attempts_used=best_attempt,
        seed_used=actual_seed,
        species_labels=unique_elements,
        pair_bonds_obs=pair_bonds_obs,
        pair_bonds_theor=pair_bonds_theor,
    )


def run_simulation(
    base_structure: StructureTemplate,
    config: CompositionConfig,
    max_tries: int = 15000,
    target_chi2: float = 0.50,
    seed: Optional[int] = None,
    strict_neutrality: bool = False,
) -> SimulationResult:
    work_struct = base_structure.with_lattice(config.lattice, config.angles)
    val = validate_setup(work_struct, config)

    if not val.has_no_overlaps:
        raise ValueError("Расчёт остановлен: " + " ".join(val.messages))
    if not val.is_valid_counts:
        raise ValueError("Расчёт остановлен: " + " ".join(val.messages))
    if strict_neutrality and not val.is_electroneutral:
        raise ValueError("Расчёт остановлен (нарушена электронейтральность): " + " ".join(val.messages))

    seed_tet = seed if seed is None else int(seed)
    seed_oct = None if seed is None else int(seed) + 104729

    tet_res = optimize_sublattice(
        work_struct, "TET", config.active_tet(), config.r_tet, max_tries=max_tries, target_chi2=target_chi2, seed=seed_tet
    )
    oct_res = optimize_sublattice(
        work_struct, "OCT", config.active_oct(), config.r_oct, max_tries=max_tries, target_chi2=target_chi2, seed=seed_oct
    )

    for local_i, global_i in enumerate(tet_res.indices):
        work_struct.atoms[global_i].element = tet_res.assigned_elements[local_i]
        work_struct.atoms[global_i].charge = tet_res.assigned_charges[local_i]

    for local_i, global_i in enumerate(oct_res.indices):
        work_struct.atoms[global_i].element = oct_res.assigned_elements[local_i]
        work_struct.atoms[global_i].charge = oct_res.assigned_charges[local_i]

    for atom in work_struct.atoms:
        if atom.site_type == "ANION":
            atom.element = config.anion_element
            atom.charge = config.anion_charge

    return SimulationResult(
        config=config,
        structure=work_struct,
        tet_result=tet_res,
        oct_result=oct_res,
        validation=val,
    )


def build_summary_dataframes(sim: SimulationResult) -> Dict[str, pd.DataFrame]:
    def _hist_df(res: SublatticeOptResult) -> pd.DataFrame:
        k_vals = np.arange(res.cn + 1)
        return pd.DataFrame({
            "Однородных соседей (k)": k_vals,
            "Факт (полиэдров)": res.obs_counts.astype(int),
            "Теория (биномиальная)": np.round(res.theor_counts, 3),
            "Разность (Факт − Теория)": np.round(res.obs_counts - res.theor_counts, 3),
            "Вклад в χ²": np.round(res.chi2_per_bin, 4),
        })

    def _pairs_df(res: SublatticeOptResult) -> pd.DataFrame:
        rows = []
        labels = res.species_labels
        total_b = max(float(np.sum(np.triu(res.pair_bonds_obs))), 1.0)
        for i in range(len(labels)):
            for j in range(i, len(labels)):
                obs_b = float(res.pair_bonds_obs[i, j])
                th_b = float(res.pair_bonds_theor[i, j])
                wc_alpha = 1.0 - (obs_b / th_b) if th_b > 1e-9 else 0.0
                rows.append({
                    "Пара катионов": f"{labels[i]}–{labels[j]}",
                    "Связей (Факт)": int(round(obs_b)),
                    "Связей (Теория)": round(th_b, 2),
                    "Доля факта (%)": round(100.0 * obs_b / total_b, 2),
                    "Параметр Уоррена–Каули α": round(wc_alpha, 4),
                })
        return pd.DataFrame(rows)

    summary_rows = []
    seen_keys: Dict[Tuple[str, float], Dict[str, object]] = {}
    for a in sim.structure.atoms:
        key = (a.element, round(a.charge, 9))
        if key not in seen_keys:
            seen_keys[key] = {
                "Элемент": a.element,
                "Заряд иона (e)": a.charge,
                "В тетраэдрах (A)": 0,
                "В октаэдрах (B)": 0,
                "В позициях анионов": 0,
                "Всего атомов": 0,
            }
        rec = seen_keys[key]
        if a.site_type == "TET":
            rec["В тетраэдрах (A)"] = int(rec["В тетраэдрах (A)"]) + 1
        elif a.site_type == "OCT":
            rec["В октаэдрах (B)"] = int(rec["В октаэдрах (B)"]) + 1
        else:
            rec["В позициях анионов"] = int(rec["В позициях анионов"]) + 1
        rec["Всего атомов"] = int(rec["Всего атомов"]) + 1

    for rec in seen_keys.values():
        rec["Суммарный заряд (e)"] = round(int(rec["Всего атомов"]) * float(rec["Заряд иона (e)"]), 6)
        summary_rows.append(rec)
    stoich_df = pd.DataFrame(summary_rows)

    coord_df = pd.DataFrame([
        {
            "No.": a.index,
            "Подрешётка": a.site_type,
            "Atomic Label": a.element,
            "x": round(a.x, 6),
            "y": round(a.y, 6),
            "z": round(a.z, 6),
            "Charge": round(a.charge, 9),
            "Occupancy": round(a.occupancy, 6),
        }
        for a in sim.structure.atoms
    ])

    return {
        "tet_hist": _hist_df(sim.tet_result),
        "oct_hist": _hist_df(sim.oct_result),
        "tet_pairs": _pairs_df(sim.tet_result),
        "oct_pairs": _pairs_df(sim.oct_result),
        "stoichiometry": stoich_df,
        "coordinates": coord_df,
    }


def to_gulp_gin(sim: SimulationResult, keyword_header: str = "single") -> str:
    lat = sim.structure.lattice
    ang = sim.structure.angles
    lines = [
        f"{keyword_header}\n",
        "cell\n",
        f"{lat[0]:.6f} {lat[1]:.6f} {lat[2]:.6f} {ang[0]:.6f} {ang[1]:.6f} {ang[2]:.6f}\n",
        "fractional\n",
    ]
    for a in sim.structure.atoms:
        lines.append(f"{a.element:<4} core  {a.x:9.6f}  {a.y:9.6f}  {a.z:9.6f}  {a.charge:13.9f}\n")
    lines.append("space\n1\n")
    return "".join(lines)


def to_vesta_cif(sim: SimulationResult) -> str:
    lat = sim.structure.lattice
    ang = sim.structure.angles
    vol = sim.structure.volume
    lines = [
        f"data_{sim.config.key}\n",
        f"_cell_length_a                  {lat[0]:.6f}\n",
        f"_cell_length_b                  {lat[1]:.6f}\n",
        f"_cell_length_c                  {lat[2]:.6f}\n",
        f"_cell_angle_alpha               {ang[0]:.4f}\n",
        f"_cell_angle_beta                {ang[1]:.4f}\n",
        f"_cell_angle_gamma               {ang[2]:.4f}\n",
        f"_cell_volume                    {vol:.4f}\n",
        "_space_group_crystal_system     triclinic\n",
        "_space_group_name_H-M_alt       'P 1'\n",
        "_space_group_IT_number          1\n\n",
        "loop_\n",
        "_atom_site_label\n",
        "_atom_site_type_symbol\n",
        "_atom_site_fract_x\n",
        "_atom_site_fract_y\n",
        "_atom_site_fract_z\n",
        "_atom_site_occupancy\n",
    ]
    for a in sim.structure.atoms:
        lines.append(
            f"{a.element}{a.index:<5} {a.element:<4} {a.x:9.6f} {a.y:9.6f} {a.z:9.6f} {a.occupancy:8.6f}\n"
        )
    return "".join(lines)


def to_coordinate_txt(sim: SimulationResult) -> str:
    lat = sim.structure.lattice
    ang = sim.structure.angles
    header = [
        "Dimensionality = 3",
        f"a = {lat[0]:10.6f}    alpha = {ang[0]:10.6f}",
        f"b = {lat[1]:10.6f}    beta  = {ang[1]:10.6f}",
        f"c = {lat[2]:10.6f}    gamma = {ang[2]:10.6f}",
        "",
        "   No.  Atomic Label      x           y           z          Charge         Occupancy",
        "-" * 86,
    ]
    rows = []
    for a in sim.structure.atoms:
        star = " " if a.index == 1 else "*"
        rows.append(
            f"{a.index:6d}  {a.element:<5} c    {a.x:9.6f} {star} {a.y:9.6f} {star} {a.z:9.6f} {star} "
            f"{a.charge:13.9f}    {a.occupancy:8.6f}"
        )
    return "\n".join(header + rows) + "\n"


def to_coordinate_docx(sim: SimulationResult, title_prefix: str = "Spinel Supercell") -> bytes:
    doc = Document()
    doc.add_heading(f"{title_prefix}: {sim.config.title}", level=2)

    lat = sim.structure.lattice
    val = sim.validation
    dens_str = f", ρ = {val.density_g_cm3:.4f} г/см³" if val.density_g_cm3 is not None else ""
    p_meta = doc.add_paragraph(
        f"Формула суперъячейки: {val.formula_summary}\n"
        f"Параметры решётки: a = {lat[0]:.4f} Å, b = {lat[1]:.4f} Å, c = {lat[2]:.4f} Å "
        f"(V = {val.volume_ang3:.2f} Å³{dens_str})\n"
        f"Тетраэдры (A, N={val.n_tet_expected}, CN={sim.tet_result.cn}, R={sim.tet_result.cutoff} Å): "
        f"χ² = {sim.tet_result.chi2_total:.4f} (попытка {sim.tet_result.attempts_used})\n"
        f"Октаэдры (B, N={val.n_oct_expected}, CN={sim.oct_result.cn}, R={sim.oct_result.cutoff} Å): "
        f"χ² = {sim.oct_result.chi2_total:.4f} (попытка {sim.oct_result.attempts_used})\n"
        f"Суммарный заряд ячейки: {val.q_total:+.6f} e"
    )
    for run in p_meta.runs:
        run.font.size = Pt(10)

    doc.add_heading("Таблица координат и зарядов (формат GULP)", level=3)
    coord_text = to_coordinate_txt(sim)
    p_table = doc.add_paragraph(coord_text)
    p_table.paragraph_format.space_before = Pt(2)
    p_table.paragraph_format.space_after = Pt(2)
    for run in p_table.runs:
        run.font.name = "Courier New"
        run.font.size = Pt(8)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


def plot_histograms(sim: SimulationResult, dpi: int = 300) -> Tuple[plt.Figure, bytes]:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
    panels = [
        (axes[0], sim.tet_result, "Подрешётка A (Тетраэдры)"),
        (axes[1], sim.oct_result, "Подрешётка B (Октаэдры)"),
    ]

    for ax, res, subtitle in panels:
        x = np.arange(res.cn + 1)
        width = 0.36
        bars_obs = ax.bar(
            x - width / 2, res.obs_counts, width,
            color="#363636", edgecolor="black", linewidth=0.8, label="Факт (модель)"
        )
        ax.bar(
            x + width / 2, res.theor_counts, width,
            color="#b5b5b5", edgecolor="black", linewidth=0.8, label="Теория (бином.)"
        )

        y_max = max(float(np.max(res.obs_counts)), float(np.max(res.theor_counts)), 1.0)
        ax.set_ylim(0, y_max * 1.16)
        ax.set_xticks(x)
        ax.set_xlabel("Число однородных контактов в 1-й сфере (k)", fontsize=10)
        ax.set_ylabel("Количество полиэдров", fontsize=10)
        ax.set_title(
            f"{subtitle} (CN = {res.cn}, R = {res.cutoff:.2f} Å)\nχ² = {res.chi2_total:.4f}",
            fontsize=10.5,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.45)

        for b in bars_obs:
            h = b.get_height()
            if h > 0:
                ax.text(
                    b.get_x() + b.get_width() / 2, h + y_max * 0.015,
                    f"{int(round(h))}", ha="center", va="bottom", fontsize=8
                )

    axes[0].legend(frameon=False, fontsize=9.5)
    plt.tight_layout()

    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=dpi, bbox_inches="tight")
    png_buf.seek(0)
    return fig, png_buf.getvalue()


def build_zip_package(sim: SimulationResult, prefix: str = "spinel_model") -> Tuple[plt.Figure, bytes]:
    gin_str = to_gulp_gin(sim)
    cif_str = to_vesta_cif(sim)
    txt_str = to_coordinate_txt(sim)
    docx_bytes = to_coordinate_docx(sim, title_prefix=prefix)
    fig, png_bytes = plot_histograms(sim, dpi=300)
    dfs = build_summary_dataframes(sim)

    hist_combined = pd.concat(
        [
            dfs["tet_hist"].assign(Подрешётка="TET (A)"),
            dfs["oct_hist"].assign(Подрешётка="OCT (B)"),
        ],
        ignore_index=True,
    )
    pairs_combined = pd.concat(
        [
            dfs["tet_pairs"].assign(Подрешётка="TET (A)"),
            dfs["oct_pairs"].assign(Подрешётка="OCT (B)"),
        ],
        ignore_index=True,
    )

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{prefix}.gin", gin_str)
        zf.writestr(f"{prefix}.cif", cif_str)
        zf.writestr(f"{prefix}_table.docx", docx_bytes)
        zf.writestr(f"{prefix}_table.txt", txt_str)
        zf.writestr(f"{prefix}_histograms.png", png_bytes)
        zf.writestr(f"{prefix}_statistics.csv", hist_combined.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr(f"{prefix}_pair_contacts.csv", pairs_combined.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr(f"{prefix}_stoichiometry.csv", dfs["stoichiometry"].to_csv(index=False).encode("utf-8-sig"))
        zf.writestr(f"{prefix}_config.json", sim.config.to_json())
    zip_buf.seek(0)
    return fig, zip_buf.getvalue()


def save_package_to_directory(sim: SimulationResult, out_dir: str, prefix: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    fig, zip_bytes = build_zip_package(sim, prefix=prefix)
    plt.close(fig)

    saved_files: List[str] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for member in zf.namelist():
            target_path = os.path.join(out_dir, member)
            with open(target_path, "wb") as f:
                f.write(zf.read(member))
            saved_files.append(target_path)
    return saved_files
