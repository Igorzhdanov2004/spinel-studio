"""
presets.py — Слой данных, кристаллографических констант и пресетов составов
для генератора неупорядоченных суперъячеек шпинели (Spinel Studio).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


ATOMIC_MASSES: Dict[str, float] = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906,
    "Mo": 95.95, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42, "Ag": 107.87,
    "Cd": 112.41, "In": 114.82, "Sn": 118.71, "Sb": 121.76, "Te": 127.60,
    "I": 126.90, "Cs": 132.91, "Ba": 137.33, "La": 138.91, "Ce": 140.12,
    "Pr": 140.91, "Nd": 144.24, "Sm": 150.36, "Eu": 151.96, "Gd": 157.25,
    "Tb": 158.93, "Dy": 162.50, "Ho": 164.93, "Er": 167.26, "Tm": 168.93,
    "Yb": 173.05, "Lu": 174.97, "Hf": 178.49, "Ta": 180.95, "W": 183.84,
    "Re": 186.21, "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97,
    "Hg": 200.59, "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "U": 238.03,
}

DEFAULT_ANION_SYMBOLS = {"O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N"}

LATTICE_TEMPLATES: Dict[str, Tuple[float, float, float]] = {
    "NiCr2O4_2x2x4 (16.598 × 16.598 × 33.196 Å)": (16.598, 16.598, 33.196),
    "MgCr2O4_2x2x4 (16.674 × 16.674 × 33.348 Å)": (16.674, 16.674, 33.348),
}


@dataclass
class SpeciesSpec:
    element: str
    count: int
    charge: float

    def __post_init__(self) -> None:
        self.element = str(self.element).strip()
        self.count = int(self.count)
        self.charge = float(self.charge)
        if not self.element:
            raise ValueError("Символ элемента не может быть пустым.")
        if self.count < 0:
            raise ValueError(f"Количество атомов '{self.element}' не может быть отрицательным ({self.count}).")


@dataclass
class CompositionConfig:
    key: str
    title: str
    description: str
    lattice: Tuple[float, float, float] = (16.598, 16.598, 33.196)
    angles: Tuple[float, float, float] = (90.0, 90.0, 90.0)
    supercell_dims: Tuple[int, int, int] = (2, 2, 4)
    tet_species: List[SpeciesSpec] = field(default_factory=list)
    oct_species: List[SpeciesSpec] = field(default_factory=list)
    anion_element: str = "O"
    anion_charge: float = -1.2
    r_tet: float = 3.90
    r_oct: float = 3.25
    is_model_labels: bool = False

    def active_tet(self) -> List[SpeciesSpec]:
        return [s for s in self.tet_species if s.count > 0]

    def active_oct(self) -> List[SpeciesSpec]:
        return [s for s in self.oct_species if s.count > 0]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["lattice"] = list(self.lattice)
        data["angles"] = list(self.angles)
        data["supercell_dims"] = list(self.supercell_dims)
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "CompositionConfig":
        tet = [SpeciesSpec(**item) for item in data.get("tet_species", [])]
        oct_ = [SpeciesSpec(**item) for item in data.get("oct_species", [])]
        return cls(
            key=str(data.get("key", "custom")),
            title=str(data.get("title", "Пользовательская конфигурация")),
            description=str(data.get("description", "")),
            lattice=tuple(float(x) for x in data.get("lattice", [16.598, 16.598, 33.196])),
            angles=tuple(float(x) for x in data.get("angles", [90.0, 90.0, 90.0])),
            supercell_dims=tuple(int(x) for x in data.get("supercell_dims", [2, 2, 4])),
            tet_species=tet,
            oct_species=oct_,
            anion_element=str(data.get("anion_element", "O")).strip(),
            anion_charge=float(data.get("anion_charge", -1.2)),
            r_tet=float(data.get("r_tet", 3.90)),
            r_oct=float(data.get("r_oct", 3.25)),
            is_model_labels=bool(data.get("is_model_labels", False)),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "CompositionConfig":
        return cls.from_dict(json.loads(json_str))


MN_EXACT_CHARGE: float = 1.802597403


def get_builtin_presets() -> Dict[str, CompositionConfig]:
    presets_list = [
        CompositionConfig(
            key="variant_1",
            title="Вариант 1: Co-Ni-Mn-Fe-Cr (Ni и Cr в октаэдрах)",
            description=(
                "Валовый состав Co77Ni77Mn77Fe77Cr76O512. "
                "Тетраэдры A (128): 39 Co, 12 Mn, 77 Fe. "
                "Октаэдры B (256): 38 Co, 77 Ni, 65 Mn, 76 Cr."
            ),
            lattice=(16.598, 16.598, 33.196),
            tet_species=[
                SpeciesSpec("Co", 39, 1.4),
                SpeciesSpec("Mn", 12, MN_EXACT_CHARGE),
                SpeciesSpec("Fe", 77, 1.8),
            ],
            oct_species=[
                SpeciesSpec("Co", 38, 1.4),
                SpeciesSpec("Ni", 77, 1.2),
                SpeciesSpec("Mn", 65, MN_EXACT_CHARGE),
                SpeciesSpec("Cr", 76, 1.8),
            ],
            anion_element="O",
            anion_charge=-1.2,
            r_tet=3.90,
            r_oct=3.25,
        ),
        CompositionConfig(
            key="variant_2",
            title="Вариант 2: Co-Ni-Mn-Fe-Cr (Co и Ni в октаэдрах)",
            description=(
                "Валовый состав Co77Ni77Mn77Fe77Cr76O512. "
                "Тетраэдры A (128): 11 Cr, 71 Mn, 46 Fe. "
                "Октаэдры B (256): 77 Co, 77 Ni, 6 Mn, 31 Fe, 65 Cr."
            ),
            lattice=(16.598, 16.598, 33.196),
            tet_species=[
                SpeciesSpec("Cr", 11, 1.8),
                SpeciesSpec("Mn", 71, MN_EXACT_CHARGE),
                SpeciesSpec("Fe", 46, 1.8),
            ],
            oct_species=[
                SpeciesSpec("Co", 77, 1.4),
                SpeciesSpec("Ni", 77, 1.2),
                SpeciesSpec("Mn", 6, MN_EXACT_CHARGE),
                SpeciesSpec("Fe", 31, 1.8),
                SpeciesSpec("Cr", 65, 1.8),
            ],
            anion_element="O",
            anion_charge=-1.2,
            r_tet=3.90,
            r_oct=3.25,
        ),
        CompositionConfig(
            key="variant_3",
            title="Вариант 3: Co-Ni-Mn-Fe-Cr (равномерное перемешивание 5 металлов)",
            description=(
                "Валовый состав Co77Ni77Mn77Fe77Cr76O512. "
                "Максимально равномерное распределение всех 5 катионов как по тетраэдрам, так и по октаэдрам."
            ),
            lattice=(16.598, 16.598, 33.196),
            tet_species=[
                SpeciesSpec("Co", 26, 1.4),
                SpeciesSpec("Ni", 26, 1.2),
                SpeciesSpec("Mn", 26, MN_EXACT_CHARGE),
                SpeciesSpec("Fe", 25, 1.8),
                SpeciesSpec("Cr", 25, 1.8),
            ],
            oct_species=[
                SpeciesSpec("Co", 51, 1.4),
                SpeciesSpec("Ni", 51, 1.2),
                SpeciesSpec("Mn", 51, MN_EXACT_CHARGE),
                SpeciesSpec("Fe", 52, 1.8),
                SpeciesSpec("Cr", 51, 1.8),
            ],
            anion_element="O",
            anion_charge=-1.2,
            r_tet=3.90,
            r_oct=3.25,
        ),
        CompositionConfig(
            key="abcde_5comp",
            title="Модельная 5-компонентная система ABCDE",
            description=(
                "Тетраэдры (128): C=11, D=71, E=46 (A=0, B=0). "
                "Октаэдры (256): A=77, B=77, C=65, D=6, E=31."
            ),
            lattice=(16.674, 16.674, 33.348),
            tet_species=[
                SpeciesSpec("C", 11, 1.2),
                SpeciesSpec("D", 71, 1.2),
                SpeciesSpec("E", 46, 1.2),
            ],
            oct_species=[
                SpeciesSpec("A", 77, 1.8),
                SpeciesSpec("B", 77, 1.8),
                SpeciesSpec("C", 65, 1.8),
                SpeciesSpec("D", 6, 1.8),
                SpeciesSpec("E", 31, 1.8),
            ],
            anion_element="O",
            anion_charge=-1.2,
            r_tet=3.95,
            r_oct=3.25,
            is_model_labels=True,
        ),
        CompositionConfig(
            key="ab_32_224",
            title="Бинарная серия AB: 32 A / 224 B (12.5% / 87.5%)",
            description="Тетраэдры полностью заняты Mg (128). В октаэдрах смешение A:B = 32:224.",
            lattice=(16.674, 16.674, 33.348),
            tet_species=[SpeciesSpec("Mg", 128, 1.2)],
            oct_species=[SpeciesSpec("A", 32, 1.8), SpeciesSpec("B", 224, 1.8)],
            anion_element="O",
            anion_charge=-1.2,
            r_tet=3.95,
            r_oct=3.25,
            is_model_labels=True,
        ),
        CompositionConfig(
            key="ab_64_192",
            title="Бинарная серия AB: 64 A / 192 B (25% / 75%)",
            description="Тетраэдры полностью заняты Mg (128). В октаэдрах смешение A:B = 64:192.",
            lattice=(16.674, 16.674, 33.348),
            tet_species=[SpeciesSpec("Mg", 128, 1.2)],
            oct_species=[SpeciesSpec("A", 64, 1.8), SpeciesSpec("B", 192, 1.8)],
            anion_element="O",
            anion_charge=-1.2,
            r_tet=3.95,
            r_oct=3.25,
            is_model_labels=True,
        ),
        CompositionConfig(
            key="ab_85_171",
            title="Бинарная серия AB: 85 A / 171 B (33.2% / 66.8%)",
            description="Тетраэдры полностью заняты Mg (128). В октаэдрах смешение A:B = 85:171 (~1:2).",
            lattice=(16.674, 16.674, 33.348),
            tet_species=[SpeciesSpec("Mg", 128, 1.2)],
            oct_species=[SpeciesSpec("A", 85, 1.8), SpeciesSpec("B", 171, 1.8)],
            anion_element="O",
            anion_charge=-1.2,
            r_tet=3.95,
            r_oct=3.25,
            is_model_labels=True,
        ),
        CompositionConfig(
            key="ab_128_128",
            title="Бинарная серия AB: 128 A / 128 B (50% / 50%)",
            description="Тетраэдры полностью заняты Mg (128). В октаэдрах эквимолярное смешение A:B = 128:128.",
            lattice=(16.674, 16.674, 33.348),
            tet_species=[SpeciesSpec("Mg", 128, 1.2)],
            oct_species=[SpeciesSpec("A", 128, 1.8), SpeciesSpec("B", 128, 1.8)],
            anion_element="O",
            anion_charge=-1.2,
            r_tet=3.95,
            r_oct=3.25,
            is_model_labels=True,
        ),
        CompositionConfig(
            key="custom",
            title="Свой состав с нуля (любые элементы и заряды)",
            description="Базовый шаблон нормальной шпинели AB2O4 для создания собственных конфигураций.",
            lattice=(16.598, 16.598, 33.196),
            tet_species=[SpeciesSpec("Mg", 128, 1.2)],
            oct_species=[SpeciesSpec("Cr", 256, 1.8)],
            anion_element="O",
            anion_charge=-1.2,
            r_tet=3.90,
            r_oct=3.25,
        ),
    ]
    return {cfg.key: cfg for cfg in presets_list}
