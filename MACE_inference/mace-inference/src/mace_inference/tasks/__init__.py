"""Task modules for MACE inference"""

from mace_inference.tasks.static import single_point_energy, optimize_structure
from mace_inference.tasks.dynamics import run_nvt_md, run_npt_md
from mace_inference.tasks.phonon import calculate_phonon, calculate_thermal_properties
from mace_inference.tasks.mechanics import calculate_bulk_modulus
from mace_inference.tasks.adsorption import calculate_adsorption_energy, analyze_coordination

__all__ = [
    "single_point_energy",
    "optimize_structure",
    "run_nvt_md",
    "run_npt_md",
    "calculate_phonon",
    "calculate_thermal_properties",
    "calculate_bulk_modulus",
    "calculate_adsorption_energy",
    "analyze_coordination",
]
