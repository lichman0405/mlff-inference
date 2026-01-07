"""
MatterSim Inference - Tasks Module

Provides implementations of various computational tasks.
"""

from .static import calculate_single_point
from .dynamics import run_md
from .phonon import calculate_phonon
from .mechanics import calculate_bulk_modulus
from .adsorption import calculate_adsorption_energy

__all__ = [
    "calculate_single_point",
    "run_md",
    "calculate_phonon",
    "calculate_bulk_modulus",
    "calculate_adsorption_energy",
]
