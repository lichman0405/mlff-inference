"""
Task modules for eSEN Inference

This package provides task-specific implementations for the 8 core inference tasks:
1. Static calculations (single-point energy/force/stress)
2. Optimization (structure relaxation)
3. Dynamics (molecular dynamics simulations)
4. Phonon (phonon & thermodynamic properties)
5. Mechanics (bulk modulus, elastic constants)
6. Adsorption (adsorption energy calculations)
7. Coordination (coordination environment analysis)
8. High-throughput screening

Each module contains task-specific functions and utilities.
"""

from esen_inference.tasks.static import StaticTask
from esen_inference.tasks.optimization import OptimizationTask
from esen_inference.tasks.dynamics import DynamicsTask, analyze_md_trajectory
from esen_inference.tasks.phonon import PhononTask, plot_phonon_dos, plot_thermal_properties
from esen_inference.tasks.mechanics import MechanicsTask, plot_eos
from esen_inference.tasks.adsorption import AdsorptionTask, find_adsorption_sites

__all__ = [
    "StaticTask",
    "OptimizationTask",
    "DynamicsTask",
    "PhononTask",
    "MechanicsTask",
    "AdsorptionTask",
    "analyze_md_trajectory",
    "plot_phonon_dos",
    "plot_thermal_properties",
    "plot_eos",
    "find_adsorption_sites",
]
