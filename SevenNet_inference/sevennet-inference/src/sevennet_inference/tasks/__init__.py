"""
SevenNet Inference - Tasks Module

Provides implementations of various computational tasks.
"""

from .static import (
    calculate_single_point,
    optimize_structure,
    batch_optimize,
    relax_with_constraints,
)

from .dynamics import (
    run_md,
    run_nve_md,
    run_nvt_md,
    run_npt_md,
    equilibrate_temperature,
)

from .phonon import (
    calculate_phonon,
    calculate_thermal_properties,
    save_thermal_properties,
    plot_phonon_bands,
    check_negative_frequencies,
    ase_to_phonopy,
    phonopy_to_ase,
)

from .mechanics import (
    calculate_bulk_modulus,
    calculate_elastic_constants,
    calculate_shear_modulus,
    calculate_elastic_moduli,
    plot_equation_of_state,
)

__all__ = [
    # Static calculations
    "calculate_single_point",
    "optimize_structure",
    "batch_optimize",
    "relax_with_constraints",
    # Molecular dynamics
    "run_md",
    "run_nve_md",
    "run_nvt_md",
    "run_npt_md",
    "equilibrate_temperature",
    # Phonon calculations
    "calculate_phonon",
    "calculate_thermal_properties",
    "save_thermal_properties",
    "plot_phonon_bands",
    "check_negative_frequencies",
    "ase_to_phonopy",
    "phonopy_to_ase",
    # Mechanical properties
    "calculate_bulk_modulus",
    "calculate_elastic_constants",
    "calculate_shear_modulus",
    "calculate_elastic_moduli",
    "plot_equation_of_state",
]
