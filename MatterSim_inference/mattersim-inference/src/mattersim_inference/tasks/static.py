"""
Single-point calculation module.

Performs single-point calculations for energy, forces, and stress.
"""

from typing import Any, Dict

import numpy as np
from ase import Atoms


def calculate_single_point(
    atoms: Atoms,
    calculator: Any
) -> Dict[str, Any]:
    """
    Perform single-point calculation.
    
    Args:
        atoms: ASE Atoms object (with calculator set)
        calculator: ASE Calculator
    
    Returns:
        dict: Calculation results
            - energy: Total energy (eV)
            - energy_per_atom: Energy per atom (eV/atom)
            - forces: Force array (N, 3) (eV/Å)
            - stress: Stress tensor (6,) (eV/Å³)
            - max_force: Maximum force component (eV/Å)
            - rms_force: RMS force (eV/Å)
            - pressure: Pressure (GPa)
    """
    atoms = atoms.copy()
    atoms.calc = calculator
    
    # Calculate energy
    energy = atoms.get_potential_energy()
    
    # Calculate forces
    forces = atoms.get_forces()
    
    # Calculate stress
    try:
        stress = atoms.get_stress()
    except Exception:
        stress = np.zeros(6)
    
    # Calculate derived quantities
    n_atoms = len(atoms)
    energy_per_atom = energy / n_atoms
    
    force_magnitudes = np.linalg.norm(forces, axis=1)
    max_force = np.max(np.abs(forces))
    rms_force = np.sqrt(np.mean(forces**2))
    
    # Pressure (GPa) = -Tr(stress) / 3
    # stress unit is eV/Å³, convert to GPa: 1 eV/Å³ = 160.2176634 GPa
    pressure = -np.mean(stress[:3]) * 160.2176634
    
    return {
        "energy": energy,
        "energy_per_atom": energy_per_atom,
        "forces": forces,
        "stress": stress,
        "max_force": max_force,
        "rms_force": rms_force,
        "pressure": pressure,
        "force_magnitudes": force_magnitudes,
    }
