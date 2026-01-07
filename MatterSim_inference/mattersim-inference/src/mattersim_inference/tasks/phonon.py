"""
Phonon calculation module.

Calculates phonon density of states and thermodynamic properties.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from ase import Atoms


def calculate_phonon(
    atoms: Atoms,
    calculator: Any,
    supercell_matrix: List[int] = [2, 2, 2],
    mesh: List[int] = [20, 20, 20],
    displacement: float = 0.01,
    t_min: float = 0,
    t_max: float = 1000,
    t_step: float = 10
) -> Dict[str, Any]:
    """
    Calculate phonon properties.
    
    Args:
        atoms: Unit cell structure (ASE Atoms)
        calculator: ASE Calculator
        supercell_matrix: Supercell size [a, b, c]
        mesh: k-point mesh [kx, ky, kz]
        displacement: Atomic displacement (Å)
        t_min: Minimum temperature (K)
        t_max: Maximum temperature (K)
        t_step: Temperature step (K)
    
    Returns:
        dict: Phonon calculation results
    """
    try:
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
    except ImportError:
        raise ImportError(
            "phonopy is required for phonon calculations. "
            "Install it with: pip install phonopy"
        )
    
    # Convert to PhonopyAtoms
    phonopy_atoms = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.get_cell(),
        scaled_positions=atoms.get_scaled_positions()
    )
    
    # Create Phonopy object
    phonon = Phonopy(
        phonopy_atoms,
        supercell_matrix=np.diag(supercell_matrix),
        factor=15.633302  # THz
    )
    
    # Generate displacements
    phonon.generate_displacements(distance=displacement)
    
    # Calculate forces
    supercells = phonon.supercells_with_displacements
    forces_sets = []
    
    for sc in supercells:
        # Convert back to ASE Atoms
        ase_sc = Atoms(
            symbols=sc.symbols,
            positions=sc.positions,
            cell=sc.cell,
            pbc=True
        )
        ase_sc.calc = calculator
        
        forces = ase_sc.get_forces()
        forces_sets.append(forces)
    
    phonon.forces = forces_sets
    
    # Produce force constants
    phonon.produce_force_constants()
    
    # Calculate density of states
    phonon.run_mesh(mesh)
    phonon.run_total_dos()
    
    dos = phonon.total_dos
    frequency_points = dos.frequency_points
    total_dos = dos.dos
    
    # Check for imaginary frequencies
    min_freq = np.min(frequency_points[total_dos > 1e-6])
    has_imaginary = min_freq < -0.01
    imaginary_modes = np.sum(frequency_points < -0.01)
    
    # Calculate thermodynamic properties
    phonon.run_thermal_properties(
        t_min=t_min,
        t_max=t_max,
        t_step=t_step
    )
    
    thermal = phonon.thermal_properties
    
    thermal_dict = {
        "temperatures": thermal.temperatures,
        "heat_capacity": thermal.heat_capacity,  # J/(mol·K)
        "entropy": thermal.entropy,              # J/(mol·K)
        "free_energy": thermal.free_energy,      # kJ/mol
    }
    
    return {
        "frequency_points": frequency_points,
        "total_dos": total_dos,
        "has_imaginary": has_imaginary,
        "imaginary_modes": int(imaginary_modes),
        "min_frequency": min_freq,
        "thermal": thermal_dict,
    }
