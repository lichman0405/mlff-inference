"""Mechanical properties calculations"""

from typing import Dict
import numpy as np
from ase import Atoms
from ase.eos import EquationOfState


def calculate_bulk_modulus(
    atoms: Atoms,
    calculator,
    n_points: int = 11,
    scale_range: tuple = (0.95, 1.05),
    eos_type: str = "birchmurnaghan"
) -> Dict[str, float]:
    """
    Calculate bulk modulus using equation of state.
    
    Args:
        atoms: Input ASE Atoms object
        calculator: ASE calculator
        n_points: Number of volume points
        scale_range: Volume scaling range (min_scale, max_scale)
        eos_type: Equation of state type ("birchmurnaghan", "murnaghan", etc.)
        
    Returns:
        Dictionary with equilibrium volume, energy, and bulk modulus
    """
    atoms = atoms.copy()
    atoms.calc = calculator
    
    # Generate volume points
    scale_factors = np.linspace(scale_range[0], scale_range[1], n_points)
    
    volumes = []
    energies = []
    
    # Calculate energy at each volume
    original_cell = atoms.get_cell()
    
    for scale in scale_factors:
        # Scale cell uniformly
        scaled_atoms = atoms.copy()
        scaled_atoms.set_cell(original_cell * scale, scale_atoms=True)
        scaled_atoms.calc = calculator
        
        volumes.append(scaled_atoms.get_volume())
        energies.append(scaled_atoms.get_potential_energy())
    
    # Fit equation of state
    eos = EquationOfState(volumes, energies, eos=eos_type)
    v0, e0, B = eos.fit()
    
    # Convert bulk modulus from eV/Å³ to GPa
    B_GPa = B * 160.21766208
    
    result = {
        "v0": v0,           # Equilibrium volume (Å³)
        "e0": e0,           # Equilibrium energy (eV)
        "B": B,             # Bulk modulus (eV/Å³)
        "B_GPa": B_GPa,     # Bulk modulus (GPa)
        "volumes": volumes,
        "energies": energies,
        "eos_type": eos_type
    }
    
    return result


def calculate_elastic_constants(
    atoms: Atoms,
    calculator,
    delta: float = 0.01,
    symmetry: str = "cubic"
) -> Dict[str, np.ndarray]:
    """
    Calculate elastic constants (placeholder for future implementation).
    
    Note: This requires more sophisticated strain analysis.
    For now, use bulk_modulus for isotropic properties.
    
    Args:
        atoms: Input ASE Atoms object
        calculator: ASE calculator
        delta: Strain magnitude
        symmetry: Crystal symmetry type
        
    Returns:
        Dictionary with elastic constants
    """
    raise NotImplementedError(
        "Full elastic constant calculation not yet implemented. "
        "Use calculate_bulk_modulus() for bulk properties."
    )
