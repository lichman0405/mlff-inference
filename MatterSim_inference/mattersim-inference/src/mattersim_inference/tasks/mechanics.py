"""
Mechanical properties module.

Calculates bulk modulus and equation of state.
"""

from typing import Any, Dict

import numpy as np
from ase import Atoms
from ase.eos import EquationOfState


def calculate_bulk_modulus(
    atoms: Atoms,
    calculator: Any,
    strain_range: float = 0.05,
    npoints: int = 11,
    eos: str = "birchmurnaghan"
) -> Dict[str, Any]:
    """
    Calculate bulk modulus.
    
    Obtains bulk modulus by fitting equation of state from energies at different volumes.
    
    Args:
        atoms: ASE Atoms object
        calculator: ASE Calculator
        strain_range: Strain range (±), e.g., 0.05 means ±5%
        npoints: Number of sampling points
        eos: Equation of state type
            - "birchmurnaghan": Birch-Murnaghan (default)
            - "vinet": Vinet EOS
            - "murnaghan": Murnaghan EOS
            - "sj": Stabilized Jellium
    
    Returns:
        dict: Bulk modulus calculation results
            - bulk_modulus: Bulk modulus (GPa)
            - v0: Equilibrium volume (Å³)
            - e0: Equilibrium energy (eV)
            - eos: EOS type used
            - volumes: Volume array
            - energies: Energy array
    """
    atoms = atoms.copy()
    atoms.calc = calculator
    
    # Original volume
    original_volume = atoms.get_volume()
    original_cell = atoms.get_cell().copy()
    
    # Generate different volumes
    strains = np.linspace(-strain_range, strain_range, npoints)
    volumes = []
    energies = []
    
    for strain in strains:
        # Scale unit cell
        scale = (1 + strain) ** (1/3)
        scaled_cell = original_cell * scale
        
        # Create scaled structure
        scaled_atoms = atoms.copy()
        scaled_atoms.set_cell(scaled_cell, scale_atoms=True)
        scaled_atoms.calc = calculator
        
        # Calculate energy
        energy = scaled_atoms.get_potential_energy()
        volume = scaled_atoms.get_volume()
        
        volumes.append(volume)
        energies.append(energy)
    
    volumes = np.array(volumes)
    energies = np.array(energies)
    
    # Fit equation of state
    try:
        eos_fit = EquationOfState(volumes, energies, eos=eos)
        v0, e0, B = eos_fit.fit()
        
        # Convert bulk modulus units: eV/Å³ -> GPa
        # 1 eV/Å³ = 160.2176634 GPa
        bulk_modulus_GPa = B * 160.2176634
        
    except Exception as e:
        return {
            "error": str(e),
            "volumes": volumes,
            "energies": energies,
        }
    
    return {
        "bulk_modulus": bulk_modulus_GPa,
        "v0": v0,
        "e0": e0,
        "eos": eos,
        "volumes": volumes,
        "energies": energies,
    }


def calculate_elastic_constants(
    atoms: Atoms,
    calculator: Any,
    delta: float = 0.01
) -> Dict[str, Any]:
    """
    Calculate elastic constant tensor (simplified version).
    
    Args:
        atoms: ASE Atoms object
        calculator: ASE Calculator
        delta: Strain increment
    
    Returns:
        dict: Elastic constants
    """
    from ase.constraints import StrainFilter
    
    atoms = atoms.copy()
    atoms.calc = calculator
    
    # Simplified implementation: only returns bulk modulus
    # Full elastic tensor calculation requires more complex strain patterns
    
    bm_result = calculate_bulk_modulus(
        atoms, calculator, strain_range=0.03, npoints=7
    )
    
    return {
        "bulk_modulus": bm_result.get("bulk_modulus"),
        "note": "Full elastic tensor calculation not implemented",
    }
