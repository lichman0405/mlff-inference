"""Adsorption energy and coordination analysis"""

from typing import Union, List, Optional, Dict, Any
import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.optimize import LBFGS


def calculate_adsorption_energy(
    mof_atoms: Atoms,
    gas_molecule: Union[str, Atoms],
    site_position: List[float],
    calculator,
    optimize_complex: bool = True,
    fmax: float = 0.05
) -> Dict[str, Any]:
    """
    Calculate gas adsorption energy in MOF.
    
    E_ads = E(MOF+gas) - E(MOF) - E(gas)
    
    Args:
        mof_atoms: MOF structure (ASE Atoms)
        gas_molecule: Gas molecule name (e.g., "CO2") or Atoms object
        site_position: Adsorption site position [x, y, z]
        calculator: ASE calculator
        optimize_complex: Whether to optimize the adsorption complex
        fmax: Force convergence for optimization
        
    Returns:
        Dictionary with adsorption energy and structures
    """
    # 1. Calculate MOF energy
    mof_copy = mof_atoms.copy()
    mof_copy.calc = calculator
    E_mof = mof_copy.get_potential_energy()
    
    # 2. Get gas molecule
    if isinstance(gas_molecule, str):
        gas = molecule(gas_molecule)
    elif isinstance(gas_molecule, Atoms):
        gas = gas_molecule.copy()
    else:
        raise TypeError("gas_molecule must be str or Atoms object")
    
    # Calculate gas energy (in vacuum)
    gas.calc = calculator
    gas.center(vacuum=10.0)  # Add vacuum around molecule
    E_gas = gas.get_potential_energy()
    
    # 3. Create adsorption complex
    complex_atoms = mof_copy.copy()
    
    # Position gas molecule at adsorption site
    gas_centered = gas.copy()
    gas_com = gas_centered.get_center_of_mass()
    translation = np.array(site_position) - gas_com
    gas_centered.translate(translation)
    
    # Combine structures
    complex_atoms += gas_centered
    complex_atoms.calc = calculator
    
    # 4. Optimize complex if requested
    if optimize_complex:
        # Fix MOF atoms, only optimize gas molecule
        from ase.constraints import FixAtoms
        n_mof_atoms = len(mof_atoms)
        constraint = FixAtoms(indices=range(n_mof_atoms))
        complex_atoms.set_constraint(constraint)
        
        opt = LBFGS(complex_atoms)
        opt.run(fmax=fmax)
    
    E_complex = complex_atoms.get_potential_energy()
    
    # 5. Calculate adsorption energy
    E_ads = E_complex - E_mof - E_gas
    
    result = {
        "E_ads": E_ads,              # Adsorption energy (eV)
        "E_mof": E_mof,              # MOF energy (eV)
        "E_gas": E_gas,              # Gas energy (eV)
        "E_complex": E_complex,      # Complex energy (eV)
        "complex_structure": complex_atoms,
        "optimized": optimize_complex
    }
    
    return result


def analyze_coordination(
    atoms: Atoms,
    metal_indices: Optional[List[int]] = None,
    cutoff_multiplier: float = 1.2
) -> Dict[str, Any]:
    """
    Analyze coordination environment around metal centers.
    
    Args:
        atoms: ASE Atoms object
        metal_indices: Indices of metal atoms (auto-detect if None)
        cutoff_multiplier: Multiplier for natural cutoff radii
        
    Returns:
        Dictionary with coordination analysis
    """
    # Auto-detect metal atoms if not provided
    if metal_indices is None:
        # Common metal elements in MOFs
        metal_symbols = ['Cu', 'Zn', 'Zr', 'Fe', 'Ni', 'Co', 'Mn', 'Cr', 'Ti', 'V',
                        'Al', 'Mg', 'Ca', 'Sr', 'Ba', 'Cd', 'Hg', 'Pd', 'Pt']
        metal_indices = [i for i, sym in enumerate(atoms.get_chemical_symbols()) 
                        if sym in metal_symbols]
    
    if not metal_indices:
        raise ValueError("No metal atoms found. Specify metal_indices manually.")
    
    # Create neighbor list with natural cutoffs
    cutoffs = natural_cutoffs(atoms, mult=cutoff_multiplier)
    nl = NeighborList(cutoffs, skin=0.3, self_interaction=False, bothways=False)
    nl.update(atoms)
    
    coordination_data = {}
    
    for metal_idx in metal_indices:
        indices, offsets = nl.get_neighbors(metal_idx)
        
        # Calculate distances
        metal_pos = atoms.positions[metal_idx]
        neighbor_data = []
        
        for neighbor_idx, offset in zip(indices, offsets):
            neighbor_pos = atoms.positions[neighbor_idx] + offset @ atoms.cell
            distance = np.linalg.norm(neighbor_pos - metal_pos)
            
            neighbor_data.append({
                "index": int(neighbor_idx),
                "symbol": atoms[neighbor_idx].symbol,
                "distance": float(distance)
            })
        
        # Sort by distance
        neighbor_data.sort(key=lambda x: x["distance"])
        
        coordination_data[int(metal_idx)] = {
            "metal_symbol": atoms[metal_idx].symbol,
            "coordination_number": len(neighbor_data),
            "neighbors": neighbor_data,
            "average_distance": float(np.mean([n["distance"] for n in neighbor_data])) if neighbor_data else 0.0
        }
    
    result = {
        "coordination": coordination_data,
        "n_metal_centers": len(metal_indices),
        "metal_indices": metal_indices
    }
    
    return result


def find_adsorption_sites(
    atoms: Atoms,
    grid_spacing: float = 0.5,
    probe_radius: float = 1.2
) -> List[np.ndarray]:
    """
    Find potential adsorption sites in porous structure (placeholder).
    
    Note: This is a simplified version. For production use, consider
    tools like Zeo++ or pymatgen's VoronoiNN.
    
    Args:
        atoms: ASE Atoms object
        grid_spacing: Grid spacing for site search (Å)
        probe_radius: Probe radius for accessibility (Å)
        
    Returns:
        List of site positions
    """
    raise NotImplementedError(
        "Automatic adsorption site finding not yet implemented. "
        "Please specify site_position manually."
    )
