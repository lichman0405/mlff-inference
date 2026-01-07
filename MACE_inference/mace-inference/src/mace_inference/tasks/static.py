"""Static calculations: single-point energy and structure optimization"""

from typing import Optional, List, Dict, Any
import numpy as np
from ase import Atoms
from ase.optimize import LBFGS, BFGS, FIRE
from ase.filters import FrechetCellFilter


def single_point_energy(
    atoms: Atoms,
    calculator,
    properties: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate single-point energy, forces, and stress.
    
    Args:
        atoms: ASE Atoms object
        calculator: ASE calculator (MACE or combined)
        properties: List of properties to calculate
        
    Returns:
        Dictionary with calculated properties
    """
    if properties is None:
        properties = ["energy", "forces", "stress"]
    
    atoms.calc = calculator
    
    result = {}
    
    if "energy" in properties:
        result["energy"] = atoms.get_potential_energy()
        result["energy_per_atom"] = result["energy"] / len(atoms)
    
    if "forces" in properties:
        forces = atoms.get_forces()
        result["forces"] = forces
        result["max_force"] = float(np.max(np.abs(forces)))
        result["rms_force"] = float(np.sqrt(np.mean(forces**2)))
    
    if "stress" in properties:
        try:
            stress = atoms.get_stress(voigt=True)
            result["stress"] = stress
            result["pressure_GPa"] = -np.mean(stress[:3]) * 160.21766208  # eV/Å³ to GPa
        except Exception:
            result["stress"] = None
            result["pressure_GPa"] = None
    
    return result


def optimize_structure(
    atoms: Atoms,
    calculator,
    fmax: float = 0.05,
    steps: int = 500,
    optimizer: str = "LBFGS",
    optimize_cell: bool = False,
    trajectory: Optional[str] = None,
    logfile: Optional[str] = None
) -> Atoms:
    """
    Optimize atomic structure.
    
    Args:
        atoms: Input ASE Atoms object
        calculator: ASE calculator
        fmax: Force convergence criterion (eV/Å)
        steps: Maximum optimization steps
        optimizer: Optimizer name ("LBFGS", "BFGS", "FIRE")
        optimize_cell: Whether to optimize cell parameters
        trajectory: Path to save optimization trajectory
        logfile: Path to save optimization log
        
    Returns:
        Optimized Atoms object
    """
    # Make a copy to avoid modifying original
    atoms = atoms.copy()
    atoms.calc = calculator
    
    # Select optimizer
    optimizer_map = {
        "LBFGS": LBFGS,
        "BFGS": BFGS,
        "FIRE": FIRE
    }
    
    if optimizer not in optimizer_map:
        raise ValueError(
            f"Unknown optimizer: {optimizer}. Choose from {list(optimizer_map.keys())}"
        )
    
    OptClass = optimizer_map[optimizer]
    
    # Apply cell filter if optimizing cell
    if optimize_cell:
        atoms_to_optimize = FrechetCellFilter(atoms)
    else:
        atoms_to_optimize = atoms
    
    # Create optimizer
    opt = OptClass(
        atoms_to_optimize,
        trajectory=trajectory,
        logfile=logfile
    )
    
    # Run optimization
    opt.run(fmax=fmax, steps=steps)
    
    # Return the original atoms object (which has been modified in-place)
    return atoms
