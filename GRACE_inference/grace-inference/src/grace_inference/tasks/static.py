"""Static calculations: single-point energy and structure optimization"""

from typing import Optional, List, Dict, Any
import numpy as np
from ase import Atoms
from ase.optimize import LBFGS, BFGS, FIRE
from ase.constraints import FrechetCellFilter, StrainFilter


def calculate_single_point(
    atoms: Atoms,
    calculator,
    properties: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate single-point energy, forces, and stress.
    
    Args:
        atoms: ASE Atoms object
        calculator: ASE calculator (GRACE calculator)
        properties: List of properties to calculate (default: ["energy", "forces", "stress"])
        
    Returns:
        Dictionary with calculated properties:
            - energy: Total energy (eV)
            - energy_per_atom: Energy per atom (eV/atom)
            - forces: Force array (eV/Å)
            - max_force: Maximum force magnitude (eV/Å)
            - rms_force: RMS force (eV/Å)
            - stress: Stress tensor in Voigt notation (eV/Å³)
            - pressure_GPa: Pressure (GPa)
        
    Examples:
        >>> from grace_inference import GRACEInference
        >>> calc = GRACEInference(model_name="grace-2l")
        >>> result = calculate_single_point(atoms, calc.calculator)
        >>> print(f"Energy: {result['energy']:.6f} eV")
    """
    if properties is None:
        properties = ["energy", "forces", "stress"]
    
    # Attach calculator
    atoms.calc = calculator
    
    result = {}
    
    if "energy" in properties:
        energy = atoms.get_potential_energy()
        result["energy"] = float(energy)
        result["energy_per_atom"] = float(energy / len(atoms))
    
    if "forces" in properties:
        forces = atoms.get_forces()
        result["forces"] = forces
        result["max_force"] = float(np.max(np.abs(forces)))
        result["rms_force"] = float(np.sqrt(np.mean(forces**2)))
    
    if "stress" in properties:
        try:
            stress = atoms.get_stress(voigt=True)  # 6-component Voigt notation
            result["stress"] = stress
            # Calculate pressure: P = -1/3 * trace(stress tensor)
            # Convert eV/Å³ to GPa: 1 eV/Å³ = 160.21766208 GPa
            result["pressure_GPa"] = float(-np.mean(stress[:3]) * 160.21766208)
        except Exception as e:
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
        
    Raises:
        ValueError: If optimizer is not recognized
        
    Examples:
        >>> from grace_inference import GRACEInference
        >>> calc = GRACEInference(model_name="grace-2l")
        >>> optimized = optimize_structure(atoms, calc.calculator, fmax=0.01)
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
        atoms_filtered = FrechetCellFilter(atoms)
    else:
        atoms_filtered = atoms
    
    # Create optimizer
    opt = OptClass(
        atoms_filtered,
        trajectory=trajectory,
        logfile=logfile
    )
    
    # Run optimization
    opt.run(fmax=fmax, steps=steps)
    
    return atoms


def batch_optimize(
    structures: List[Atoms],
    calculator,
    fmax: float = 0.05,
    steps: int = 500,
    optimizer: str = "LBFGS",
    optimize_cell: bool = False,
    output_dir: Optional[str] = None
) -> List[Atoms]:
    """
    Optimize multiple structures in batch.
    
    Args:
        structures: List of ASE Atoms objects
        calculator: ASE calculator
        fmax: Force convergence criterion (eV/Å)
        steps: Maximum optimization steps
        optimizer: Optimizer name
        optimize_cell: Whether to optimize cell parameters
        output_dir: Directory to save optimized structures
        
    Returns:
        List of optimized Atoms objects
        
    Examples:
        >>> structures = [atoms1, atoms2, atoms3]
        >>> optimized = batch_optimize(structures, calc.calculator)
    """
    from pathlib import Path
    
    optimized_structures = []
    
    for i, atoms in enumerate(structures):
        print(f"Optimizing structure {i+1}/{len(structures)}...")
        
        trajectory = None
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            trajectory = str(output_path / f"opt_{i:03d}.traj")
        
        optimized = optimize_structure(
            atoms, calculator, fmax, steps, optimizer,
            optimize_cell, trajectory
        )
        
        optimized_structures.append(optimized)
        
        if output_dir:
            from ..utils.io import write_structure
            output_file = output_path / f"optimized_{i:03d}.cif"
            write_structure(optimized, output_file)
    
    return optimized_structures


def relax_with_constraints(
    atoms: Atoms,
    calculator,
    fix_atoms: Optional[List[int]] = None,
    fix_cell: bool = False,
    fmax: float = 0.05,
    steps: int = 500,
    optimizer: str = "LBFGS",
    trajectory: Optional[str] = None
) -> Atoms:
    """
    Optimize structure with constraints.
    
    Args:
        atoms: Input ASE Atoms object
        calculator: ASE calculator
        fix_atoms: List of atom indices to fix
        fix_cell: Whether to fix cell parameters
        fmax: Force convergence criterion (eV/Å)
        steps: Maximum optimization steps
        optimizer: Optimizer name
        trajectory: Path to save optimization trajectory
        
    Returns:
        Optimized Atoms object with constraints
        
    Examples:
        >>> # Fix first 10 atoms (e.g., substrate)
        >>> optimized = relax_with_constraints(
        ...     atoms, calc.calculator, fix_atoms=list(range(10))
        ... )
    """
    from ase.constraints import FixAtoms
    
    atoms = atoms.copy()
    atoms.calc = calculator
    
    # Apply constraints
    if fix_atoms:
        constraint = FixAtoms(indices=fix_atoms)
        atoms.set_constraint(constraint)
    
    # Optimize
    return optimize_structure(
        atoms, calculator, fmax, steps, optimizer,
        optimize_cell=not fix_cell, trajectory=trajectory
    )
