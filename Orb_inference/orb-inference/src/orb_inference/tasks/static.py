"""Static calculations: single-point energy and structure optimization."""

from typing import Optional, List, Dict, Any
import numpy as np
from ase import Atoms
from ase.optimize import LBFGS, BFGS, FIRE
from ase.constraints import FrechetCellFilter


def single_point_energy(
    atoms: Atoms,
    calculator,
    properties: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Perform single-point energy calculation.
    
    Args:
        atoms: ASE Atoms object with calculator attached
        calculator: ORBCalculator instance
        properties: Properties to calculate (default: all)
        
    Returns:
        Dictionary with:
            - energy: Total energy (eV)
            - energy_per_atom: Energy per atom (eV/atom)
            - forces: Atomic forces (eV/Å), shape (N, 3)
            - max_force: Maximum force magnitude (eV/Å)
            - rms_force: RMS force (eV/Å)
            - stress: Stress tensor (eV/Å³), shape (6,) Voigt
            - pressure_GPa: Pressure (GPa), if stress available
            
    Examples:
        >>> from orb_models.forcefield import pretrained
        >>> from orb_models.forcefield.calculator import ORBCalculator
        >>> orbff = pretrained.orb_v3_conservative_inf_omat(device="cuda")
        >>> calc = ORBCalculator(orbff, device="cuda")
        >>> atoms.calc = calc
        >>> result = single_point_energy(atoms, calc)
        >>> print(f"Energy: {result['energy']:.6f} eV")
    """
    if atoms.calc is None:
        atoms.calc = calculator
    
    # Calculate energy and forces
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    
    # Calculate derived quantities
    energy_per_atom = energy / len(atoms)
    max_force = np.max(np.linalg.norm(forces, axis=1))
    rms_force = np.sqrt(np.mean(np.sum(forces**2, axis=1)))
    
    result = {
        "energy": energy,
        "energy_per_atom": energy_per_atom,
        "forces": forces,
        "max_force": max_force,
        "rms_force": rms_force,
    }
    
    # Calculate stress if requested
    if properties is None or "stress" in properties:
        try:
            stress = atoms.get_stress(voigt=True)  # 6-component Voigt
            result["stress"] = stress
            
            # Calculate pressure (GPa)
            # P = -Tr(σ)/3, convert eV/Å³ to GPa
            pressure_GPa = -np.trace(stress[:3]) / 3 * 160.21766208
            result["pressure_GPa"] = pressure_GPa
        except:
            result["stress"] = None
            result["pressure_GPa"] = None
    else:
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
    logfile: Optional[str] = None,
    output: Optional[str] = None
) -> Atoms:
    """
    Optimize atomic structure.
    
    Args:
        atoms: Initial structure
        calculator: ORBCalculator instance
        fmax: Force convergence criterion (eV/Å)
        steps: Maximum optimization steps
        optimizer: Optimizer name ("LBFGS", "BFGS", "FIRE")
        optimize_cell: Whether to optimize cell parameters
        trajectory: Trajectory file path (.traj)
        logfile: Log file path
        output: Output structure file path
        
    Returns:
        Optimized Atoms object
        
    Examples:
        >>> optimized = optimize_structure(
        ...     atoms, calc, fmax=0.05, optimize_cell=True,
        ...     trajectory="opt.traj", output="optimized.cif"
        ... )
    """
    if atoms.calc is None:
        atoms.calc = calculator
    
    # Select optimizer
    optimizer_map = {
        "LBFGS": LBFGS,
        "BFGS": BFGS,
        "FIRE": FIRE,
    }
    
    if optimizer not in optimizer_map:
        raise ValueError(
            f"Unknown optimizer: {optimizer}. "
            f"Choose from {list(optimizer_map.keys())}"
        )
    
    optimizer_class = optimizer_map[optimizer]
    
    # Optimize cell if requested (using FrechetCellFilter, ASE >= 3.23.0)
    if optimize_cell:
        atoms_to_opt = FrechetCellFilter(atoms)
    else:
        atoms_to_opt = atoms
    
    # Create optimizer
    opt = optimizer_class(
        atoms_to_opt,
        trajectory=trajectory,
        logfile=logfile
    )
    
    # Run optimization
    opt.run(fmax=fmax, steps=steps)
    
    # Save output if requested
    if output is not None:
        from orb_inference.utils.io import save_structure
        save_structure(atoms, output)
    
    return atoms


def calculate_forces(atoms: Atoms, calculator) -> np.ndarray:
    """
    Calculate atomic forces.
    
    Args:
        atoms: ASE Atoms object
        calculator: ORBCalculator instance
        
    Returns:
        Forces array, shape (N, 3), units: eV/Å
    """
    if atoms.calc is None:
        atoms.calc = calculator
    
    return atoms.get_forces()


def calculate_stress(atoms: Atoms, calculator, voigt: bool = True) -> np.ndarray:
    """
    Calculate stress tensor.
    
    Args:
        atoms: ASE Atoms object
        calculator: ORBCalculator instance
        voigt: Return in Voigt notation (6 components)
        
    Returns:
        Stress tensor (eV/Å³)
            - If voigt=True: shape (6,) [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]
            - If voigt=False: shape (3, 3)
    """
    if atoms.calc is None:
        atoms.calc = calculator
    
    return atoms.get_stress(voigt=voigt)
