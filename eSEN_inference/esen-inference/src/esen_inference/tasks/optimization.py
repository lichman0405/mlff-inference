"""
Structure optimization for eSEN models
"""

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import LBFGS, BFGS, FIRE
from ase.constraints import ExpCellFilter, StrainFilter
from typing import Dict, Any, Optional
import numpy as np


class OptimizationTask:
    """
    Handler for structure optimization (geometry relaxation).
    
    Supports:
    - Coordinates-only optimization
    - Full optimization (coordinates + cell)
    - Hydrostatic strain optimization
    - Custom constraints
    """
    
    def __init__(self, calculator: Calculator):
        """
        Initialize OptimizationTask.
        
        Args:
            calculator: ASE calculator (OCPCalculator for eSEN)
        """
        self.calculator = calculator
    
    def optimize(
        self,
        atoms: Atoms,
        fmax: float = 0.01,
        optimizer: str = 'LBFGS',
        relax_cell: bool = False,
        max_steps: int = 500,
        trajectory: Optional[str] = None,
        logfile: Optional[str] = None,
        pressure: float = 0.0,
        hydrostatic_strain: bool = False
    ) -> Dict[str, Any]:
        """
        Optimize atomic structure.
        
        Args:
            atoms: Atoms object to optimize (will be modified in-place)
            fmax: Convergence criterion (max force in eV/Å)
            optimizer: Optimizer to use ('LBFGS', 'BFGS', 'FIRE')
            relax_cell: Whether to optimize cell parameters
            max_steps: Maximum optimization steps
            trajectory: Trajectory file path (optional)
            logfile: Log file path (optional)
            pressure: External pressure in GPa (only for relax_cell=True)
            hydrostatic_strain: Only allow isotropic cell changes
        
        Returns:
            Dictionary containing:
            - converged: Whether optimization converged
            - steps: Number of steps taken
            - initial_energy: Initial energy (eV)
            - final_energy: Final energy (eV)
            - energy_change: Energy change (eV)
            - initial_fmax: Initial max force (eV/Å)
            - final_fmax: Final max force (eV/Å)
            - atoms: Optimized Atoms object
            - trajectory: List of Atoms (if trajectory='' passed as empty string)
        """
        # Attach calculator
        atoms.calc = self.calculator
        
        # Get initial state
        initial_energy = atoms.get_potential_energy()
        initial_forces = atoms.get_forces()
        initial_fmax = np.max(np.linalg.norm(initial_forces, axis=1))
        
        # Choose optimizer
        optimizer_map = {
            'LBFGS': LBFGS,
            'BFGS': BFGS,
            'FIRE': FIRE
        }
        
        if optimizer not in optimizer_map:
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. "
                f"Choose from: {list(optimizer_map.keys())}"
            )
        
        OptClass = optimizer_map[optimizer]
        
        # Apply cell relaxation if requested
        if relax_cell:
            # Convert pressure from GPa to eV/Å³
            # 1 GPa = 0.00624150907 eV/Å³
            pressure_eV_A3 = pressure * 0.00624150907
            
            if hydrostatic_strain:
                # Only isotropic cell changes
                atoms_opt = StrainFilter(atoms, mask=[1, 1, 1, 0, 0, 0])
            else:
                # Full cell optimization
                atoms_opt = ExpCellFilter(atoms, scalar_pressure=pressure_eV_A3)
        else:
            atoms_opt = atoms
        
        # Initialize optimizer
        if trajectory == '':
            # Return trajectory as list
            traj_list = []
            opt = OptClass(atoms_opt, logfile=logfile)
            
            def observer():
                traj_list.append(atoms.copy())
            
            opt.attach(observer)
            use_traj_list = True
        elif trajectory is not None:
            opt = OptClass(atoms_opt, trajectory=trajectory, logfile=logfile)
            use_traj_list = False
        else:
            opt = OptClass(atoms_opt, logfile=logfile)
            use_traj_list = False
        
        # Run optimization
        try:
            opt.run(fmax=fmax, steps=max_steps)
            converged = opt.converged()
        except Exception as e:
            converged = False
            raise RuntimeError(f"Optimization failed: {e}") from e
        
        # Get final state
        final_energy = atoms.get_potential_energy()
        final_forces = atoms.get_forces()
        final_fmax = np.max(np.linalg.norm(final_forces, axis=1))
        
        result = {
            'converged': converged,
            'steps': opt.nsteps,
            'initial_energy': float(initial_energy),
            'final_energy': float(final_energy),
            'energy_change': float(final_energy - initial_energy),
            'initial_fmax': float(initial_fmax),
            'final_fmax': float(final_fmax),
            'atoms': atoms
        }
        
        if use_traj_list:
            result['trajectory'] = traj_list
        
        return result
