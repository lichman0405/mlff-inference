"""
MatterSim Inference - Core Module

Provides the main MatterSimInference class that encapsulates all inference functionality.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS, FIRE, LBFGS
from ase.constraints import UnitCellFilter, ExpCellFilter
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

from .utils.device import get_device, print_device_info
from .utils.io import read_structure, write_structure, validate_structure
from .tasks.static import calculate_single_point
from .tasks.dynamics import run_md as _run_md
from .tasks.phonon import calculate_phonon
from .tasks.mechanics import calculate_bulk_modulus
from .tasks.adsorption import calculate_adsorption_energy


# Available model list
AVAILABLE_MODELS = [
    "MatterSim-v1-1M",
    "MatterSim-v1-5M",
]


class MatterSimInference:
    """
    Main MatterSim inference class.
    
    Encapsulates MatterSim model and provides a unified interface for materials property calculations.
    MatterSim ranks #3 in MOFSimBench and #1 in adsorption energy calculations.
    
    Attributes:
        model_name: Model name
        device: Computing device ('cuda' / 'cpu')
        calculator: ASE Calculator instance
    
    Example:
        >>> calc = MatterSimInference(model_name="MatterSim-v1-5M", device="cuda")
        >>> result = calc.single_point("MOF-5.cif")
        >>> print(f"Energy: {result['energy']:.4f} eV")
    """
    
    def __init__(
        self,
        model_name: str = "MatterSim-v1-5M",
        device: str = "auto",
        **kwargs
    ) -> None:
        """
        Initialize MatterSimInference.
        
        Args:
            model_name: Model name
                - "MatterSim-v1-1M": 1M parameter lightweight version
                - "MatterSim-v1-5M": 5M parameter standard version (default/recommended)
            device: Computing device
                - "auto": Auto detect (GPU preferred)
                - "cuda": Force use GPU
                - "cpu": Force use CPU
            **kwargs: Additional parameters passed to MatterSimCalculator
        """
        self.model_name = model_name
        self.device = get_device(device)
        self._kwargs = kwargs
        
        # Validate model name
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {AVAILABLE_MODELS}"
            )
        
        # Initialize calculator
        self._calculator = None
        self._init_calculator()
    
    def _init_calculator(self) -> None:
        """Initialize MatterSim calculator."""
        try:
            from mattersim.forcefield import MatterSimCalculator
            
            self._calculator = MatterSimCalculator(
                load_path=self.model_name,
                device=self.device,
                **self._kwargs
            )
        except ImportError:
            raise ImportError(
                "MatterSim is not installed. "
                "Please install it with: pip install mattersim"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MatterSim calculator: {e}")
    
    @property
    def calculator(self):
        """Return ASE Calculator instance."""
        return self._calculator
    
    def __repr__(self) -> str:
        return f"MatterSimInference(model={self.model_name}, device={self.device})"
    
    def _prepare_atoms(self, atoms: Union[Atoms, str, Path]) -> Atoms:
        """Prepare Atoms object."""
        if isinstance(atoms, (str, Path)):
            atoms = read_structure(atoms)
        atoms = atoms.copy()
        atoms.calc = self._calculator
        return atoms
    
    # =========================================================================
    # Task 1: Single Point Calculation
    # =========================================================================
    
    def single_point(
        self,
        atoms: Union[Atoms, str, Path]
    ) -> Dict[str, Any]:
        """
        Single point energy calculation.
        
        Args:
            atoms: ASE Atoms object or structure file path
        
        Returns:
            dict: Contains the following keys:
                - energy: Total energy (eV)
                - energy_per_atom: Energy per atom (eV/atom)
                - forces: Force array (N, 3) (eV/Å)
                - stress: Stress tensor (6,) (eV/Å³)
                - max_force: Maximum force component (eV/Å)
                - rms_force: RMS force (eV/Å)
                - pressure: Pressure (GPa)
        """
        atoms = self._prepare_atoms(atoms)
        return calculate_single_point(atoms, self._calculator)
    
    # =========================================================================
    # Task 2: Structure Optimization
    # =========================================================================
    
    def optimize(
        self,
        atoms: Union[Atoms, str, Path],
        fmax: float = 0.05,
        optimizer: str = "LBFGS",
        optimize_cell: bool = False,
        max_steps: int = 500,
        output: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Structure optimization.
        
        Args:
            atoms: ASE Atoms object or structure file path
            fmax: Force convergence threshold (eV/Å)
            optimizer: Optimizer type ("LBFGS", "BFGS", "FIRE")
            optimize_cell: Whether to optimize cell simultaneously
            max_steps: Maximum optimization steps
            output: Output file path (optional)
        
        Returns:
            dict: Contains the following keys:
                - converged: Whether converged
                - steps: Optimization steps
                - initial_energy: Initial energy (eV)
                - final_energy: Final energy (eV)
                - energy_change: Energy change (eV)
                - final_fmax: Final maximum force (eV/Å)
                - atoms: Optimized Atoms object
        """
        atoms = self._prepare_atoms(atoms)
        initial_energy = atoms.get_potential_energy()
        
        # Select optimizer
        optimizers = {
            "LBFGS": LBFGS,
            "BFGS": BFGS,
            "FIRE": FIRE,
        }
        
        if optimizer.upper() not in optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        OptClass = optimizers[optimizer.upper()]
        
        # Set optimization object
        if optimize_cell:
            opt_atoms = ExpCellFilter(atoms)
        else:
            opt_atoms = atoms
        
        # Run optimization
        opt = OptClass(opt_atoms, logfile=None)
        converged = opt.run(fmax=fmax, steps=max_steps)
        
        final_energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        max_force = np.max(np.abs(forces))
        
        # Save output
        if output:
            write_structure(atoms, output)
        
        return {
            "converged": converged,
            "steps": opt.nsteps,
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_change": final_energy - initial_energy,
            "final_fmax": max_force,
            "atoms": atoms,
        }
    
    # =========================================================================
    # Task 3: Molecular Dynamics
    # =========================================================================
    
    def run_md(
        self,
        atoms: Union[Atoms, str, Path],
        ensemble: str = "nvt",
        temperature: float = 300.0,
        pressure: Optional[float] = None,
        steps: int = 10000,
        timestep: float = 1.0,
        trajectory: Optional[str] = None,
        logfile: Optional[str] = None,
        log_interval: int = 100
    ) -> Atoms:
        """
        Molecular dynamics simulation.
        
        MatterSim ranks #1 in MD stability in MOFSimBench (tied with eSEN).
        
        Args:
            atoms: ASE Atoms object or structure file path
            ensemble: Ensemble type ("nve", "nvt", "npt")
            temperature: Temperature (K)
            pressure: Pressure (GPa), required for NPT only
            steps: Simulation steps
            timestep: Time step (fs)
            trajectory: Trajectory file path
            logfile: Log file path
            log_interval: Log interval
        
        Returns:
            Atoms: Final structure
        """
        atoms = self._prepare_atoms(atoms)
        return _run_md(
            atoms=atoms,
            calculator=self._calculator,
            ensemble=ensemble,
            temperature=temperature,
            pressure=pressure,
            steps=steps,
            timestep=timestep,
            trajectory=trajectory,
            logfile=logfile,
            log_interval=log_interval
        )
    
    # =========================================================================
    # Task 4: Phonon Calculation
    # =========================================================================
    
    def phonon(
        self,
        atoms: Union[Atoms, str, Path],
        supercell_matrix: List[int] = [2, 2, 2],
        mesh: List[int] = [20, 20, 20],
        displacement: float = 0.01,
        t_min: float = 0,
        t_max: float = 1000,
        t_step: float = 10
    ) -> Dict[str, Any]:
        """
        Phonon calculation.
        
        Args:
            atoms: Primitive cell structure
            supercell_matrix: Supercell size [a, b, c]
            mesh: k-point mesh [kx, ky, kz]
            displacement: Atomic displacement (Å)
            t_min: Minimum temperature (K)
            t_max: Maximum temperature (K)
            t_step: Temperature step (K)
        
        Returns:
            dict: Contains the following keys:
                - frequency_points: Frequency points (THz)
                - total_dos: Density of states
                - has_imaginary: Whether has imaginary frequencies
                - imaginary_modes: Number of imaginary modes
                - thermal: Thermodynamic properties dictionary
        """
        atoms = self._prepare_atoms(atoms)
        return calculate_phonon(
            atoms=atoms,
            calculator=self._calculator,
            supercell_matrix=supercell_matrix,
            mesh=mesh,
            displacement=displacement,
            t_min=t_min,
            t_max=t_max,
            t_step=t_step
        )
    
    # =========================================================================
    # Task 5: Mechanical Properties
    # =========================================================================
    
    def bulk_modulus(
        self,
        atoms: Union[Atoms, str, Path],
        strain_range: float = 0.05,
        npoints: int = 11,
        eos: str = "birchmurnaghan"
    ) -> Dict[str, Any]:
        """
        Bulk modulus calculation.
        
        Args:
            atoms: ASE Atoms object or structure file path
            strain_range: Strain range (±)
            npoints: Number of sampling points
            eos: Equation of state type
        
        Returns:
            dict: Contains bulk_modulus, v0, e0, etc.
        """
        atoms = self._prepare_atoms(atoms)
        return calculate_bulk_modulus(
            atoms=atoms,
            calculator=self._calculator,
            strain_range=strain_range,
            npoints=npoints,
            eos=eos
        )
    
    # =========================================================================
    # Task 6: Adsorption Energy Calculation (MatterSim's Strongest Area)
    # =========================================================================
    
    def adsorption_energy(
        self,
        mof_structure: Union[Atoms, str, Path],
        gas_molecule: str,
        site_position: List[float],
        optimize_complex: bool = True,
        fmax: float = 0.05
    ) -> Dict[str, Any]:
        """
        Adsorption energy calculation.
        
        **MatterSim performs best on this task** (#1 in MOFSimBench).
        
        Args:
            mof_structure: MOF structure
            gas_molecule: Gas molecule name ("CO2", "H2O", "CH4", ...)
            site_position: Adsorption site coordinates [x, y, z]
            optimize_complex: Whether to optimize complex
            fmax: Optimization convergence threshold (eV/Å)
        
        Returns:
            dict: Contains the following keys:
                - E_ads: Adsorption energy (eV)
                - E_mof: MOF energy (eV)
                - E_gas: Gas molecule energy (eV)
                - E_complex: Complex energy (eV)
                - complex_atoms: Complex structure
        """
        mof = self._prepare_atoms(mof_structure)
        return calculate_adsorption_energy(
            mof=mof,
            gas=gas_molecule,
            site=site_position,
            calculator=self._calculator,
            optimize=optimize_complex,
            fmax=fmax
        )
    
    # =========================================================================
    # Task 7: Coordination Analysis
    # =========================================================================
    
    def coordination(
        self,
        atoms: Union[Atoms, str, Path],
        cutoff: float = 3.0
    ) -> Dict[str, Any]:
        """
        Coordination environment analysis.
        
        Args:
            atoms: ASE Atoms object or structure file path
            cutoff: Cutoff distance for coordination determination (Å)
        
        Returns:
            dict: Contains coordination, metal_indices, etc.
        """
        atoms = self._prepare_atoms(atoms)
        
        # Identify metal atoms
        from ase.data import atomic_numbers
        metal_symbols = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',
                         'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra',
                         'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                         'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                         'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
                         'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                         'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                         'Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Po']
        
        symbols = atoms.get_chemical_symbols()
        metal_indices = [i for i, s in enumerate(symbols) if s in metal_symbols]
        
        coordination_info = {}
        
        for metal_idx in metal_indices:
            metal_pos = atoms.positions[metal_idx]
            neighbors = []
            distances = []
            
            for j, pos in enumerate(atoms.positions):
                if j != metal_idx:
                    dist = np.linalg.norm(pos - metal_pos)
                    if dist < cutoff:
                        neighbors.append(j)
                        distances.append(dist)
            
            coordination_info[metal_idx] = {
                "coordination_number": len(neighbors),
                "neighbors": neighbors,
                "distances": distances,
                "average_distance": np.mean(distances) if distances else 0.0,
            }
        
        return {
            "coordination": coordination_info,
            "metal_indices": metal_indices,
        }


def get_available_models() -> List[str]:
    """Return available models list."""
    return AVAILABLE_MODELS.copy()
