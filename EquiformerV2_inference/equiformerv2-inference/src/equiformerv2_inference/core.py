"""
EquiformerV2 Inference - Core Module

Main inference class for EquiformerV2 calculations.
"""

from pathlib import Path
from typing import Union, Optional, Dict, Any, List

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from .utils import get_device, read_structure, write_structure
from .tasks import (
    calculate_single_point,
    run_md,
    calculate_phonon,
    calculate_bulk_modulus,
)


class EquiformerV2Inference:
    """
    EquiformerV2 Inference Calculator
    
    A unified interface for EquiformerV2 equivariant transformer force field calculations.
    EquiformerV2 is a next-generation SO(3)-equivariant transformer architecture designed
    for molecular modeling, trained on the Open Catalyst Project dataset.
    
    Args:
        model_name: Model name (e.g., "EquiformerV2-31M-S2EF", "EquiformerV2-153M-S2EF")
        model_path: Custom model checkpoint path (optional)
        device: Computing device ("auto", "cuda", "cpu", "cuda:0", etc.)
        dtype: Data type ("float32" or "float64")
    
    Example:
        >>> from equiformerv2_inference import EquiformerV2Inference
        >>> calc = EquiformerV2Inference(model_name="EquiformerV2-31M-S2EF", device="cuda")
        >>> result = calc.single_point("MOF-5.cif")
        >>> print(result['energy'])
    """
    
    def __init__(
        self,
        model_name: str = "EquiformerV2-31M-S2EF",
        model_path: Optional[str] = None,
        device: str = "auto",
        dtype: str = "float32",
    ):
        self.model_name = model_name
        self.device = get_device(device)
        self.dtype = dtype
        
        # Load EquiformerV2 model
        self._load_model(model_path)
    
    def _load_model(self, model_path: Optional[str] = None):
        """Load EquiformerV2 model from Open Catalyst Project."""
        try:
            from ocpmodels.common.relaxation.ase_utils import OCPCalculator
            
            if model_path:
                # Load custom checkpoint
                self.calculator = OCPCalculator(
                    checkpoint_path=model_path,
                    cpu=(self.device == "cpu")
                )
            else:
                # Map model names to checkpoint paths/URLs
                model_map = {
                    "EquiformerV2-31M-S2EF": "equiformer_v2_31M_s2ef_all_md",
                    "EquiformerV2-153M-S2EF": "equiformer_v2_153M_s2ef_all_md",
                    "equiformer_v2_31M": "equiformer_v2_31M_s2ef_all_md",
                    "equiformer_v2_153M": "equiformer_v2_153M_s2ef_all_md",
                }
                
                checkpoint = model_map.get(self.model_name, self.model_name)
                
                self.calculator = OCPCalculator(
                    checkpoint_path=checkpoint,
                    cpu=(self.device == "cpu")
                )
                
        except ImportError:
            raise ImportError(
                "Open Catalyst Project (ocp) not installed. Install with:\n"
                "  pip install torch\n"
                "  pip install git+https://github.com/Open-Catalyst-Project/ocp.git"
            )
    
    # ==================== Task 1: Single-Point Calculation ====================
    
    def single_point(
        self,
        structure: Union[str, Path, Atoms],
        properties: List[str] = ["energy", "forces", "stress"]
    ) -> Dict[str, Any]:
        """
        Single-point energy and force calculation.
        
        Args:
            structure: Structure file path or ASE Atoms object
            properties: Properties to calculate
        
        Returns:
            dict: Calculation results with energy, forces, stress, etc.
        
        Example:
            >>> result = calc.single_point("structure.cif")
            >>> print(f"Energy: {result['energy']:.4f} eV")
        """
        if isinstance(structure, (str, Path)):
            atoms = read_structure(structure)
        else:
            atoms = structure
        
        return calculate_single_point(atoms, self.calculator)
    
    # ==================== Task 2: Structure Optimization ====================
    
    def optimize(
        self,
        structure: Union[str, Path, Atoms],
        fmax: float = 0.05,
        max_steps: int = 500,
        optimize_cell: bool = False,
        optimizer: str = "LBFGS",
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Structure optimization.
        
        Args:
            structure: Input structure
            fmax: Force convergence threshold (eV/Å)
            max_steps: Maximum optimization steps
            optimize_cell: Whether to optimize cell parameters
            optimizer: Optimizer type ("LBFGS", "BFGS", "FIRE")
            output_file: Output structure file path
        
        Returns:
            dict: Optimization results
        
        Example:
            >>> result = calc.optimize("MOF.cif", fmax=0.01, optimize_cell=True)
            >>> print(f"Converged: {result['converged']}")
        """
        from ase.optimize import LBFGS, BFGS, FIRE
        from ase.constraints import ExpCellFilter
        
        if isinstance(structure, (str, Path)):
            atoms = read_structure(structure)
        else:
            atoms = structure.copy()
        
        atoms.calc = self.calculator
        
        # Select optimizer
        opt_class = {"LBFGS": LBFGS, "BFGS": BFGS, "FIRE": FIRE}[optimizer]
        
        if optimize_cell:
            atoms_opt = ExpCellFilter(atoms)
        else:
            atoms_opt = atoms
        
        opt = opt_class(atoms_opt, logfile="-")
        opt.run(fmax=fmax, steps=max_steps)
        
        converged = opt.converged()
        final_energy = atoms.get_potential_energy()
        
        if output_file:
            write_structure(atoms, output_file)
        
        return {
            "converged": converged,
            "steps": opt.get_number_of_steps(),
            "final_energy": final_energy,
            "atoms": atoms,
        }
    
    # ==================== Task 3: Molecular Dynamics ====================
    
    def run_md(
        self,
        structure: Union[str, Path, Atoms],
        ensemble: str = "nvt",
        temperature: float = 300.0,
        temperature_K: Optional[float] = None,
        pressure: Optional[float] = None,
        pressure_GPa: Optional[float] = None,
        timestep: float = 1.0,
        steps: int = 10000,
        trajectory_file: Optional[str] = None,
        trajectory: Optional[str] = None,
        logfile: Optional[str] = None,
        log_interval: int = 100
    ) -> Atoms:
        """
        Molecular dynamics simulation.
        
        Args:
            structure: Input structure
            ensemble: Ensemble type ("nve", "nvt", "npt")
            temperature: Temperature (K) - alias for temperature_K
            temperature_K: Temperature (K)
            pressure: Pressure (GPa) - alias for pressure_GPa
            pressure_GPa: Pressure (GPa, for NPT)
            timestep: Time step (fs)
            steps: Number of MD steps
            trajectory_file: Trajectory output file - alias for trajectory
            trajectory: Trajectory output file
            logfile: MD log file
            log_interval: Logging interval
        
        Returns:
            Atoms: Final structure after MD
        
        Example:
            >>> final = calc.run_md(
            ...     "MOF.cif",
            ...     ensemble="nvt",
            ...     temperature_K=300,
            ...     steps=50000
            ... )
        """
        if isinstance(structure, (str, Path)):
            atoms = read_structure(structure)
        else:
            atoms = structure.copy()
        
        atoms.calc = self.calculator
        
        # Handle parameter aliases
        temp_final = temperature_K if temperature_K is not None else temperature
        press_final = pressure_GPa if pressure_GPa is not None else pressure
        traj_final = trajectory if trajectory is not None else trajectory_file
        
        return run_md(
            atoms=atoms,
            ensemble=ensemble,
            temperature=temp_final,
            pressure=press_final,
            timestep=timestep,
            steps=steps,
            trajectory_file=traj_final,
            logfile=logfile,
            log_interval=log_interval
        )
    
    # ==================== Task 4: Phonon Calculation ====================
    
    def calculate_phonon(
        self,
        structure: Union[str, Path, Atoms],
        supercell: List[int] = [2, 2, 2],
        mesh: List[int] = [20, 20, 20],
        temperature_range: tuple = (0, 500, 50)
    ) -> Dict[str, Any]:
        """
        Phonon and thermodynamic property calculation.
        
        Args:
            structure: Primitive cell structure
            supercell: Supercell size [nx, ny, nz]
            mesh: q-point mesh [mx, my, mz]
            temperature_range: (T_min, T_max, T_step) in Kelvin
        
        Returns:
            dict: Phonon properties (DOS, free energy, heat capacity, etc.)
        
        Example:
            >>> result = calc.calculate_phonon("MOF.cif", supercell=[2, 2, 2])
            >>> print(f"Zero-point Energy: {result['ZPE']:.4f} eV")
        """
        if isinstance(structure, (str, Path)):
            atoms = read_structure(structure)
        else:
            atoms = structure
        
        return calculate_phonon(
            atoms=atoms,
            calculator=self.calculator,
            supercell=supercell,
            mesh=mesh,
            temperature_range=temperature_range
        )
    
    # ==================== Task 5: Mechanical Properties ====================
    
    def calculate_bulk_modulus(
        self,
        structure: Union[str, Path, Atoms],
        strain_range: float = 0.05,
        npoints: int = 11
    ) -> Dict[str, float]:
        """
        Bulk modulus calculation using equation of state.
        
        Args:
            structure: Input structure
            strain_range: Strain range (e.g., 0.05 = ±5%)
            npoints: Number of strain points
        
        Returns:
            dict: Bulk modulus and fit parameters
        
        Example:
            >>> result = calc.calculate_bulk_modulus("MOF.cif")
            >>> print(f"Bulk Modulus: {result['bulk_modulus']:.2f} GPa")
        """
        if isinstance(structure, (str, Path)):
            atoms = read_structure(structure)
        else:
            atoms = structure
        
        return calculate_bulk_modulus(
            atoms=atoms,
            calculator=self.calculator,
            strain_range=strain_range,
            npoints=npoints
        )
    
    # ==================== Utility Methods ====================
    
    def get_calculator(self) -> Calculator:
        """
        Return ASE calculator object for direct use.
        
        Returns:
            Calculator: The underlying EquiformerV2/OCP calculator
            
        Example:
            >>> atoms.calc = calc.get_calculator()
            >>> energy = atoms.get_potential_energy()
        """
        return self.calculator
