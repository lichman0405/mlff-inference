"""
GRACE Inference - Core Module

Main inference class for GRACE calculations.
"""

from pathlib import Path
from typing import Union, Optional, Dict, Any, List

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from .utils import get_device, read_structure, write_structure
from .tasks import (
    calculate_single_point,
    optimize_structure,
    run_md,
    calculate_phonon,
    calculate_bulk_modulus,
    calculate_adsorption_energy,
)


class GRACEInference:
    """
    GRACE Inference Calculator
    
    A unified interface for GRACE graph attention-based force field calculations.
    
    Args:
        model_name: Model name (e.g., "grace-2l", "grace-3l")
        model_path: Custom model checkpoint path (optional)
        device: Computing device ("auto", "cuda", "cpu", "cuda:0", etc.)
        dtype: Data type ("float32" or "float64")
    
    Example:
        >>> from grace_inference import GRACEInference
        >>> calc = GRACEInference(model_name="grace-2l", device="cuda")
        >>> result = calc.single_point("MOF-5.cif")
        >>> print(result['energy'])
    """
    
    def __init__(
        self,
        model_name: str = "grace-2l",
        model_path: Optional[str] = None,
        device: str = "auto",
        dtype: str = "float32",
    ):
        self.model_name = model_name
        self.device = get_device(device)
        self.dtype = dtype
        
        # Load GRACE model
        self._load_model(model_path)
    
    def _load_model(self, model_path: Optional[str] = None):
        """Load GRACE model."""
        try:
            # Import GRACE calculator
            # Note: Adjust import based on actual GRACE package structure
            from grace.calculator import GRACECalculator
            
            if model_path:
                self.calculator = GRACECalculator(model_path, device=self.device)
            else:
                # Use default model
                self.calculator = GRACECalculator(self.model_name, device=self.device)
                
        except ImportError:
            raise ImportError(
                "GRACE not installed. Install with: pip install grace-calculator"
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
        
        return calculate_single_point(atoms, self.calculator, properties)
    
    # ==================== Task 2: Structure Optimization ====================
    
    def optimize(
        self,
        structure: Union[str, Path, Atoms],
        fmax: float = 0.05,
        steps: int = 500,
        optimizer: str = "LBFGS",
        optimize_cell: bool = False,
        trajectory: Optional[str] = None,
        output: Optional[str] = None
    ) -> Atoms:
        """
        Optimize atomic structure.
        
        Args:
            structure: Structure file path or ASE Atoms object
            fmax: Force convergence criterion (eV/Å)
            steps: Maximum optimization steps
            optimizer: Optimizer name ("LBFGS", "BFGS", "FIRE")
            optimize_cell: Whether to optimize cell parameters
            trajectory: Path to save optimization trajectory
            output: Path to save optimized structure
        
        Returns:
            Optimized Atoms object
        
        Example:
            >>> optimized = calc.optimize("input.cif", fmax=0.01, optimize_cell=True)
            >>> optimized.write("optimized.cif")
        """
        if isinstance(structure, (str, Path)):
            atoms = read_structure(structure)
        else:
            atoms = structure
        
        optimized = optimize_structure(
            atoms, self.calculator, fmax, steps, optimizer, 
            optimize_cell, trajectory
        )
        
        if output:
            write_structure(optimized, output)
        
        return optimized
    
    # ==================== Task 3: Molecular Dynamics ====================
    
    def molecular_dynamics(
        self,
        structure: Union[str, Path, Atoms],
        ensemble: str = "nvt",
        temperature_K: float = 300,
        pressure_GPa: Optional[float] = None,
        timestep: float = 1.0,
        steps: int = 1000,
        trajectory: Optional[str] = None,
        logfile: Optional[str] = None,
        log_interval: int = 100
    ) -> Atoms:
        """
        Run molecular dynamics simulation.
        
        Args:
            structure: Structure file path or ASE Atoms object
            ensemble: MD ensemble ("nve", "nvt", "npt")
            temperature_K: Target temperature (K)
            pressure_GPa: Target pressure for NPT (GPa)
            timestep: Time step (fs)
            steps: Number of MD steps
            trajectory: Trajectory file path
            logfile: Log file path
            log_interval: Logging interval (steps)
        
        Returns:
            Final Atoms object after MD
        
        Example:
            >>> final = calc.molecular_dynamics(
            ...     "init.cif", 
            ...     ensemble="nvt",
            ...     temperature_K=300,
            ...     steps=10000,
            ...     trajectory="md.traj"
            ... )
        """
        if isinstance(structure, (str, Path)):
            atoms = read_structure(structure)
        else:
            atoms = structure
        
        return run_md(
            atoms, self.calculator, ensemble, temperature_K, pressure_GPa,
            timestep, steps, trajectory, logfile, log_interval
        )
    
    # ==================== Task 4: Phonon Calculation ====================
    
    def phonon(
        self,
        structure: Union[str, Path, Atoms],
        supercell: tuple = (2, 2, 2),
        mesh: tuple = (20, 20, 20),
        temperature_range: Optional[tuple] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate phonon properties.
        
        Args:
            structure: Structure file path or ASE Atoms object
            supercell: Supercell size (nx, ny, nz)
            mesh: k-point mesh (nx, ny, nz)
            temperature_range: (T_min, T_max, T_step) in K
            output_dir: Directory to save results
        
        Returns:
            Dictionary with phonon results
        
        Example:
            >>> phonon_result = calc.phonon(
            ...     "structure.cif",
            ...     supercell=(3, 3, 3),
            ...     temperature_range=(0, 1000, 10)
            ... )
        """
        if isinstance(structure, (str, Path)):
            atoms = read_structure(structure)
        else:
            atoms = structure
        
        return calculate_phonon(
            atoms, self.calculator, supercell, mesh,
            temperature_range, output_dir
        )
    
    # ==================== Task 5: Bulk Modulus ====================
    
    def bulk_modulus(
        self,
        structure: Union[str, Path, Atoms],
        num_points: int = 11,
        strain_range: float = 0.05,
        eos: str = "birchmurnaghan",
        plot: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate bulk modulus using equation of state fitting.
        
        Args:
            structure: Structure file path or ASE Atoms object
            num_points: Number of volume points
            strain_range: Strain range (±)
            eos: Equation of state ("birchmurnaghan", "murnaghan", "vinet")
            plot: Whether to generate plots
            output_dir: Directory to save results
        
        Returns:
            Dictionary with bulk modulus and EOS parameters
        
        Example:
            >>> result = calc.bulk_modulus("structure.cif", num_points=15)
            >>> print(f"Bulk modulus: {result['bulk_modulus_GPa']:.2f} GPa")
        """
        if isinstance(structure, (str, Path)):
            atoms = read_structure(structure)
        else:
            atoms = structure
        
        return calculate_bulk_modulus(
            atoms, self.calculator, num_points, strain_range,
            eos, plot, output_dir
        )
    
    # ==================== Task 6: Adsorption Energy ====================
    
    def adsorption_energy(
        self,
        host_structure: Union[str, Path, Atoms],
        adsorbate_structure: Union[str, Path, Atoms],
        combined_structure: Union[str, Path, Atoms],
        relax_host: bool = True,
        relax_adsorbate: bool = True,
        relax_combined: bool = True,
        fmax: float = 0.05
    ) -> Dict[str, Any]:
        """
        Calculate adsorption energy.
        
        Args:
            host_structure: Host (e.g., MOF) structure
            adsorbate_structure: Adsorbate (e.g., CO2) structure
            combined_structure: Combined host+adsorbate structure
            relax_host: Whether to relax host structure
            relax_adsorbate: Whether to relax adsorbate structure
            relax_combined: Whether to relax combined structure
            fmax: Force convergence criterion for relaxation
        
        Returns:
            Dictionary with adsorption energy and related properties
        
        Example:
            >>> result = calc.adsorption_energy(
            ...     "MOF-5.cif",
            ...     "CO2.xyz",
            ...     "MOF-5_CO2.cif"
            ... )
            >>> print(f"Adsorption energy: {result['adsorption_energy_eV']:.4f} eV")
        """
        if isinstance(host_structure, (str, Path)):
            host = read_structure(host_structure)
        else:
            host = host_structure
        
        if isinstance(adsorbate_structure, (str, Path)):
            adsorbate = read_structure(adsorbate_structure)
        else:
            adsorbate = adsorbate_structure
        
        if isinstance(combined_structure, (str, Path)):
            combined = read_structure(combined_structure)
        else:
            combined = combined_structure
        
        return calculate_adsorption_energy(
            host, adsorbate, combined, self.calculator,
            relax_host, relax_adsorbate, relax_combined, fmax
        )
    
    # ==================== Utility Methods ====================
    
    def get_calculator(self) -> Calculator:
        """
        Get the underlying ASE calculator.
        
        Returns:
            ASE calculator object
        """
        return self.calculator
    
    def set_calculator(self, calculator: Calculator):
        """
        Set a custom ASE calculator.
        
        Args:
            calculator: ASE calculator object
        """
        self.calculator = calculator
