"""Core MACE Inference class"""

from typing import Optional, Union, List, Literal, Dict, Any
from pathlib import Path
import numpy as np
from ase import Atoms

from mace_inference.utils.device import get_device, validate_device
from mace_inference.utils.d3_correction import create_combined_calculator
from mace_inference.utils.io import parse_structure_input, save_structure, create_supercell
from mace_inference.tasks import (
    single_point_energy,
    optimize_structure,
    run_nvt_md,
    run_npt_md,
    calculate_phonon,
    calculate_thermal_properties,
    calculate_bulk_modulus,
    calculate_adsorption_energy,
    analyze_coordination,
)


class MACEInference:
    """
    High-level interface for MACE machine learning force field inference.
    
    This class provides a unified API for common inference tasks including:
    - Single-point energy calculations
    - Structure optimization
    - Molecular dynamics (NVT/NPT)
    - Phonon calculations
    - Mechanical properties
    - Adsorption energies
    
    Args:
        model: MACE model name ("small", "medium", "large") or path to custom model
        device: Compute device ("auto", "cpu", or "cuda")
        enable_d3: Enable DFT-D3 dispersion correction
        d3_damping: D3 damping function ("bj", "zero", "zerom", "bjm")
        d3_xc: D3 exchange-correlation functional (e.g., "pbe", "b3lyp")
        default_dtype: Default data type ("float32" or "float64")
        
    Examples:
        >>> # Basic usage
        >>> calc = MACEInference(model="medium", device="auto")
        >>> result = calc.single_point("structure.cif")
        
        >>> # With D3 correction
        >>> calc = MACEInference(model="medium", enable_d3=True)
        
        >>> # GPU acceleration
        >>> calc = MACEInference(model="large", device="cuda")
    """
    
    def __init__(
        self,
        model: str = "medium",
        device: Literal["auto", "cpu", "cuda"] = "auto",
        enable_d3: bool = False,
        d3_damping: str = "bj",
        d3_xc: str = "pbe",
        default_dtype: str = "float64",
    ):
        # Device setup
        self.device = get_device(device)
        validate_device(self.device)
        
        # Model configuration
        self.model_name = model
        self.enable_d3 = enable_d3
        self.d3_damping = d3_damping
        self.d3_xc = d3_xc
        self.default_dtype = default_dtype
        
        # Initialize calculator
        self.calculator = self._create_calculator()
        
    def _create_calculator(self):
        """Create MACE calculator with optional D3 correction."""
        try:
            from mace.calculators import mace_mp, mace_off, MACECalculator
        except ImportError:
            raise ImportError(
                "mace-torch is required. Install with: pip install mace-torch"
            )
        
        # Create base MACE calculator
        if self.model_name in ["small", "medium", "large"]:
            # Use pre-trained MACE-MP models
            mace_calc = mace_mp(
                model=self.model_name,
                device=self.device,
                default_dtype=self.default_dtype
            )
        elif Path(self.model_name).exists():
            # Load custom model from file
            mace_calc = MACECalculator(
                model_paths=self.model_name,
                device=self.device,
                default_dtype=self.default_dtype
            )
        else:
            raise ValueError(
                f"Invalid model: {self.model_name}. "
                "Use 'small', 'medium', 'large', or path to custom model."
            )
        
        # Add D3 correction if enabled
        return create_combined_calculator(
            mace_calc,
            enable_d3=self.enable_d3,
            d3_device=self.device,
            d3_damping=self.d3_damping,
            d3_xc=self.d3_xc
        )
    
    def single_point(
        self,
        structure: Union[str, Path, Atoms],
        properties: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform single-point energy calculation.
        
        Args:
            structure: Structure file path or Atoms object
            properties: Properties to calculate (default: ["energy", "forces", "stress"])
            
        Returns:
            Dictionary with calculated properties
            
        Examples:
            >>> result = calc.single_point("structure.cif")
            >>> print(f"Energy: {result['energy']:.4f} eV")
            >>> print(f"Max force: {result['forces'].max():.4f} eV/Å")
        """
        atoms = parse_structure_input(structure)
        return single_point_energy(atoms, self.calculator, properties)
    
    def optimize(
        self,
        structure: Union[str, Path, Atoms],
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
            structure: Input structure
            fmax: Force convergence criterion (eV/Å)
            steps: Maximum optimization steps
            optimizer: Optimization algorithm ("LBFGS", "BFGS", "FIRE")
            optimize_cell: Whether to optimize cell parameters
            trajectory: Trajectory file path
            logfile: Optimization log file path
            output: Output structure file path
            
        Returns:
            Optimized Atoms object
            
        Examples:
            >>> optimized = calc.optimize("structure.cif", fmax=0.05)
            >>> optimized = calc.optimize("structure.cif", optimize_cell=True, output="opt.cif")
        """
        atoms = parse_structure_input(structure)
        optimized_atoms = optimize_structure(
            atoms=atoms,
            calculator=self.calculator,
            fmax=fmax,
            steps=steps,
            optimizer=optimizer,
            optimize_cell=optimize_cell,
            trajectory=trajectory,
            logfile=logfile
        )
        
        if output:
            save_structure(optimized_atoms, output)
        
        return optimized_atoms
    
    def run_md(
        self,
        structure: Union[str, Path, Atoms],
        ensemble: Literal["nvt", "npt"] = "nvt",
        temperature_K: float = 300,
        steps: int = 1000,
        timestep: float = 1.0,
        pressure_GPa: Optional[float] = None,
        taut: Optional[float] = None,
        taup: Optional[float] = None,
        trajectory: Optional[str] = None,
        logfile: Optional[str] = None,
        log_interval: int = 100
    ) -> Atoms:
        """
        Run molecular dynamics simulation.
        
        Args:
            structure: Input structure
            ensemble: MD ensemble ("nvt" or "npt")
            temperature_K: Target temperature (K)
            steps: Number of MD steps
            timestep: Time step (fs)
            pressure_GPa: Target pressure for NPT (GPa)
            taut: Temperature coupling time (fs)
            taup: Pressure coupling time (fs)
            trajectory: Trajectory file path
            logfile: MD log file path
            log_interval: Logging interval (steps)
            
        Returns:
            Final Atoms object
            
        Examples:
            >>> # NVT simulation
            >>> final = calc.run_md("structure.cif", ensemble="nvt", temperature_K=300, steps=10000)
            
            >>> # NPT simulation
            >>> final = calc.run_md("structure.cif", ensemble="npt", temperature_K=300, 
            ...                     pressure_GPa=1.0, steps=10000)
        """
        atoms = parse_structure_input(structure)
        
        if ensemble == "nvt":
            return run_nvt_md(
                atoms=atoms,
                calculator=self.calculator,
                temperature_K=temperature_K,
                timestep=timestep,
                steps=steps,
                trajectory=trajectory,
                logfile=logfile,
                log_interval=log_interval,
                taut=taut
            )
        elif ensemble == "npt":
            if pressure_GPa is None:
                pressure_GPa = 0.0  # Default to 0 GPa (1 atm ≈ 0 GPa)
            
            return run_npt_md(
                atoms=atoms,
                calculator=self.calculator,
                temperature_K=temperature_K,
                pressure_GPa=pressure_GPa,
                timestep=timestep,
                steps=steps,
                trajectory=trajectory,
                logfile=logfile,
                log_interval=log_interval,
                taut=taut,
                taup=taup
            )
        else:
            raise ValueError(f"Invalid ensemble: {ensemble}. Use 'nvt' or 'npt'")
    
    def phonon(
        self,
        structure: Union[str, Path, Atoms],
        supercell_matrix: Union[List[int], int] = 2,
        displacement: float = 0.01,
        mesh: List[int] = None,
        temperature_range: Optional[tuple] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate phonon properties.
        
        Args:
            structure: Input structure
            supercell_matrix: Supercell size for phonon calculation
            displacement: Atomic displacement distance (Å)
            mesh: k-point mesh for phonon DOS (default: [20, 20, 20])
            temperature_range: (t_min, t_max, t_step) for thermal properties (K)
            output_dir: Output directory for phonon files
            
        Returns:
            Dictionary with phonon results
            
        Examples:
            >>> result = calc.phonon("structure.cif", supercell_matrix=[2, 2, 2])
            >>> result = calc.phonon("structure.cif", temperature_range=(0, 1000, 10))
        """
        atoms = parse_structure_input(structure)
        
        phonon_result = calculate_phonon(
            atoms=atoms,
            calculator=self.calculator,
            supercell_matrix=supercell_matrix,
            displacement=displacement,
            mesh=mesh,
            output_dir=output_dir
        )
        
        # Calculate thermal properties if temperature range specified
        if temperature_range:
            t_min, t_max, t_step = temperature_range
            thermal = calculate_thermal_properties(
                phonon=phonon_result["phonon"],
                t_min=t_min,
                t_max=t_max,
                t_step=t_step
            )
            phonon_result["thermal_properties"] = thermal
        
        return phonon_result
    
    def bulk_modulus(
        self,
        structure: Union[str, Path, Atoms],
        n_points: int = 11,
        scale_range: tuple = (0.95, 1.05),
        eos_type: str = "birchmurnaghan"
    ) -> Dict[str, float]:
        """
        Calculate bulk modulus using equation of state.
        
        Args:
            structure: Input structure
            n_points: Number of volume points
            scale_range: Volume scaling range (min, max)
            eos_type: Equation of state type
            
        Returns:
            Dictionary with v0, e0, B (GPa)
            
        Examples:
            >>> result = calc.bulk_modulus("structure.cif")
            >>> print(f"Bulk Modulus: {result['B_GPa']:.2f} GPa")
        """
        atoms = parse_structure_input(structure)
        return calculate_bulk_modulus(
            atoms=atoms,
            calculator=self.calculator,
            n_points=n_points,
            scale_range=scale_range,
            eos_type=eos_type
        )
    
    def adsorption_energy(
        self,
        mof_structure: Union[str, Path, Atoms],
        gas_molecule: Union[str, Atoms],
        site_position: List[float],
        optimize_complex: bool = True,
        fmax: float = 0.05
    ) -> Dict[str, Any]:
        """
        Calculate gas adsorption energy.
        
        Args:
            mof_structure: MOF structure
            gas_molecule: Gas molecule name (e.g., "CO2", "H2O") or Atoms object
            site_position: Adsorption site position [x, y, z]
            optimize_complex: Whether to optimize the adsorption complex
            fmax: Force convergence for optimization
            
        Returns:
            Dictionary with adsorption energy and structures
            
        Examples:
            >>> result = calc.adsorption_energy("mof.cif", "CO2", [10.0, 10.0, 10.0])
            >>> print(f"E_ads = {result['E_ads']:.3f} eV")
        """
        mof_atoms = parse_structure_input(mof_structure)
        
        return calculate_adsorption_energy(
            mof_atoms=mof_atoms,
            gas_molecule=gas_molecule,
            site_position=site_position,
            calculator=self.calculator,
            optimize_complex=optimize_complex,
            fmax=fmax
        )
    
    def coordination(
        self,
        structure: Union[str, Path, Atoms],
        metal_indices: Optional[List[int]] = None,
        cutoff_multiplier: float = 1.2
    ) -> Dict[str, Any]:
        """
        Analyze coordination environment.
        
        Args:
            structure: Input structure
            metal_indices: Indices of metal atoms (auto-detect if None)
            cutoff_multiplier: Cutoff radius multiplier for neighbor search
            
        Returns:
            Dictionary with coordination analysis
            
        Examples:
            >>> result = calc.coordination("mof.cif")
            >>> for metal_idx, info in result["coordination"].items():
            ...     print(f"Metal {metal_idx}: CN = {info['coordination_number']}")
        """
        atoms = parse_structure_input(structure)
        
        return analyze_coordination(
            atoms=atoms,
            metal_indices=metal_indices,
            cutoff_multiplier=cutoff_multiplier
        )
    
    def __repr__(self) -> str:
        return (
            f"MACEInference(model='{self.model_name}', device='{self.device}', "
            f"enable_d3={self.enable_d3})"
        )
