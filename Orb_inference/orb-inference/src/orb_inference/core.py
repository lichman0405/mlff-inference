"""Core OrbInference class providing unified interface."""

from typing import Optional, Union, List, Dict, Any
from pathlib import Path
import torch
from ase import Atoms

from .utils.device import get_device, get_device_info
from .utils.io import load_structure, save_structure, parse_structure_input
from .tasks.static import single_point_energy, optimize_structure
from .tasks.dynamics import run_nvt_md, run_npt_md
from .tasks.phonon import calculate_phonon, calculate_thermal_properties
from .tasks.mechanics import calculate_bulk_modulus
from .tasks.adsorption import calculate_adsorption_energy, analyze_coordination


class OrbInference:
    """
    Unified interface for Orb model inference.
    
    This class provides high-level methods for common materials science tasks
    using Orb's GNS-based machine learning force fields.
    
    Args:
        model_name: Orb model identifier:
                   - 'orb-v3-omat' (OMAT24 dataset, conservative forces)
                   - 'orb-v3-mpa' (Materials Project + Alexandria, conservative)
                   - 'orb-d3-v2' (with D3 dispersion, older version)
                   - 'orb-mptraj-only-v2' (Materials Project only, v2)
        device: Computation device ('cuda', 'cpu', or 'mps')
        precision: Model precision ('float32-high', 'float32-highest', 'float64')
        
    Attributes:
        calculator: ORBCalculator instance
        model_name: Name of loaded model
        device: Device being used
        
    Examples:
        >>> orb = OrbInference(model_name='orb-v3-omat', device='cuda')
        >>> result = orb.single_point(atoms)
        >>> energy = result['energy']
        
        >>> # Structure optimization
        >>> opt_result = orb.optimize(atoms, fmax=0.01)
        >>> optimized_atoms = opt_result['atoms']
        
        >>> # Molecular dynamics
        >>> traj = orb.run_md(atoms, temperature=300, steps=1000)
    """
    
    def __init__(
        self,
        model_name: str = "orb-v3-omat",
        device: Optional[str] = None,
        precision: str = "float32-high"
    ):
        """Initialize OrbInference with specified model."""
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        
        # Validate model name
        valid_models = [
            'orb-v3-omat',
            'orb-v3-mpa', 
            'orb-d3-v2',
            'orb-mptraj-only-v2'
        ]
        if model_name not in valid_models:
            raise ValueError(
                f"Invalid model_name '{model_name}'. "
                f"Choose from: {valid_models}"
            )
        
        # Setup device
        if device is None:
            device = get_device()
        else:
            device = torch.device(device)
        
        self.model_name = model_name
        self.device = device
        self.precision = precision
        
        print(f"Loading Orb model: {model_name}")
        print(f"  Device: {device}")
        print(f"  Precision: {precision}")
        
        # Load pretrained model based on name
        if model_name == 'orb-v3-omat':
            orbff = pretrained.orb_v3_conservative_inf_omat(device=device)
        elif model_name == 'orb-v3-mpa':
            orbff = pretrained.orb_v3_conservative_inf_mpa(device=device)
        elif model_name == 'orb-d3-v2':
            orbff = pretrained.orb_d3_v2(device=device)
        elif model_name == 'orb-mptraj-only-v2':
            orbff = pretrained.orb_mptraj_only_v2(device=device)
        
        # Create calculator with specified precision
        self.calculator = ORBCalculator(orbff, device=device, precision=precision)
        
        device_info = get_device_info(device)
        print(f"Model loaded successfully!")
        if device.type == 'cuda':
            print(f"  GPU: {device_info['name']}")
            print(f"  Memory: {device_info['memory_allocated']:.2f} / "
                  f"{device_info['memory_total']:.2f} GB")
    
    def single_point(
        self,
        atoms: Union[Atoms, str, Path],
        properties: List[str] = None
    ) -> Dict[str, Any]:
        """
        Single-point energy calculation.
        
        Args:
            atoms: Structure (Atoms object or file path)
            properties: Properties to calculate (default: energy, forces, stress)
            
        Returns:
            Dictionary with calculated properties
            
        Examples:
            >>> result = orb.single_point('structure.cif')
            >>> E = result['energy']
            >>> F = result['forces']
        """
        atoms = parse_structure_input(atoms)
        return single_point_energy(atoms, self.calculator, properties=properties)
    
    def optimize(
        self,
        atoms: Union[Atoms, str, Path],
        fmax: float = 0.05,
        optimizer: str = "LBFGS",
        relax_cell: bool = False,
        output: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize structure.
        
        Args:
            atoms: Structure to optimize
            fmax: Force convergence criterion (eV/Å)
            optimizer: Optimizer type ('LBFGS', 'BFGS', 'FIRE')
            relax_cell: Whether to relax cell vectors
            output: Output trajectory path
            
        Returns:
            Dictionary with optimized structure and convergence info
            
        Examples:
            >>> result = orb.optimize('structure.cif', fmax=0.01)
            >>> opt_atoms = result['atoms']
            >>> save_structure(opt_atoms, 'optimized.cif')
        """
        atoms = parse_structure_input(atoms)
        result = optimize_structure(
            atoms, self.calculator,
            fmax=fmax,
            optimizer=optimizer,
            relax_cell=relax_cell,
            trajectory=output
        )
        return result
    
    def run_md(
        self,
        atoms: Union[Atoms, str, Path],
        temperature: float = 300.0,
        steps: int = 1000,
        timestep: float = 1.0,
        ensemble: str = "nvt",
        pressure: Optional[float] = None,
        trajectory: Optional[str] = None,
        logfile: Optional[str] = None
    ) -> Atoms:
        """
        Run molecular dynamics.
        
        Args:
            atoms: Initial structure
            temperature: Temperature (K)
            steps: Number of MD steps
            timestep: Timestep (fs)
            ensemble: 'nvt' or 'npt'
            pressure: Target pressure for NPT (GPa)
            trajectory: Trajectory output file
            logfile: Log file path
            
        Returns:
            Final structure after MD
            
        Examples:
            >>> final = orb.run_md(
            ...     atoms, temperature=300, steps=5000,
            ...     ensemble='nvt', trajectory='md.traj'
            ... )
        """
        atoms = parse_structure_input(atoms)
        
        if ensemble.lower() == 'nvt':
            final_atoms = run_nvt_md(
                atoms, self.calculator,
                temperature=temperature,
                steps=steps,
                timestep=timestep,
                trajectory=trajectory,
                logfile=logfile
            )
        elif ensemble.lower() == 'npt':
            if pressure is None:
                raise ValueError("Must specify pressure for NPT ensemble")
            final_atoms = run_npt_md(
                atoms, self.calculator,
                temperature=temperature,
                pressure=pressure,
                steps=steps,
                timestep=timestep,
                trajectory=trajectory,
                logfile=logfile
            )
        else:
            raise ValueError(f"Unknown ensemble '{ensemble}'. Use 'nvt' or 'npt'.")
        
        return final_atoms
    
    def phonon(
        self,
        atoms: Union[Atoms, str, Path],
        supercell_matrix: Union[List[int], List[List[int]]] = None,
        mesh: List[int] = None,
        t_min: float = 0,
        t_max: float = 1000,
        t_step: float = 10
    ) -> Dict[str, Any]:
        """
        Calculate phonon and thermal properties.
        
        Args:
            atoms: Primitive cell (should be optimized)
            supercell_matrix: Supercell for force constants
            mesh: k-point mesh for DOS
            t_min: Minimum temperature (K)
            t_max: Maximum temperature (K)
            t_step: Temperature step (K)
            
        Returns:
            Dictionary with phonon results and thermal properties
            
        Examples:
            >>> result = orb.phonon(atoms, supercell_matrix=[2,2,2])
            >>> Cv = result['thermal']['heat_capacity']
            >>> T = result['thermal']['temperatures']
        """
        atoms = parse_structure_input(atoms)
        
        # Calculate phonon
        phonon_result = calculate_phonon(
            atoms, self.calculator,
            supercell_matrix=supercell_matrix,
            mesh=mesh
        )
        
        # Calculate thermal properties
        thermal_result = calculate_thermal_properties(
            phonon_result['phonon'],
            t_min=t_min, t_max=t_max, t_step=t_step
        )
        
        return {
            "phonon": phonon_result['phonon'],
            "frequency_points": phonon_result['frequency_points'],
            "total_dos": phonon_result['total_dos'],
            "thermal": thermal_result,
        }
    
    def bulk_modulus(
        self,
        atoms: Union[Atoms, str, Path],
        strain_range: float = 0.05,
        n_points: int = 7,
        optimize_first: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate bulk modulus from equation of state.
        
        Args:
            atoms: Structure
            strain_range: Volume strain range
            n_points: Number of volume points
            optimize_first: Optimize before EOS calculation
            
        Returns:
            Dictionary with bulk modulus and EOS fit
            
        Examples:
            >>> result = orb.bulk_modulus(atoms)
            >>> B = result['bulk_modulus']
            >>> print(f"Bulk modulus: {B:.2f} GPa")
        """
        atoms = parse_structure_input(atoms)
        return calculate_bulk_modulus(
            atoms, self.calculator,
            strain_range=strain_range,
            n_points=n_points,
            optimize_first=optimize_first
        )
    
    def adsorption_energy(
        self,
        host: Union[Atoms, str, Path],
        guest: Union[Atoms, str, Path],
        complex_atoms: Union[Atoms, str, Path],
        optimize_complex: bool = True
    ) -> Dict[str, float]:
        """
        Calculate adsorption energy.
        
        Args:
            host: MOF or host structure
            guest: Adsorbate molecule
            complex_atoms: Host + guest complex
            optimize_complex: Optimize complex before energy calc
            
        Returns:
            Dictionary with adsorption energy
            
        Examples:
            >>> result = orb.adsorption_energy(
            ...     host='mof.cif',
            ...     guest='co2.xyz',
            ...     complex_atoms='mof_co2.cif'
            ... )
            >>> E_ads = result['E_ads']
        """
        host = parse_structure_input(host)
        guest = parse_structure_input(guest)
        complex_atoms = parse_structure_input(complex_atoms)
        
        return calculate_adsorption_energy(
            host, guest, complex_atoms, self.calculator,
            optimize_complex=optimize_complex
        )
    
    def coordination(
        self,
        atoms: Union[Atoms, str, Path],
        center_indices: Optional[List[int]] = None,
        cutoff_scale: float = 1.3
    ) -> Dict[str, Any]:
        """
        Analyze coordination environment.
        
        Args:
            atoms: Structure to analyze
            center_indices: Indices of center atoms (e.g., metal sites)
            cutoff_scale: Scaling factor for natural cutoffs
            
        Returns:
            Dictionary with coordination numbers and neighbor lists
            
        Examples:
            >>> result = orb.coordination(atoms, center_indices=[0, 1])
            >>> cn = result['coordination_numbers']
        """
        atoms = parse_structure_input(atoms)
        return analyze_coordination(
            atoms,
            center_indices=center_indices,
            cutoff_scale=cutoff_scale
        )
    
    def info(self) -> Dict[str, Any]:
        """
        Get information about loaded model and device.
        
        Returns:
            Dictionary with model and device info
            
        Examples:
            >>> info = orb.info()
            >>> print(info['model_name'])
            >>> print(info['device'])
        """
        device_info = get_device_info(self.device)
        
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "precision": self.precision,
            "device_info": device_info,
        }
