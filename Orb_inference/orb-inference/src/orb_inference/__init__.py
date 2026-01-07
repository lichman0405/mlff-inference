"""
Orb Inference Library
=====================

A high-level Python library for materials science calculations using Orb's 
GNS-based machine learning force fields.

Orb models use Graph Network Simulator (GNS) architecture that learns equivariance
through data augmentation rather than predefined symmetries, enabling flexible and
accurate predictions for diverse materials including MOFs.

Quick Start
-----------

    >>> from orb_inference import OrbInference
    >>> 
    >>> # Initialize with v3 model (conservative forces)
    >>> orb = OrbInference(model_name='orb-v3-omat', device='cuda')
    >>> 
    >>> # Single-point energy
    >>> result = orb.single_point('structure.cif')
    >>> print(f"Energy: {result['energy']:.4f} eV")
    >>> 
    >>> # Structure optimization
    >>> opt_result = orb.optimize('structure.cif', fmax=0.01)
    >>> optimized = opt_result['atoms']
    >>> 
    >>> # Molecular dynamics
    >>> final = orb.run_md(optimized, temperature=300, steps=5000, ensemble='nvt')

Available Models
----------------

- **orb-v3-omat**: v3 model trained on OMAT24, conservative forces (recommended)
- **orb-v3-mpa**: v3 model on Materials Project + Alexandria, conservative  
- **orb-d3-v2**: v2 model with D3 dispersion correction
- **orb-mptraj-only-v2**: v2 model on Materials Project trajectories only

Key Differences v3 vs v2:
- v3 uses conservative forces (energy gradients), v2 does not
- v3 has no neighbor limit, v2 limited to 30 neighbors
- v3 recommended for production use

Module Organization
-------------------

- **core**: OrbInference main class
- **utils**: Device management, I/O utilities
- **tasks**: Task-specific functions (static, dynamics, phonon, mechanics, adsorption)

Examples
--------

**Phonon calculation:**

    >>> result = orb.phonon(atoms, supercell_matrix=[2, 2, 2])
    >>> Cv = result['thermal']['heat_capacity']

**Bulk modulus:**

    >>> result = orb.bulk_modulus(atoms, strain_range=0.05)
    >>> B = result['bulk_modulus']
    >>> print(f"Bulk modulus: {B:.2f} GPa")

**Adsorption energy:**

    >>> result = orb.adsorption_energy(
    ...     host='mof.cif',
    ...     guest='co2.xyz', 
    ...     complex_atoms='mof_co2.cif'
    ... )
    >>> E_ads = result['E_ads']

See Also
--------

- Documentation: ../Orb_inference_tasks.md
- API Reference: ../Orb_inference_API_reference.md
- Installation: ../INSTALL.md
"""

__version__ = "0.1.0"
__author__ = "MLFF-inference Project"

from .core import OrbInference
from .utils.device import get_device, validate_device, get_device_info
from .utils.io import load_structure, save_structure, parse_structure_input

# Task functions for direct use
from .tasks.static import single_point_energy, optimize_structure
from .tasks.dynamics import run_nvt_md, run_npt_md, analyze_md_trajectory
from .tasks.phonon import calculate_phonon, calculate_thermal_properties
from .tasks.mechanics import calculate_bulk_modulus, plot_eos
from .tasks.adsorption import (
    calculate_adsorption_energy, 
    analyze_coordination,
    find_adsorption_sites
)

__all__ = [
    # Core class
    'OrbInference',
    
    # Utils
    'get_device',
    'validate_device', 
    'get_device_info',
    'load_structure',
    'save_structure',
    'parse_structure_input',
    
    # Static tasks
    'single_point_energy',
    'optimize_structure',
    
    # Dynamics tasks
    'run_nvt_md',
    'run_npt_md',
    'analyze_md_trajectory',
    
    # Phonon tasks
    'calculate_phonon',
    'calculate_thermal_properties',
    
    # Mechanics tasks
    'calculate_bulk_modulus',
    'plot_eos',
    
    # Adsorption tasks
    'calculate_adsorption_energy',
    'analyze_coordination',
    'find_adsorption_sites',
]
