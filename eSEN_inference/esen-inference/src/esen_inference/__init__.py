"""
eSEN Inference - Python Inference Library for eSEN Models

eSEN (Smooth & Expressive Equivariant Networks) is the #1 ranked model
in the MOFSimBench benchmark. This library provides a production-ready
Python interface for running inference with eSEN models.

Key Features:
- 8 inference tasks (energy, optimization, MD, phonon, mechanics, adsorption, etc.)
- 2 pre-trained models (esen-30m-oam, esen-30m-mp)
- GPU/CPU support with automatic device management
- CLI tools for batch processing
- Comprehensive API documentation

Quick Start:
    >>> from esen_inference import ESENInference
    >>> from ase.io import read
    >>>
    >>> # Initialize eSEN model
    >>> esen = ESENInference(model_name='esen-30m-oam', device='cuda')
    >>>
    >>> # Load MOF structure
    >>> atoms = read('MOF-5.cif')
    >>>
    >>> # Single-point calculation
    >>> result = esen.single_point(atoms)
    >>> print(f"Energy: {result['energy']:.6f} eV")
    >>>
    >>> # Structure optimization
    >>> opt_result = esen.optimize(atoms, fmax=0.01, relax_cell=True)

For more details, see:
- eSEN_inference_tasks.md: Task-specific documentation
- eSEN_inference_API_reference.md: Complete API reference
- eSEN_inference_INSTALL.md: Installation guide

MOFSimBench Performance:
- Overall Ranking: #1 🥇
- Energy MAE: 0.041 eV/atom (#1)
- Bulk Modulus MAE: 2.64 GPa (#1)
- Optimization Success: 89% (#1)
- MD Stability: Excellent (#1)
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"

from esen_inference.core import ESENInference

__all__ = [
    "ESENInference",
    "__version__",
]
