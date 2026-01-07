"""
MatterSim Inference - Materials Property Inference Package

Materials property calculation tools based on MatterSim model.

Example:
    >>> from mattersim_inference import MatterSimInference
    >>> calc = MatterSimInference(model_name="MatterSim-v1-5M")
    >>> result = calc.single_point("structure.cif")
"""

__version__ = "0.1.0"
__author__ = "Materials ML Team"

from .core import MatterSimInference

__all__ = ["MatterSimInference", "__version__"]
