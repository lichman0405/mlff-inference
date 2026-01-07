"""
MACE Inference: High-level Python library for MACE machine learning force field inference tasks
"""

__version__ = "0.1.0"
__author__ = "MACE Inference Contributors"

from mace_inference.core import MACEInference
from mace_inference.utils.device import get_device

__all__ = ["MACEInference", "get_device", "__version__"]
