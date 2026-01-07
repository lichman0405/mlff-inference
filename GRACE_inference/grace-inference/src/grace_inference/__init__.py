"""
GRACE Inference Package

A unified inference package for GRACE (Graph Attention-based Convolution for Energy) force field.
Provides tools for structure optimization, molecular dynamics, and property calculations.
"""

__version__ = "0.1.0"
__author__ = "GRACE Inference Contributors"

from .core import GRACEInference

__all__ = ["GRACEInference"]
