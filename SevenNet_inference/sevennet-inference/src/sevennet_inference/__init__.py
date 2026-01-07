"""
SevenNet Inference Package

A unified inference package for SevenNet equivariant graph neural network force field.
Provides tools for structure optimization, molecular dynamics, and property calculations.
"""

__version__ = "0.1.0"
__author__ = "SevenNet Inference Contributors"

from .core import SevenNetInference

__all__ = ["SevenNetInference"]
