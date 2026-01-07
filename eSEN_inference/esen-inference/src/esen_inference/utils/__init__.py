"""
Utility modules for eSEN Inference

This package provides utility functions for device management, I/O operations,
and other helper functions used throughout the eSEN Inference library.

Modules:
- device: GPU/CPU device detection and management
- io: Structure I/O operations (read/write CIF, POSCAR, etc.)
"""

from esen_inference.utils.device import get_device_info, select_device
from esen_inference.utils.io import read_structure, write_structure

__all__ = [
    "get_device_info",
    "select_device",
    "read_structure",
    "write_structure",
]
