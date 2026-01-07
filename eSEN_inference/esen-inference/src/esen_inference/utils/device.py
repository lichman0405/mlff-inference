"""
Device management utilities for eSEN Inference

This module provides functions for detecting and managing computation devices
(CPU, CUDA, MPS) for PyTorch models.
"""

import torch
from typing import Dict, Any, Optional, Union
import warnings


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available computation devices.
    
    Returns:
        Dict containing device information:
        - device_type: 'cuda', 'mps', or 'cpu'
        - device_name: Name of the device
        - cuda_version: CUDA version (if available)
        - gpu_memory_total: Total GPU memory in MB (if GPU available)
        - gpu_memory_free: Free GPU memory in MB (if GPU available)
        - gpu_count: Number of GPUs (if CUDA available)
    
    Example:
        >>> info = get_device_info()
        >>> print(f"Device: {info['device_type']}")
        >>> if info['device_type'] == 'cuda':
        ...     print(f"GPU: {info['device_name']}")
        ...     print(f"Memory: {info['gpu_memory_free']}/{info['gpu_memory_total']} MB")
    """
    info = {}
    
    if torch.cuda.is_available():
        info['device_type'] = 'cuda'
        info['device_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        
        # Get memory info for GPU 0
        try:
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**2
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**2
            
            info['gpu_memory_total'] = int(mem_total)
            info['gpu_memory_reserved'] = int(mem_reserved)
            info['gpu_memory_allocated'] = int(mem_allocated)
            info['gpu_memory_free'] = int(mem_total - mem_reserved)
        except Exception as e:
            warnings.warn(f"Could not retrieve GPU memory info: {e}")
            info['gpu_memory_total'] = None
            info['gpu_memory_free'] = None
    
    elif torch.backends.mps.is_available():
        info['device_type'] = 'mps'
        info['device_name'] = 'Apple Metal Performance Shaders (MPS)'
        # MPS doesn't provide memory info via PyTorch
        info['gpu_memory_total'] = None
        info['gpu_memory_free'] = None
    
    else:
        info['device_type'] = 'cpu'
        info['device_name'] = 'CPU'
        import multiprocessing
        info['cpu_count'] = multiprocessing.cpu_count()
    
    return info


def select_device(
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = True
) -> torch.device:
    """
    Select and validate computation device.
    
    Args:
        device: Device string ('cuda', 'cpu', 'cuda:0', 'mps') or torch.device.
                If None, automatically selects best available device.
        verbose: Whether to print device information.
    
    Returns:
        torch.device object
    
    Raises:
        RuntimeError: If specified device is not available.
    
    Example:
        >>> device = select_device('cuda')  # Use GPU
        >>> device = select_device('cpu')   # Force CPU
        >>> device = select_device()        # Auto-select
    """
    if device is None:
        # Auto-select best device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Validate device
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but CUDA is not available. "
                "Please install PyTorch with CUDA support or use device='cpu'"
            )
        if device.index is not None:
            if device.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device {device.index} requested but only "
                    f"{torch.cuda.device_count()} CUDA devices available"
                )
    
    elif device.type == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS device requested but MPS is not available. "
                "MPS is only available on macOS with Apple Silicon (M1/M2/M3)."
            )
    
    if verbose:
        info = get_device_info()
        if info['device_type'] == 'cuda':
            print(f"Using device: {device}")
            print(f"  GPU: {info['device_name']}")
            print(f"  CUDA version: {info['cuda_version']}")
            print(f"  Memory: {info['gpu_memory_free']}/{info['gpu_memory_total']} MB free")
        elif info['device_type'] == 'mps':
            print(f"Using device: {device}")
            print(f"  {info['device_name']}")
        else:
            print(f"Using device: {device}")
            print(f"  CPU cores: {info.get('cpu_count', 'unknown')}")
    
    return device


def empty_cache():
    """
    Empty CUDA cache to free up GPU memory.
    
    This is useful between inference runs to prevent memory buildup.
    Safe to call even if CUDA is not available.
    
    Example:
        >>> from esen_inference.utils.device import empty_cache
        >>> for mof_file in mof_files:
        ...     result = esen.optimize(read(mof_file))
        ...     empty_cache()  # Clean up after each MOF
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_num_threads(n: int):
    """
    Set number of threads for CPU operations.
    
    Args:
        n: Number of threads to use
    
    Example:
        >>> from esen_inference.utils.device import set_num_threads
        >>> set_num_threads(16)  # Use 16 CPU threads
    """
    torch.set_num_threads(n)
    try:
        import mkl
        mkl.set_num_threads(n)
    except ImportError:
        pass


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage (GPU or CPU).
    
    Returns:
        Dict with memory usage in MB:
        - allocated: Currently allocated memory
        - reserved: Reserved memory (CUDA only)
        - free: Free memory (CUDA only)
    
    Example:
        >>> mem = get_memory_usage()
        >>> print(f"Allocated: {mem['allocated']:.2f} MB")
    """
    usage = {}
    
    if torch.cuda.is_available():
        usage['allocated'] = torch.cuda.memory_allocated() / 1024**2
        usage['reserved'] = torch.cuda.memory_reserved() / 1024**2
        usage['free'] = (
            torch.cuda.get_device_properties(0).total_memory / 1024**2 
            - torch.cuda.memory_reserved() / 1024**2
        )
    else:
        # For CPU, use psutil if available
        try:
            import psutil
            process = psutil.Process()
            usage['allocated'] = process.memory_info().rss / 1024**2
            usage['reserved'] = 0
            usage['free'] = psutil.virtual_memory().available / 1024**2
        except ImportError:
            usage['allocated'] = 0
            usage['reserved'] = 0
            usage['free'] = 0
    
    return usage
