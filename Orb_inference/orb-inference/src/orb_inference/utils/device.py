"""Device management utilities for Orb models."""

import torch
from typing import Literal


def get_device(device: Literal["auto", "cpu", "cuda"] = "auto") -> str:
    """
    Get the compute device for Orb calculations.
    
    Args:
        device: Device specification
            - "auto": Automatically detect (prefer CUDA if available)
            - "cpu": Force CPU
            - "cuda" or "cuda:0": Use GPU
            
    Returns:
        Device string ("cpu" or "cuda" or "cuda:N")
        
    Examples:
        >>> device = get_device("auto")
        >>> print(device)
        'cuda'  # if GPU available, else 'cpu'
        
        >>> device = get_device("cpu")
        >>> print(device)
        'cpu'
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cpu":
        return "cpu"
    elif device.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return device
    else:
        raise ValueError(
            f"Invalid device: {device}. Must be 'auto', 'cpu', or 'cuda'."
        )


def validate_device(device: str) -> bool:
    """
    Validate that the specified device is available.
    
    Args:
        device: Device string to validate
        
    Returns:
        True if device is valid and available
        
    Raises:
        RuntimeError: If device is not available
        
    Examples:
        >>> validate_device("cpu")
        True
        
        >>> validate_device("cuda")
        True  # if GPU available, else raises RuntimeError
    """
    if device == "cpu":
        return True
    elif device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Device '{device}' requested but CUDA is not available. "
                f"Please check your PyTorch and CUDA installation."
            )
        # Check specific GPU index if provided
        if ":" in device:
            gpu_id = int(device.split(":")[1])
            if gpu_id >= torch.cuda.device_count():
                raise RuntimeError(
                    f"GPU {gpu_id} requested but only {torch.cuda.device_count()} "
                    f"GPU(s) available."
                )
        return True
    else:
        raise ValueError(f"Invalid device: {device}")


def get_device_info() -> dict:
    """
    Get detailed information about available compute devices.
    
    Returns:
        Dictionary with device information:
            - pytorch_version: PyTorch version
            - cuda_available: Whether CUDA is available
            - cuda_version: CUDA version (if available)
            - device_count: Number of GPUs (if CUDA available)
            - devices: List of device names
            
    Examples:
        >>> info = get_device_info()
        >>> print(info['pytorch_version'])
        '2.3.1'
        >>> print(info['cuda_available'])
        True
        >>> print(info['devices'])
        ['NVIDIA GeForce RTX 3090']
    """
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["device_count"] = torch.cuda.device_count()
        info["devices"] = [
            torch.cuda.get_device_name(i) 
            for i in range(torch.cuda.device_count())
        ]
        info["current_device"] = torch.cuda.current_device()
    else:
        info["cuda_version"] = None
        info["device_count"] = 0
        info["devices"] = []
        info["current_device"] = None
    
    return info


def print_device_info():
    """
    Print detailed device information in a user-friendly format.
    
    Examples:
        >>> print_device_info()
        PyTorch Version: 2.3.1
        CUDA Available: True
        CUDA Version: 12.1
        GPU Count: 1
        GPU 0: NVIDIA GeForce RTX 3090
    """
    info = get_device_info()
    
    print("=" * 60)
    print("Device Information")
    print("=" * 60)
    print(f"PyTorch Version: {info['pytorch_version']}")
    print(f"CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"GPU Count: {info['device_count']}")
        for i, device_name in enumerate(info['devices']):
            print(f"GPU {i}: {device_name}")
        print(f"Current Device: {info['current_device']}")
    else:
        print("(No CUDA devices available - CPU mode only)")
    
    print("=" * 60)
