"""Device management utilities for MACE inference"""

import torch
from typing import Literal, Union

DeviceType = Literal["auto", "cpu", "cuda"]


def get_device(device: DeviceType = "auto") -> str:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cpu", or "cuda")
        
    Returns:
        Device string ("cpu" or "cuda")
        
    Raises:
        ValueError: If CUDA is requested but not available
        
    Examples:
        >>> device = get_device("auto")  # Auto-detect
        >>> device = get_device("cuda")  # Force CUDA
        >>> device = get_device("cpu")   # Force CPU
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA requested but not available. "
                "Install GPU version of PyTorch or use device='auto' or device='cpu'"
            )
        return "cuda"
    elif device == "cpu":
        return "cpu"
    else:
        raise ValueError(f"Invalid device: {device}. Must be 'auto', 'cpu', or 'cuda'")


def validate_device(device: str) -> None:
    """
    Validate that the specified device is available.
    
    Args:
        device: Device string to validate
        
    Raises:
        ValueError: If device is invalid or unavailable
    """
    if device not in ["cpu", "cuda"]:
        raise ValueError(f"Invalid device: {device}")
    
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but not available")


def get_device_info() -> dict:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "pytorch_version": torch.__version__,
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory": f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
            }
            for i in range(torch.cuda.device_count())
        ]
    
    return info
