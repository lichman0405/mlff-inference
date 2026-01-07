"""Device management utilities for EquiformerV2 inference"""

import torch
from typing import Literal, List

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


def get_available_devices() -> List[str]:
    """
    Return list of all available devices.
    
    Returns:
        List of available devices, e.g., ["cpu", "cuda:0", "cuda:1"]
        
    Examples:
        >>> devices = get_available_devices()
        >>> print(devices)  # ["cpu"] or ["cpu", "cuda:0", "cuda:1"]
    """
    devices = ["cpu"]
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
    
    return devices


def print_device_info() -> None:
    """
    Print detailed device information.
    
    Examples:
        >>> print_device_info()
        ==================================================
        Device Information
        ==================================================
        PyTorch version: 2.0.1
        CUDA available: True
        CUDA version: 11.8
        Number of GPUs: 1
        ...
    """
    print("=" * 50)
    print("Device Information")
    print("=" * 50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
    else:
        print("No CUDA devices available")
    
    print("=" * 50)


def check_gpu_memory(device: str = "cuda:0") -> dict:
    """
    Check GPU memory usage.
    
    Args:
        device: GPU device string (e.g., "cuda:0")
        
    Returns:
        Dictionary with memory information (in GB)
        
    Raises:
        ValueError: If device is not a CUDA device or not available
        
    Examples:
        >>> mem_info = check_gpu_memory("cuda:0")
        >>> print(f"Free: {mem_info['free']:.2f} GB")
    """
    if not device.startswith("cuda"):
        raise ValueError(f"Device must be a CUDA device, got: {device}")
    
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    
    # Parse device index
    if ":" in device:
        device_idx = int(device.split(":")[1])
    else:
        device_idx = 0
    
    if device_idx >= torch.cuda.device_count():
        raise ValueError(f"Device {device} not available. Only {torch.cuda.device_count()} GPU(s) found.")
    
    # Get memory info
    total_memory = torch.cuda.get_device_properties(device_idx).total_memory
    allocated_memory = torch.cuda.memory_allocated(device_idx)
    reserved_memory = torch.cuda.memory_reserved(device_idx)
    free_memory = total_memory - allocated_memory
    
    return {
        "total": total_memory / 1e9,
        "allocated": allocated_memory / 1e9,
        "reserved": reserved_memory / 1e9,
        "free": free_memory / 1e9,
        "device": device,
    }


def validate_device(device: str) -> None:
    """
    Validate that the specified device is available.
    
    Args:
        device: Device string to validate
        
    Raises:
        ValueError: If device is invalid or unavailable
        
    Examples:
        >>> validate_device("cuda")  # Raises error if CUDA unavailable
        >>> validate_device("cpu")   # Always succeeds
    """
    if device not in ["cpu", "cuda"] and not device.startswith("cuda:"):
        raise ValueError(f"Invalid device: {device}")
    
    if device == "cuda" or device.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but not available")
        
        if ":" in device:
            device_idx = int(device.split(":")[1])
            if device_idx >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device {device_idx} not available. "
                    f"Only {torch.cuda.device_count()} GPU(s) found."
                )
