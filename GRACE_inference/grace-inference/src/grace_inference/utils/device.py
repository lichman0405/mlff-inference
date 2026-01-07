"""Device management utilities for GRACE inference"""

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
        # Support specific GPU selection like "cuda:0", "cuda:1"
        if device.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise ValueError(
                    f"CUDA requested ({device}) but not available. "
                    "Install GPU version of PyTorch or use device='cpu'"
                )
            gpu_id = int(device.split(":")[1])
            if gpu_id >= torch.cuda.device_count():
                raise ValueError(
                    f"GPU {gpu_id} not available. Available GPUs: 0-{torch.cuda.device_count()-1}"
                )
            return device
        raise ValueError(f"Invalid device: {device}. Must be 'auto', 'cpu', 'cuda', or 'cuda:N'")


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
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")
    else:
        print("No GPU available. Using CPU.")
    
    print("=" * 50)


def check_gpu_memory(device: str = "cuda:0") -> dict:
    """
    Check GPU memory usage.
    
    Args:
        device: GPU device ID (e.g., "cuda:0")
        
    Returns:
        Dictionary with memory information in GB
        
    Raises:
        ValueError: If CUDA is not available or device is invalid
        
    Examples:
        >>> mem_info = check_gpu_memory("cuda:0")
        >>> print(f"Free: {mem_info['free_GB']:.2f} GB")
    """
    if not torch.cuda.is_available():
        raise ValueError("CUDA not available")
    
    if device.startswith("cuda:"):
        gpu_id = int(device.split(":")[1])
    elif device == "cuda":
        gpu_id = 0
    else:
        raise ValueError(f"Invalid GPU device: {device}")
    
    if gpu_id >= torch.cuda.device_count():
        raise ValueError(f"GPU {gpu_id} not available")
    
    # Get memory info
    total = torch.cuda.get_device_properties(gpu_id).total_memory
    allocated = torch.cuda.memory_allocated(gpu_id)
    reserved = torch.cuda.memory_reserved(gpu_id)
    free = total - reserved
    
    return {
        "total_GB": total / 1e9,
        "allocated_GB": allocated / 1e9,
        "reserved_GB": reserved / 1e9,
        "free_GB": free / 1e9,
    }


def validate_device(device: str) -> bool:
    """
    Validate if a device is available.
    
    Args:
        device: Device string to validate
        
    Returns:
        True if device is available, False otherwise
        
    Examples:
        >>> if validate_device("cuda:0"):
        ...     print("GPU available")
    """
    try:
        get_device(device)
        return True
    except ValueError:
        return False
