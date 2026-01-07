"""
Device management module.

Provides CUDA/CPU device detection and management functionality.
"""

from typing import List
import warnings


def get_device(device: str = "auto") -> str:
    """
    Get computing device.
    
    Args:
        device: Device selection
            - "auto": Auto-detect (GPU priority)
            - "cuda" / "cuda:0" / "cuda:1": Use specified GPU
            - "cpu": Use CPU
    
    Returns:
        str: The actual device string to use
    
    Example:
        >>> device = get_device("auto")
        >>> print(device)  # "cuda" or "cpu"
    """
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    
    if device.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                warnings.warn(
                    f"CUDA is not available. Falling back to CPU.",
                    RuntimeWarning
                )
                return "cpu"
            
            # Check if the specified GPU exists
            if ":" in device:
                gpu_id = int(device.split(":")[1])
                if gpu_id >= torch.cuda.device_count():
                    warnings.warn(
                        f"GPU {gpu_id} not available. Using GPU 0.",
                        RuntimeWarning
                    )
                    return "cuda:0"
            return device
        except ImportError:
            warnings.warn(
                "PyTorch not installed. Falling back to CPU.",
                RuntimeWarning
            )
            return "cpu"
    
    return device


def get_available_devices() -> List[str]:
    """
    Return list of all available devices.
    
    Returns:
        List[str]: List of available devices, e.g., ["cpu", "cuda:0", "cuda:1"]
    """
    devices = ["cpu"]
    
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
    except ImportError:
        pass
    
    return devices


def print_device_info() -> None:
    """Print device information."""
    print("=" * 50)
    print("Device Information")
    print("=" * 50)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {props.name}")
                print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
                print(f"  Compute capability: {props.major}.{props.minor}")
    except ImportError:
        print("PyTorch not installed")
    
    print("=" * 50)


def check_gpu_memory(device: str = "cuda:0") -> dict:
    """
    Check GPU memory usage.
    
    Args:
        device: GPU device
    
    Returns:
        dict: Contains total, used, free (GB)
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        gpu_id = 0
        if ":" in device:
            gpu_id = int(device.split(":")[1])
        
        props = torch.cuda.get_device_properties(gpu_id)
        total = props.total_memory / 1e9
        
        torch.cuda.set_device(gpu_id)
        allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
        reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
        
        return {
            "total": total,
            "allocated": allocated,
            "reserved": reserved,
            "free": total - reserved,
        }
    except Exception as e:
        return {"error": str(e)}
