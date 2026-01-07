"""D3 dispersion correction utilities"""

from typing import Optional
import warnings


def create_d3_calculator(device: str = "cpu", damping: str = "bj", xc: str = "pbe"):
    """
    Create a DFT-D3 dispersion correction calculator.
    
    Args:
        device: Compute device ("cpu" or "cuda")
        damping: Damping function ("zero", "bj", "zerom", "bjm")
        xc: Exchange-correlation functional (e.g., "pbe", "b3lyp")
        
    Returns:
        TorchDFTD3Calculator instance
        
    Raises:
        ImportError: If torch-dftd is not installed
        
    Examples:
        >>> d3_calc = create_d3_calculator(device="cuda", damping="bj", xc="pbe")
    """
    try:
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
    except ImportError:
        raise ImportError(
            "torch-dftd is required for D3 dispersion correction. "
            "Install with: pip install torch-dftd or pip install mace-inference[d3]"
        )
    
    return TorchDFTD3Calculator(device=device, damping=damping, xc=xc)


def create_combined_calculator(
    mace_calculator,
    enable_d3: bool = False,
    d3_device: Optional[str] = None,
    d3_damping: str = "bj",
    d3_xc: str = "pbe"
):
    """
    Create a combined MACE + D3 calculator using ASE's SumCalculator.
    
    Args:
        mace_calculator: MACE calculator instance
        enable_d3: Whether to enable D3 correction
        d3_device: Device for D3 calculator (defaults to MACE device)
        d3_damping: D3 damping function
        d3_xc: D3 exchange-correlation functional
        
    Returns:
        Combined calculator (or just MACE calculator if D3 disabled)
        
    Examples:
        >>> from mace.calculators import mace_mp
        >>> mace_calc = mace_mp(model="medium", device="cuda")
        >>> combined = create_combined_calculator(mace_calc, enable_d3=True)
    """
    if not enable_d3:
        return mace_calculator
    
    try:
        from ase.calculators.mixing import SumCalculator
    except ImportError:
        warnings.warn("ASE SumCalculator not available, D3 correction disabled")
        return mace_calculator
    
    # Determine D3 device from MACE calculator if not specified
    if d3_device is None:
        # Try to extract device from MACE calculator
        if hasattr(mace_calculator, 'device'):
            d3_device = mace_calculator.device
        else:
            d3_device = "cpu"
    
    d3_calc = create_d3_calculator(device=d3_device, damping=d3_damping, xc=d3_xc)
    
    return SumCalculator([mace_calculator, d3_calc])


def check_d3_available() -> bool:
    """Check if torch-dftd is installed."""
    try:
        import torch_dftd
        return True
    except ImportError:
        return False
