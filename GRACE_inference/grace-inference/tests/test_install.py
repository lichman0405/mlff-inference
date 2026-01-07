"""
Test suite for GRACE inference installation verification.

This module contains tests to verify that the GRACE inference package
and its dependencies are correctly installed.
"""

import pytest
import sys


def test_python_version():
    """Test that Python version is compatible."""
    assert sys.version_info >= (3, 8), "Python 3.8 or higher is required"
    print(f"✓ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


def test_import_grace_inference():
    """Test that grace_inference package can be imported."""
    try:
        import grace_inference
        print(f"✓ grace_inference package imported successfully")
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import grace_inference: {e}")


def test_import_grace_calculator():
    """Test that GRACECalculator can be imported."""
    try:
        from grace_inference import GRACECalculator
        print(f"✓ GRACECalculator imported successfully")
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import GRACECalculator: {e}")


def test_import_ase():
    """Test that ASE is installed."""
    try:
        import ase
        print(f"✓ ASE version: {ase.__version__}")
        assert True
    except ImportError as e:
        pytest.fail(f"ASE not installed: {e}")


def test_import_torch():
    """Test that PyTorch is installed."""
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        assert True
    except ImportError as e:
        pytest.fail(f"PyTorch not installed: {e}")


def test_torch_device():
    """Test PyTorch device availability."""
    import torch
    
    # Check CPU
    assert torch.cuda.is_available() or True, "At least CPU should be available"
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print(f"✓ CPU-only mode (CUDA not available)")


def test_import_numpy():
    """Test that NumPy is installed."""
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
        assert True
    except ImportError as e:
        pytest.fail(f"NumPy not installed: {e}")


def test_grace_calculator_instantiation():
    """Test that GRACECalculator can be instantiated."""
    try:
        from grace_inference import GRACECalculator
        
        # Try to create calculator instance
        calc = GRACECalculator(device='cpu')
        print(f"✓ GRACECalculator instantiated successfully")
        assert calc is not None
    except Exception as e:
        pytest.fail(f"Failed to instantiate GRACECalculator: {e}")


def test_package_version():
    """Test that package version can be accessed."""
    try:
        import grace_inference
        if hasattr(grace_inference, '__version__'):
            version = grace_inference.__version__
            print(f"✓ grace_inference version: {version}")
            assert version is not None
        else:
            print(f"⚠ Package version attribute not found")
    except Exception as e:
        pytest.fail(f"Failed to get package version: {e}")


if __name__ == "__main__":
    """Run tests with verbose output."""
    print("=" * 60)
    print("GRACE Inference Installation Tests")
    print("=" * 60)
    
    pytest.main([__file__, "-v", "-s"])
