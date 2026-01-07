"""
Installation test script for orb-inference.

Run this after installation to verify that all dependencies are correctly installed
and the environment is properly configured.
"""

import sys
from pathlib import Path


def test_python_version():
    """Test Python version."""
    print("Checking Python version...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("  ✗ Python 3.8+ required")
        return False
    elif version.minor > 11:
        print("  ⚠ Warning: Python 3.12+ not fully tested")
    
    print("  ✓ Python version OK")
    return True


def test_pytorch():
    """Test PyTorch installation."""
    print("\nChecking PyTorch...")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"  CUDA available: Yes")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  CUDA available: No (CPU only)")
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  MPS available: Yes (Apple Silicon)")
        
        print("  ✓ PyTorch OK")
        return True
        
    except ImportError as e:
        print(f"  ✗ PyTorch import failed: {e}")
        return False


def test_ase():
    """Test ASE installation."""
    print("\nChecking ASE...")
    try:
        import ase
        print(f"  ASE version: {ase.__version__}")
        
        # Check version
        version_parts = ase.__version__.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if major < 3 or (major == 3 and minor < 23):
            print(f"  ⚠ Warning: ASE >= 3.23.0 recommended (FrechetCellFilter)")
        
        print("  ✓ ASE OK")
        return True
        
    except ImportError as e:
        print(f"  ✗ ASE import failed: {e}")
        return False


def test_orb_models():
    """Test orb-models installation."""
    print("\nChecking orb-models...")
    try:
        import orb_models
        print(f"  orb-models installed")
        
        # Try importing key modules
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        
        print("  ✓ orb-models OK")
        return True
        
    except ImportError as e:
        print(f"  ✗ orb-models import failed: {e}")
        print(f"    Install with: pip install orb-models")
        return False


def test_phonopy():
    """Test Phonopy installation."""
    print("\nChecking Phonopy...")
    try:
        import phonopy
        print(f"  Phonopy version: {phonopy.__version__}")
        
        # Check version
        version_parts = phonopy.__version__.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if major < 2 or (major == 2 and minor < 20):
            print(f"  ⚠ Warning: Phonopy >= 2.20.0 recommended")
        
        print("  ✓ Phonopy OK")
        return True
        
    except ImportError as e:
        print(f"  ✗ Phonopy import failed: {e}")
        return False


def test_optional_dependencies():
    """Test optional dependencies."""
    print("\nChecking optional dependencies...")
    
    optional_ok = True
    
    # Matplotlib
    try:
        import matplotlib
        print(f"  matplotlib: {matplotlib.__version__} ✓")
    except ImportError:
        print(f"  matplotlib: Not installed (optional, for plotting)")
        optional_ok = False
    
    # SciPy
    try:
        import scipy
        print(f"  scipy: {scipy.__version__} ✓")
    except ImportError:
        print(f"  scipy: Not installed (optional, for adsorption analysis)")
        optional_ok = False
    
    # Click
    try:
        import click
        print(f"  click: {click.__version__} ✓")
    except ImportError:
        print(f"  click: Not installed (required for CLI)")
        optional_ok = False
    
    return optional_ok


def test_orb_inference():
    """Test orb-inference package."""
    print("\nChecking orb-inference...")
    try:
        import orb_inference
        print(f"  orb-inference version: {orb_inference.__version__}")
        
        # Test imports
        from orb_inference import OrbInference
        from orb_inference.utils.device import get_device
        from orb_inference.utils.io import load_structure
        
        # Test device
        device = get_device()
        print(f"  Default device: {device}")
        
        print("  ✓ orb-inference OK")
        return True
        
    except ImportError as e:
        print(f"  ✗ orb-inference import failed: {e}")
        print(f"    Install with: pip install -e .")
        return False


def test_model_loading():
    """Test loading an Orb model."""
    print("\nTesting model loading (this may download model on first run)...")
    try:
        from orb_inference import OrbInference
        
        print("  Loading orb-v3-omat model...")
        orb = OrbInference(model_name='orb-v3-omat', device='cpu')
        
        print("  ✓ Model loaded successfully")
        
        # Get info
        info = orb.info()
        print(f"    Device: {info['device']}")
        print(f"    Precision: {info['precision']}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("orb-inference Installation Test")
    print("="*60)
    
    results = {
        'Python version': test_python_version(),
        'PyTorch': test_pytorch(),
        'ASE': test_ase(),
        'orb-models': test_orb_models(),
        'Phonopy': test_phonopy(),
        'Optional deps': test_optional_dependencies(),
        'orb-inference': test_orb_inference(),
        'Model loading': test_model_loading(),
    }
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:<20} {status}")
    
    all_critical_passed = all([
        results['Python version'],
        results['PyTorch'],
        results['ASE'],
        results['orb-models'],
        results['orb-inference'],
    ])
    
    print("\n" + "="*60)
    if all_critical_passed:
        print("✓ All critical tests passed!")
        print("Installation is ready for use.")
    else:
        print("✗ Some critical tests failed.")
        print("Please check the error messages above and fix issues.")
        sys.exit(1)
    
    print("="*60)


if __name__ == '__main__':
    main()
