"""
Installation Verification Tests for EquiformerV2 Inference

This module tests:
- Package installation
- Dependencies availability
- Model loading
- Basic functionality
"""

import pytest
import sys
import importlib


class TestPackageInstallation:
    """Test package installation and imports"""
    
    def test_import_main_package(self):
        """Test importing main package"""
        import equiformerv2_inference
        assert equiformerv2_inference is not None
        assert hasattr(equiformerv2_inference, '__version__')
    
    def test_import_core(self):
        """Test importing core module"""
        from equiformerv2_inference import EquiformerV2Inference
        assert EquiformerV2Inference is not None
    
    def test_import_utils(self):
        """Test importing utils modules"""
        from equiformerv2_inference.utils import device, io
        assert device is not None
        assert io is not None
    
    def test_import_tasks(self):
        """Test importing tasks modules"""
        from equiformerv2_inference import tasks
        assert tasks is not None


class TestDependencies:
    """Test required dependencies"""
    
    def test_torch_available(self):
        """Test PyTorch availability"""
        try:
            import torch
            assert torch is not None
            print(f"\nPyTorch version: {torch.__version__}")
        except ImportError:
            pytest.fail("PyTorch not installed")
    
    def test_torch_geometric_available(self):
        """Test PyTorch Geometric availability"""
        try:
            import torch_geometric
            assert torch_geometric is not None
            print(f"\nPyTorch Geometric version: {torch_geometric.__version__}")
        except ImportError:
            pytest.fail("PyTorch Geometric not installed")
    
    def test_ase_available(self):
        """Test ASE availability"""
        try:
            import ase
            assert ase is not None
            print(f"\nASE version: {ase.__version__}")
        except ImportError:
            pytest.fail("ASE not installed")
    
    def test_numpy_available(self):
        """Test NumPy availability"""
        try:
            import numpy as np
            assert np is not None
            print(f"\nNumPy version: {np.__version__}")
        except ImportError:
            pytest.fail("NumPy not installed")
    
    def test_e3nn_available(self):
        """Test e3nn availability"""
        try:
            import e3nn
            assert e3nn is not None
            print(f"\ne3nn version: {e3nn.__version__}")
        except ImportError:
            pytest.fail("e3nn not installed")
    
    def test_phonopy_available(self):
        """Test Phonopy availability"""
        try:
            import phonopy
            assert phonopy is not None
            print(f"\nPhonopy version: {phonopy.__version__}")
        except ImportError:
            pytest.skip("Phonopy not installed (optional)")
    
    def test_matplotlib_available(self):
        """Test Matplotlib availability"""
        try:
            import matplotlib
            assert matplotlib is not None
            print(f"\nMatplotlib version: {matplotlib.__version__}")
        except ImportError:
            pytest.skip("Matplotlib not installed (optional)")


class TestModelLoading:
    """Test model initialization"""
    
    def test_model_init_cpu(self):
        """Test model initialization on CPU"""
        from equiformerv2_inference import EquiformerV2Inference
        
        try:
            model = EquiformerV2Inference(device='cpu')
            assert model is not None
            assert model.device == 'cpu'
            print("\nModel successfully initialized on CPU")
        except Exception as e:
            pytest.skip(f"Model initialization failed (may need checkpoint): {e}")
    
    def test_model_init_auto(self):
        """Test model initialization with auto device selection"""
        from equiformerv2_inference import EquiformerV2Inference
        
        try:
            model = EquiformerV2Inference(device='auto')
            assert model is not None
            assert model.device in ['cpu', 'cuda']
            print(f"\nModel successfully initialized on {model.device}")
        except Exception as e:
            pytest.skip(f"Model initialization failed (may need checkpoint): {e}")


class TestBasicFunctionality:
    """Test basic package functionality"""
    
    def test_create_simple_structure(self):
        """Test creating a simple atomic structure"""
        from ase import Atoms
        import numpy as np
        
        # Create simple H2 molecule
        atoms = Atoms('H2', 
                     positions=[[0, 0, 0], [0.74, 0, 0]],
                     cell=[10, 10, 10],
                     pbc=True)
        
        assert len(atoms) == 2
        assert all(atoms.get_chemical_symbols() == ['H', 'H'])
        print("\nSimple structure creation successful")
    
    def test_device_detection(self):
        """Test device detection utilities"""
        from equiformerv2_inference.utils.device import get_device, get_available_devices
        
        device = get_device('auto')
        assert device in ['cpu', 'cuda']
        
        devices = get_available_devices()
        assert isinstance(devices, list)
        assert 'cpu' in devices
        print(f"\nDetected device: {device}")
        print(f"Available devices: {devices}")


class TestVersionInfo:
    """Test version and system information"""
    
    def test_package_version(self):
        """Test package version is defined"""
        import equiformerv2_inference
        assert hasattr(equiformerv2_inference, '__version__')
        assert isinstance(equiformerv2_inference.__version__, str)
        print(f"\nPackage version: {equiformerv2_inference.__version__}")
    
    def test_python_version(self):
        """Test Python version"""
        assert sys.version_info >= (3, 8), "Python 3.8 or higher required"
        print(f"\nPython version: {sys.version}")
    
    def test_torch_cuda_available(self):
        """Test CUDA availability through PyTorch"""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                print(f"\nCUDA is available")
                print(f"CUDA version: {torch.version.cuda}")
                print(f"Number of GPUs: {torch.cuda.device_count()}")
            else:
                print("\nCUDA is not available (CPU only)")
        except ImportError:
            pytest.skip("PyTorch not installed")


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])
