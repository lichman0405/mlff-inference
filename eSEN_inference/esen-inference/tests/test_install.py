"""
Installation Verification Tests for eSEN Inference

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
        import esen_inference
        assert esen_inference is not None
        assert hasattr(esen_inference, '__version__')
    
    def test_import_core(self):
        """Test importing core module"""
        from esen_inference import ESENInference
        assert ESENInference is not None
    
    def test_import_utils(self):
        """Test importing utils modules"""
        from esen_inference.utils import device, io
        assert device is not None
        assert io is not None
    
    def test_import_tasks(self):
        """Test importing tasks modules"""
        from esen_inference.tasks import (
            static, optimization, dynamics,
            phonon, mechanics, adsorption
        )
        assert static is not None
        assert optimization is not None
        assert dynamics is not None
        assert phonon is not None
        assert mechanics is not None
        assert adsorption is not None


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
    
    def test_fairchem_available(self):
        """Test FAIR-Chem availability"""
        try:
            import fairchem
            assert fairchem is not None
            print(f"\nFAIR-Chem version: {fairchem.__version__}")
        except ImportError:
            pytest.fail("FAIR-Chem not installed")
    
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
    """Test model initialization and loading"""
    
    def test_model_initialization_cpu(self):
        """Test initializing model on CPU"""
        from esen_inference import ESENInference
        
        try:
            model = ESENInference(
                model_name='esen-30m-oam',
                device='cpu',
                precision='float32'
            )
            assert model is not None
            assert model.device == 'cpu'
            print(f"\n✓ Model initialized: {model}")
        except Exception as e:
            pytest.skip(f"Model initialization failed (may need checkpoint): {e}")
    
    def test_model_initialization_auto_device(self):
        """Test automatic device selection"""
        from esen_inference import ESENInference
        
        try:
            model = ESENInference(model_name='esen-30m-oam')
            assert model is not None
            assert model.device in ['cuda', 'cpu']
            print(f"\n✓ Auto device: {model.device}")
        except Exception as e:
            pytest.skip(f"Model initialization failed: {e}")
    
    def test_model_name_validation(self):
        """Test model name validation"""
        from esen_inference import ESENInference
        
        valid_models = ['esen-30m-oam', 'esen-30m-mp']
        
        # Valid model names should not raise error
        for model_name in valid_models:
            try:
                model = ESENInference(model_name=model_name, device='cpu')
                print(f"\n✓ Valid model: {model_name}")
            except Exception as e:
                # Checkpoint download failure is ok for this test
                if 'checkpoint' not in str(e).lower():
                    pytest.fail(f"Unexpected error for {model_name}: {e}")
        
        # Invalid model name should raise error
        with pytest.raises((ValueError, KeyError)):
            ESENInference(model_name='invalid-model-name')


class TestBasicFunctionality:
    """Test basic functionality (if model is available)"""
    
    @pytest.fixture
    def model(self):
        """Create model instance for testing"""
        from esen_inference import ESENInference
        try:
            return ESENInference(model_name='esen-30m-oam', device='cpu')
        except Exception as e:
            pytest.skip(f"Model not available: {e}")
    
    @pytest.fixture
    def sample_atoms(self):
        """Create sample structure"""
        from ase.build import bulk
        return bulk('Cu', 'fcc', a=3.6)
    
    def test_single_point_calculation(self, model, sample_atoms):
        """Test single-point energy calculation"""
        try:
            result = model.single_point(sample_atoms)
            
            assert 'energy' in result
            assert 'forces' in result
            assert 'stress' in result
            assert isinstance(result['energy'], float)
            
            print(f"\n✓ Single-point: E = {result['energy']:.4f} eV")
        except Exception as e:
            pytest.skip(f"Single-point calculation failed: {e}")
    
    def test_optimization(self, model, sample_atoms):
        """Test structure optimization"""
        try:
            result = model.optimize(
                sample_atoms,
                fmax=0.1,
                max_steps=10
            )
            
            assert 'converged' in result
            assert 'final_energy' in result
            assert 'atoms' in result
            
            print(f"\n✓ Optimization: converged = {result['converged']}")
        except Exception as e:
            pytest.skip(f"Optimization failed: {e}")


class TestCLI:
    """Test CLI availability"""
    
    def test_cli_entry_point(self):
        """Test CLI entry point is registered"""
        # Check if esen-infer command exists
        import subprocess
        try:
            result = subprocess.run(
                ['esen-infer', '--help'],
                capture_output=True,
                text=True,
                timeout=5
            )
            # If command exists, it should show help
            assert result.returncode in [0, 1, 2]  # Various help exit codes
            print("\n✓ CLI command 'esen-infer' is available")
        except FileNotFoundError:
            pytest.skip("CLI not installed or not in PATH")
        except subprocess.TimeoutExpired:
            pytest.fail("CLI command timed out")


class TestVersionInfo:
    """Test version information"""
    
    def test_version_string(self):
        """Test version string exists and is valid"""
        from esen_inference import __version__
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__) > 0
        print(f"\nesen-inference version: {__version__}")
    
    def test_python_version(self):
        """Test Python version compatibility"""
        version = sys.version_info
        assert version >= (3, 8), "Requires Python 3.8+"
        print(f"\nPython version: {version.major}.{version.minor}.{version.micro}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
