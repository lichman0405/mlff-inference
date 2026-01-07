"""
Installation verification tests.

Test whether the mattersim_inference package is correctly installed.
"""

import sys
import pytest


def test_import_package():
    """Test importing the main package."""
    import mattersim_inference
    assert hasattr(mattersim_inference, "__version__")
    assert hasattr(mattersim_inference, "MatterSimInference")


def test_import_core():
    """Test importing core module."""
    from mattersim_inference import MatterSimInference
    from mattersim_inference.core import get_available_models
    
    models = get_available_models()
    assert isinstance(models, list)
    assert "MatterSim-v1-5M" in models
    assert "MatterSim-v1-1M" in models


def test_import_utils():
    """Test importing utils module."""
    from mattersim_inference.utils import (
        get_device,
        get_available_devices,
        read_structure,
        write_structure,
        validate_structure,
    )
    
    assert callable(get_device)
    assert callable(get_available_devices)
    assert callable(read_structure)


def test_import_tasks():
    """Test importing tasks module."""
    from mattersim_inference.tasks import (
        calculate_single_point,
        run_md,
        calculate_phonon,
        calculate_bulk_modulus,
        calculate_adsorption_energy,
    )
    
    assert callable(calculate_single_point)
    assert callable(run_md)
    assert callable(calculate_adsorption_energy)


def test_import_cli():
    """Test importing CLI module."""
    from mattersim_inference.cli import main, create_parser
    
    assert callable(main)
    parser = create_parser()
    assert parser is not None


def test_version():
    """Test version number."""
    from mattersim_inference import __version__
    
    assert isinstance(__version__, str)
    assert len(__version__.split(".")) >= 2


def test_device_detection():
    """Test device detection."""
    from mattersim_inference.utils import get_device, get_available_devices
    
    device = get_device("auto")
    assert device in ["cpu", "cuda"] or device.startswith("cuda:")
    
    devices = get_available_devices()
    assert "cpu" in devices


def test_dependencies():
    """Test whether dependencies are available."""
    # ASE
    import ase
    assert hasattr(ase, "Atoms")
    
    # NumPy
    import numpy as np
    assert np.__version__
    
    # Phonopy (optional)
    try:
        import phonopy
        assert hasattr(phonopy, "Phonopy")
    except ImportError:
        pytest.skip("phonopy not installed")


class TestMattersimAvailable:
    """MatterSim availability test (optional skip)."""
    
    @pytest.fixture(autouse=True)
    def check_mattersim(self):
        """Check whether MatterSim is installed."""
        try:
            import mattersim
            self.mattersim_available = True
        except ImportError:
            self.mattersim_available = False
            pytest.skip("MatterSim not installed")
    
    def test_mattersim_calculator(self):
        """Test MatterSim calculator."""
        from mattersim.forcefield import MatterSimCalculator
        assert MatterSimCalculator is not None
    
    def test_inference_init(self):
        """Test inference class initialization."""
        from mattersim_inference import MatterSimInference
        
        calc = MatterSimInference(
            model_name="MatterSim-v1-5M",
            device="cpu"
        )
        assert calc.model_name == "MatterSim-v1-5M"
        assert calc.device == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
