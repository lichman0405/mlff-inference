"""
Utils module tests.

Test the functionality of device and io modules.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk


class TestDeviceUtils:
    """Device utils tests."""
    
    def test_get_device_auto(self):
        """Test automatic device detection."""
        from mattersim_inference.utils.device import get_device
        
        device = get_device("auto")
        assert device in ["cpu", "cuda"]
    
    def test_get_device_cpu(self):
        """Test CPU device."""
        from mattersim_inference.utils.device import get_device
        
        device = get_device("cpu")
        assert device == "cpu"
    
    def test_get_available_devices(self):
        """Test available devices list."""
        from mattersim_inference.utils.device import get_available_devices
        
        devices = get_available_devices()
        assert isinstance(devices, list)
        assert "cpu" in devices
    
    def test_check_gpu_memory(self):
        """Test GPU memory check."""
        from mattersim_inference.utils.device import check_gpu_memory
        
        result = check_gpu_memory()
        assert isinstance(result, dict)


class TestIOUtils:
    """I/O utils tests."""
    
    def test_read_structure_cif(self):
        """Test reading CIF file."""
        from mattersim_inference.utils.io import read_structure, write_structure
        
        # Create and save test structure
        atoms = bulk("Cu", "fcc", a=3.6)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.cif"
            write_structure(atoms, filepath)
            
            # Read
            loaded = read_structure(filepath)
            assert len(loaded) == len(atoms)
            assert loaded.get_chemical_formula() == atoms.get_chemical_formula()
    
    def test_read_structure_xyz(self):
        """Test reading XYZ file."""
        from mattersim_inference.utils.io import read_structure, write_structure
        
        atoms = bulk("Fe", "bcc", a=2.87)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.xyz"
            write_structure(atoms, filepath)
            
            loaded = read_structure(filepath)
            assert len(loaded) == len(atoms)
    
    def test_write_structure_creates_dir(self):
        """Test creating directory when writing."""
        from mattersim_inference.utils.io import write_structure
        
        atoms = bulk("Al", "fcc", a=4.05)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "test.cif"
            write_structure(atoms, filepath)
            assert filepath.exists()
    
    def test_validate_structure_valid(self):
        """Test validating valid structure."""
        from mattersim_inference.utils.io import validate_structure
        
        atoms = bulk("Cu", "fcc", a=3.6)
        is_valid, message = validate_structure(atoms)
        
        assert is_valid is True
        assert "valid" in message.lower()
    
    def test_validate_structure_empty(self):
        """Test validating empty structure."""
        from mattersim_inference.utils.io import validate_structure
        
        atoms = Atoms()
        is_valid, message = validate_structure(atoms)
        
        assert is_valid is False
        assert "no atoms" in message.lower()
    
    def test_get_structure_info(self):
        """Test getting structure information."""
        from mattersim_inference.utils.io import get_structure_info
        
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
        info = get_structure_info(atoms)
        
        assert info["n_atoms"] == 32
        assert "Cu" in info["composition"]
        assert info["composition"]["Cu"] == 32
        assert info["volume"] > 0
    
    def test_read_nonexistent_file(self):
        """Test reading non-existent file."""
        from mattersim_inference.utils.io import read_structure
        
        with pytest.raises(FileNotFoundError):
            read_structure("nonexistent_file.cif")
    
    def test_read_trajectory(self):
        """Test reading trajectory file."""
        from ase.io import write as ase_write
        from mattersim_inference.utils.io import read_trajectory
        
        atoms = bulk("Cu", "fcc", a=3.6)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "traj.xyz"
            
            # Write multiple frames
            frames = [atoms.copy() for _ in range(3)]
            ase_write(str(filepath), frames)
            
            loaded = read_trajectory(filepath)
            assert len(loaded) == 3


class TestStaticTask:
    """Static calculation task tests (requires MatterSim)."""
    
    @pytest.fixture(autouse=True)
    def check_mattersim(self):
        """Check MatterSim availability."""
        try:
            import mattersim
            self.available = True
        except ImportError:
            self.available = False
            pytest.skip("MatterSim not installed")
    
    def test_single_point_result_keys(self):
        """Test single point calculation result keys."""
        from mattersim_inference import MatterSimInference
        
        calc = MatterSimInference(device="cpu")
        atoms = bulk("Cu", "fcc", a=3.6)
        
        result = calc.single_point(atoms)
        
        assert "energy" in result
        assert "energy_per_atom" in result
        assert "forces" in result
        assert "stress" in result
        assert "max_force" in result
        assert "pressure" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
