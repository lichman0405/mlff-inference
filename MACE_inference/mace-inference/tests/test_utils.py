"""Unit tests for MACE inference utilities"""

import pytest
import numpy as np
from ase.build import bulk


class TestDeviceUtils:
    """Test device management utilities"""
    
    def test_get_device_auto(self):
        from mace_inference.utils.device import get_device
        device = get_device("auto")
        assert device in ["cpu", "cuda"]
    
    def test_get_device_cpu(self):
        from mace_inference.utils.device import get_device
        device = get_device("cpu")
        assert device == "cpu"
    
    def test_invalid_device(self):
        from mace_inference.utils.device import get_device
        with pytest.raises(ValueError):
            get_device("invalid")
    
    def test_get_device_info(self):
        from mace_inference.utils.device import get_device_info
        info = get_device_info()
        assert "cuda_available" in info
        assert "pytorch_version" in info


class TestIOUtils:
    """Test I/O utilities"""
    
    def test_create_supercell(self):
        from mace_inference.utils.io import create_supercell
        
        atoms = bulk('Cu', 'fcc', a=3.6)
        supercell = create_supercell(atoms, [2, 2, 2])
        
        assert len(supercell) == len(atoms) * 8
        assert np.allclose(supercell.get_volume(), atoms.get_volume() * 8)
    
    def test_create_supercell_isotropic(self):
        from mace_inference.utils.io import create_supercell
        
        atoms = bulk('Cu', 'fcc', a=3.6)
        supercell = create_supercell(atoms, 2)
        
        assert len(supercell) == len(atoms) * 8
    
    def test_atoms_to_dict(self):
        from mace_inference.utils.io import atoms_to_dict, dict_to_atoms
        
        atoms = bulk('Cu', 'fcc', a=3.6)
        data = atoms_to_dict(atoms)
        
        assert "symbols" in data
        assert "positions" in data
        assert "cell" in data
        
        # Test round-trip
        reconstructed = dict_to_atoms(data)
        assert len(reconstructed) == len(atoms)
        assert reconstructed.get_chemical_formula() == atoms.get_chemical_formula()


class TestD3Correction:
    """Test D3 correction utilities"""
    
    def test_check_d3_available(self):
        from mace_inference.utils.d3_correction import check_d3_available
        # Should not raise error
        is_available = check_d3_available()
        assert isinstance(is_available, bool)


class TestStaticTasks:
    """Test static calculation tasks"""
    
    def test_single_point_energy_structure(self):
        """Test that single_point_energy can be called with structure"""
        from mace_inference.tasks.static import single_point_energy
        from unittest.mock import Mock
        
        atoms = bulk('Cu', 'fcc', a=3.6)
        
        # Mock calculator
        mock_calc = Mock()
        mock_calc.get_potential_energy = Mock(return_value=10.0)
        mock_calc.get_forces = Mock(return_value=np.zeros((len(atoms), 3)))
        mock_calc.get_stress = Mock(return_value=np.zeros(6))
        
        atoms.calc = mock_calc
        result = single_point_energy(atoms, mock_calc)
        
        assert "energy" in result
        assert "forces" in result
        assert "stress" in result


class TestCore:
    """Test core MACEInference class"""
    
    def test_init_cpu(self):
        """Test initialization with CPU device"""
        from mace_inference import MACEInference
        
        # This will fail if mace-torch is not installed, which is expected
        try:
            calc = MACEInference(model="medium", device="cpu")
            assert calc.device == "cpu"
        except ImportError:
            pytest.skip("mace-torch not installed")
    
    def test_repr(self):
        """Test string representation"""
        from mace_inference import MACEInference
        
        try:
            calc = MACEInference(model="medium", device="cpu")
            repr_str = repr(calc)
            assert "MACEInference" in repr_str
            assert "medium" in repr_str
        except ImportError:
            pytest.skip("mace-torch not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
