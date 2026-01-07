"""
Tests for orb_inference utilities.
"""

import pytest
import torch
from ase import Atoms
from ase.build import bulk
from pathlib import Path
import tempfile

from orb_inference.utils.device import (
    get_device, 
    validate_device, 
    get_device_info,
    print_device_info
)
from orb_inference.utils.io import (
    load_structure,
    save_structure,
    parse_structure_input,
    create_supercell,
    atoms_to_dict,
    dict_to_atoms
)


class TestDevice:
    """Tests for device utilities."""
    
    def test_get_device_returns_valid_device(self):
        """Test that get_device returns a valid PyTorch device."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cuda', 'cpu', 'mps']
    
    def test_validate_device_cpu(self):
        """Test CPU device validation."""
        device = validate_device('cpu')
        assert device.type == 'cpu'
    
    def test_validate_device_cuda_when_available(self):
        """Test CUDA device validation when available."""
        if torch.cuda.is_available():
            device = validate_device('cuda')
            assert device.type == 'cuda'
        else:
            with pytest.warns(UserWarning):
                device = validate_device('cuda')
                assert device.type == 'cpu'
    
    def test_validate_device_invalid_raises(self):
        """Test that invalid device string raises ValueError."""
        with pytest.raises(ValueError):
            validate_device('invalid_device')
    
    def test_get_device_info(self):
        """Test device info retrieval."""
        device = get_device()
        info = get_device_info(device)
        
        assert isinstance(info, dict)
        assert 'type' in info
        assert 'name' in info
        
        if device.type == 'cuda':
            assert 'memory_total' in info
            assert 'memory_allocated' in info


class TestIO:
    """Tests for I/O utilities."""
    
    @pytest.fixture
    def sample_atoms(self):
        """Create a sample Atoms object."""
        return bulk('Cu', 'fcc', a=3.6)
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_atoms_to_dict_and_back(self, sample_atoms):
        """Test round-trip conversion Atoms -> dict -> Atoms."""
        atoms_dict = atoms_to_dict(sample_atoms)
        
        assert isinstance(atoms_dict, dict)
        assert 'symbols' in atoms_dict
        assert 'positions' in atoms_dict
        assert 'cell' in atoms_dict
        assert 'pbc' in atoms_dict
        
        atoms_reconstructed = dict_to_atoms(atoms_dict)
        
        assert len(atoms_reconstructed) == len(sample_atoms)
        assert all(atoms_reconstructed.get_chemical_symbols() == 
                  sample_atoms.get_chemical_symbols())
        assert atoms_reconstructed.cell == pytest.approx(sample_atoms.cell.array)
    
    def test_save_and_load_structure_cif(self, sample_atoms, temp_dir):
        """Test saving and loading CIF files."""
        filepath = temp_dir / 'test.cif'
        
        save_structure(sample_atoms, filepath)
        assert filepath.exists()
        
        loaded_atoms = load_structure(filepath)
        assert len(loaded_atoms) == len(sample_atoms)
    
    def test_save_and_load_structure_vasp(self, sample_atoms, temp_dir):
        """Test saving and loading VASP POSCAR files."""
        filepath = temp_dir / 'POSCAR'
        
        save_structure(sample_atoms, filepath, format='vasp')
        assert filepath.exists()
        
        loaded_atoms = load_structure(filepath)
        assert len(loaded_atoms) == len(sample_atoms)
    
    def test_parse_structure_input_atoms(self, sample_atoms):
        """Test parsing Atoms object input."""
        result = parse_structure_input(sample_atoms)
        assert isinstance(result, Atoms)
        assert len(result) == len(sample_atoms)
    
    def test_parse_structure_input_filepath(self, sample_atoms, temp_dir):
        """Test parsing filepath input."""
        filepath = temp_dir / 'test.cif'
        save_structure(sample_atoms, filepath)
        
        result = parse_structure_input(str(filepath))
        assert isinstance(result, Atoms)
        assert len(result) == len(sample_atoms)
    
    def test_create_supercell(self, sample_atoms):
        """Test supercell creation."""
        supercell = create_supercell(sample_atoms, [2, 2, 2])
        
        assert len(supercell) == len(sample_atoms) * 8
        assert supercell.get_volume() == pytest.approx(
            sample_atoms.get_volume() * 8
        )
    
    def test_load_nonexistent_file_raises(self):
        """Test that loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_structure('nonexistent_file.cif')
    
    def test_save_unsupported_format_raises(self, sample_atoms, temp_dir):
        """Test that unsupported format raises ValueError."""
        with pytest.raises(ValueError):
            save_structure(sample_atoms, temp_dir / 'test.xyz', format='unsupported')


class TestIntegration:
    """Integration tests combining multiple utilities."""
    
    def test_device_and_io_workflow(self):
        """Test combined device selection and I/O operations."""
        # Get device
        device = get_device()
        assert device is not None
        
        # Create structure
        atoms = bulk('Si', 'diamond', a=5.43)
        
        # Convert to dict and back
        atoms_dict = atoms_to_dict(atoms)
        atoms_restored = dict_to_atoms(atoms_dict)
        
        assert len(atoms_restored) == len(atoms)
        
        # Create supercell
        supercell = create_supercell(atoms, [2, 2, 2])
        assert len(supercell) == len(atoms) * 8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
