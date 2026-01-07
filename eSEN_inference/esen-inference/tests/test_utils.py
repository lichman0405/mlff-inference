"""
Tests for eSEN Inference Utilities

This module tests:
- Device detection and management
- Structure I/O operations
- Data validation
"""

import pytest
import numpy as np
from esen_inference.utils.device import (
    get_device,
    get_available_devices,
    set_torch_threads,
    print_device_info
)
from esen_inference.utils.io import (
    read_structure,
    write_structure,
    validate_structure,
    convert_structure
)
from ase import Atoms
from ase.build import bulk
import tempfile
import os


class TestDeviceUtils:
    """Test device detection and management"""
    
    def test_get_device_auto(self):
        """Test automatic device selection"""
        device = get_device('auto')
        assert device in ['cuda', 'cpu']
    
    def test_get_device_cpu(self):
        """Test CPU device selection"""
        device = get_device('cpu')
        assert device == 'cpu'
    
    def test_get_device_cuda(self):
        """Test CUDA device selection"""
        device = get_device('cuda')
        # Should either return 'cuda' or 'cpu' (if no CUDA)
        assert device in ['cuda', 'cpu']
    
    def test_get_available_devices(self):
        """Test listing available devices"""
        devices = get_available_devices()
        assert isinstance(devices, list)
        assert 'cpu' in devices
        # CUDA may or may not be available
    
    def test_set_torch_threads(self):
        """Test setting thread count"""
        # Should not raise error
        set_torch_threads(4)
        set_torch_threads('auto')
    
    def test_print_device_info(self, capsys):
        """Test printing device information"""
        print_device_info()
        captured = capsys.readouterr()
        assert 'Device Information' in captured.out


class TestIOUtils:
    """Test structure I/O operations"""
    
    @pytest.fixture
    def sample_structure(self):
        """Create sample structure for testing"""
        return bulk('Cu', 'fcc', a=3.6)
    
    def test_read_cif(self, sample_structure):
        """Test reading CIF files"""
        with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as f:
            filename = f.name
        
        try:
            # Write test file
            from ase.io import write
            write(filename, sample_structure)
            
            # Read back
            atoms = read_structure(filename)
            assert isinstance(atoms, Atoms)
            assert len(atoms) == len(sample_structure)
            
        finally:
            os.unlink(filename)
    
    def test_read_poscar(self, sample_structure):
        """Test reading POSCAR files"""
        with tempfile.NamedTemporaryFile(suffix='.poscar', delete=False, mode='w') as f:
            filename = f.name
        
        try:
            # Write test file
            from ase.io import write
            write(filename, sample_structure, format='vasp')
            
            # Read back
            atoms = read_structure(filename)
            assert isinstance(atoms, Atoms)
            
        finally:
            os.unlink(filename)
    
    def test_write_cif(self, sample_structure):
        """Test writing CIF files"""
        with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as f:
            filename = f.name
        
        try:
            write_structure(sample_structure, filename)
            assert os.path.exists(filename)
            
            # Verify readable
            from ase.io import read
            atoms = read(filename)
            assert len(atoms) == len(sample_structure)
            
        finally:
            os.unlink(filename)
    
    def test_validate_structure_valid(self, sample_structure):
        """Test validation of valid structure"""
        is_valid, message = validate_structure(sample_structure)
        assert is_valid
        assert 'valid' in message.lower()
    
    def test_validate_structure_no_atoms(self):
        """Test validation of empty structure"""
        empty_atoms = Atoms()
        is_valid, message = validate_structure(empty_atoms)
        assert not is_valid
        assert 'no atoms' in message.lower()
    
    def test_validate_structure_no_cell(self):
        """Test validation of structure without cell"""
        atoms = Atoms('Cu', positions=[(0, 0, 0)])
        is_valid, message = validate_structure(atoms)
        assert not is_valid
        assert 'cell' in message.lower()
    
    def test_validate_structure_overlapping(self):
        """Test detection of overlapping atoms"""
        atoms = bulk('Cu', 'fcc', a=3.6)
        # Create overlapping atoms
        atoms.append(Atoms('Cu', positions=[atoms.positions[0]]))
        
        is_valid, message = validate_structure(atoms)
        assert not is_valid
        assert 'overlap' in message.lower() or 'close' in message.lower()
    
    def test_convert_structure_cif_to_poscar(self, sample_structure):
        """Test format conversion"""
        with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as f:
            input_file = f.name
        with tempfile.NamedTemporaryFile(suffix='.poscar', delete=False) as f:
            output_file = f.name
        
        try:
            # Write CIF
            from ase.io import write
            write(input_file, sample_structure)
            
            # Convert
            convert_structure(input_file, output_file, output_format='vasp')
            
            # Verify
            assert os.path.exists(output_file)
            from ase.io import read
            atoms = read(output_file, format='vasp')
            assert len(atoms) == len(sample_structure)
            
        finally:
            os.unlink(input_file)
            os.unlink(output_file)
    
    def test_read_structure_invalid_file(self):
        """Test reading nonexistent file"""
        with pytest.raises(FileNotFoundError):
            read_structure('nonexistent_file.cif')
    
    def test_write_structure_invalid_format(self, sample_structure):
        """Test writing with invalid format"""
        with pytest.raises(ValueError):
            write_structure(sample_structure, 'output.xyz', format='invalid_format')


class TestIntegration:
    """Integration tests for utils"""
    
    def test_device_and_io_workflow(self):
        """Test complete workflow using device and I/O utils"""
        # Get device
        device = get_device('auto')
        assert device is not None
        
        # Create structure
        atoms = bulk('Al', 'fcc', a=4.05)
        
        # Validate
        is_valid, _ = validate_structure(atoms)
        assert is_valid
        
        # Write and read back
        with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as f:
            filename = f.name
        
        try:
            write_structure(atoms, filename)
            atoms_read = read_structure(filename)
            
            # Verify
            assert len(atoms_read) == len(atoms)
            assert np.allclose(atoms_read.positions, atoms.positions, atol=1e-3)
            
        finally:
            os.unlink(filename)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
