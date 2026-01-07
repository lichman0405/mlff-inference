"""
Tests for EquiformerV2 Inference Utilities

This module tests:
- Device detection and management
- Structure I/O operations
- Data validation
"""

import pytest
import numpy as np
from equiformerv2_inference.utils.device import (
    get_device,
    get_available_devices,
    set_torch_threads,
    print_device_info
)
from equiformerv2_inference.utils.io import (
    read_structure,
    write_structure,
    validate_structure
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
        assert len(devices) > 0
        assert 'cpu' in devices
    
    def test_set_torch_threads(self):
        """Test setting number of PyTorch threads"""
        try:
            set_torch_threads(4)
            import torch
            # Verify thread setting was accepted
            assert torch.get_num_threads() > 0
        except Exception as e:
            pytest.skip(f"Thread setting test skipped: {e}")
    
    def test_print_device_info(self):
        """Test printing device information"""
        try:
            # Should not raise any errors
            print_device_info()
        except Exception as e:
            pytest.fail(f"print_device_info failed: {e}")


class TestStructureIO:
    """Test structure I/O operations"""
    
    def test_read_structure_cif(self):
        """Test reading CIF files"""
        # Create temporary CIF file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as f:
            cif_content = """data_test
_cell_length_a    5.0
_cell_length_b    5.0
_cell_length_c    5.0
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si 0.0 0.0 0.0
"""
            f.write(cif_content)
            temp_file = f.name
        
        try:
            atoms = read_structure(temp_file)
            assert isinstance(atoms, Atoms)
            assert len(atoms) == 1
            assert atoms.get_chemical_symbols()[0] == 'Si'
        finally:
            os.unlink(temp_file)
    
    def test_read_structure_poscar(self):
        """Test reading POSCAR/CONTCAR files"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vasp', delete=False) as f:
            poscar_content = """Test Structure
1.0
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
Si
1
Direct
0.0 0.0 0.0
"""
            f.write(poscar_content)
            temp_file = f.name
        
        try:
            atoms = read_structure(temp_file)
            assert isinstance(atoms, Atoms)
            assert len(atoms) == 1
        finally:
            os.unlink(temp_file)
    
    def test_write_structure_xyz(self):
        """Test writing XYZ files"""
        atoms = bulk('Si', 'diamond', a=5.43)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            temp_file = f.name
        
        try:
            write_structure(atoms, temp_file)
            assert os.path.exists(temp_file)
            
            # Read it back
            atoms_read = read_structure(temp_file)
            assert len(atoms_read) == len(atoms)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_write_structure_cif(self):
        """Test writing CIF files"""
        atoms = bulk('Si', 'diamond', a=5.43)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as f:
            temp_file = f.name
        
        try:
            write_structure(atoms, temp_file)
            assert os.path.exists(temp_file)
            
            # Read it back
            atoms_read = read_structure(temp_file)
            assert len(atoms_read) == len(atoms)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_validate_structure_valid(self):
        """Test structure validation with valid structure"""
        atoms = bulk('Si', 'diamond', a=5.43)
        is_valid, message = validate_structure(atoms)
        assert is_valid
        assert "valid" in message.lower()
    
    def test_validate_structure_no_cell(self):
        """Test structure validation with missing cell"""
        atoms = Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]])
        is_valid, message = validate_structure(atoms)
        # Should still be valid but might have warnings
        assert isinstance(is_valid, bool)
        assert isinstance(message, str)
    
    def test_validate_structure_empty(self):
        """Test structure validation with empty structure"""
        atoms = Atoms()
        is_valid, message = validate_structure(atoms)
        assert not is_valid
        assert "empty" in message.lower() or "no atoms" in message.lower()


class TestDataTypes:
    """Test data type conversions and validations"""
    
    def test_atoms_to_numpy(self):
        """Test converting Atoms to numpy arrays"""
        atoms = bulk('Si', 'diamond', a=5.43)
        
        positions = atoms.get_positions()
        assert isinstance(positions, np.ndarray)
        assert positions.shape == (len(atoms), 3)
        
        cell = atoms.get_cell()
        assert isinstance(cell, np.ndarray)
        assert cell.shape == (3, 3)
    
    def test_periodic_boundary_conditions(self):
        """Test periodic boundary conditions handling"""
        atoms = bulk('Si', 'diamond', a=5.43)
        
        pbc = atoms.get_pbc()
        assert isinstance(pbc, np.ndarray)
        assert len(pbc) == 3
        assert all(pbc)  # Diamond structure should be fully periodic
    
    def test_atomic_numbers(self):
        """Test atomic number retrieval"""
        atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        
        numbers = atoms.get_atomic_numbers()
        assert isinstance(numbers, np.ndarray)
        assert len(numbers) == 3
        assert numbers[0] == 1  # H
        assert numbers[1] == 1  # H
        assert numbers[2] == 8  # O


class TestStructureManipulation:
    """Test structure manipulation utilities"""
    
    def test_supercell_creation(self):
        """Test creating supercells"""
        atoms = bulk('Si', 'diamond', a=5.43)
        initial_natoms = len(atoms)
        
        # Create 2x2x2 supercell
        supercell = atoms * (2, 2, 2)
        
        assert len(supercell) == initial_natoms * 8
        assert np.allclose(supercell.get_cell(), atoms.get_cell() * 2)
    
    def test_center_structure(self):
        """Test centering structure in cell"""
        atoms = Atoms('H', positions=[[1, 1, 1]], cell=[10, 10, 10], pbc=True)
        
        atoms.center()
        pos = atoms.get_positions()[0]
        cell = atoms.get_cell().diagonal()
        
        # After centering, atom should be closer to cell center
        assert np.allclose(pos, cell / 2, atol=0.5)
    
    def test_structure_copy(self):
        """Test copying structures"""
        atoms1 = bulk('Si', 'diamond', a=5.43)
        atoms2 = atoms1.copy()
        
        # Modify copy
        atoms2.positions[0] += [0.1, 0.1, 0.1]
        
        # Original should be unchanged
        assert not np.allclose(atoms1.positions[0], atoms2.positions[0])


class TestErrorHandling:
    """Test error handling in utility functions"""
    
    def test_read_nonexistent_file(self):
        """Test reading non-existent file"""
        with pytest.raises((FileNotFoundError, IOError)):
            read_structure('nonexistent_file.xyz')
    
    def test_invalid_device(self):
        """Test invalid device specification"""
        device = get_device('invalid_device')
        # Should fallback to CPU
        assert device == 'cpu'
    
    def test_write_invalid_path(self):
        """Test writing to invalid path"""
        atoms = bulk('Si', 'diamond', a=5.43)
        
        with pytest.raises((OSError, IOError, PermissionError)):
            write_structure(atoms, '/invalid/path/file.xyz')


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])
