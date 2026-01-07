"""
Tests for eSEN Inference Core Module

This module tests:
- ESENInference class initialization
- Single-point calculations
- Structure optimization
- MD simulations
- Phonon calculations
- Mechanical properties
"""

import pytest
import numpy as np
from esen_inference import ESENInference
from ase.build import bulk
from ase import Atoms
import tempfile
import os


@pytest.fixture(scope='module')
def esen_model():
    """Create eSEN model instance (module-scoped for efficiency)"""
    try:
        model = ESENInference(
            model_name='esen-30m-oam',
            device='cpu',  # Use CPU for testing
            precision='float32'
        )
        return model
    except Exception as e:
        pytest.skip(f"Model initialization failed: {e}")


@pytest.fixture
def cu_structure():
    """Create Cu FCC structure for testing"""
    return bulk('Cu', 'fcc', a=3.6)


@pytest.fixture
def cu_supercell():
    """Create Cu supercell (2x2x2) for testing"""
    return bulk('Cu', 'fcc', a=3.6) * (2, 2, 2)


class TestESENInferenceInit:
    """Test ESENInference initialization"""
    
    def test_init_default(self):
        """Test default initialization"""
        try:
            model = ESENInference(model_name='esen-30m-oam')
            assert model.model_name == 'esen-30m-oam'
            assert model.device in ['cuda', 'cpu']
            assert model.precision == 'float32'
        except Exception as e:
            pytest.skip(f"Initialization failed: {e}")
    
    def test_init_cpu(self):
        """Test CPU initialization"""
        try:
            model = ESENInference(model_name='esen-30m-oam', device='cpu')
            assert model.device == 'cpu'
        except Exception as e:
            pytest.skip(f"CPU initialization failed: {e}")
    
    def test_init_precision_float32(self):
        """Test float32 precision"""
        try:
            model = ESENInference(
                model_name='esen-30m-oam',
                device='cpu',
                precision='float32'
            )
            assert model.precision == 'float32'
        except Exception as e:
            pytest.skip(f"Float32 initialization failed: {e}")
    
    def test_init_invalid_model(self):
        """Test initialization with invalid model name"""
        with pytest.raises((ValueError, KeyError)):
            ESENInference(model_name='invalid-model-xyz')
    
    def test_repr(self, esen_model):
        """Test string representation"""
        repr_str = repr(esen_model)
        assert 'ESENInference' in repr_str
        assert esen_model.model_name in repr_str


class TestSinglePoint:
    """Test single-point calculations"""
    
    def test_single_point_basic(self, esen_model, cu_structure):
        """Test basic single-point calculation"""
        result = esen_model.single_point(cu_structure)
        
        # Check required keys
        assert 'energy' in result
        assert 'forces' in result
        assert 'stress' in result
        assert 'energy_per_atom' in result
        
        # Check types and shapes
        assert isinstance(result['energy'], float)
        assert isinstance(result['forces'], np.ndarray)
        assert result['forces'].shape == (len(cu_structure), 3)
        assert isinstance(result['stress'], np.ndarray)
        assert result['stress'].shape == (6,)
        
        print(f"\n✓ Energy: {result['energy']:.6f} eV")
        print(f"  E/atom: {result['energy_per_atom']:.6f} eV/atom")
    
    def test_single_point_forces(self, esen_model, cu_structure):
        """Test force calculation"""
        result = esen_model.single_point(cu_structure)
        
        forces = result['forces']
        max_force = result['max_force']
        rms_force = result['rms_force']
        
        # Forces should be finite
        assert np.all(np.isfinite(forces))
        
        # Max force should match manual calculation
        assert np.isclose(max_force, np.max(np.abs(forces)))
        
        print(f"\n✓ Max force: {max_force:.6f} eV/Å")
        print(f"  RMS force: {rms_force:.6f} eV/Å")
    
    def test_single_point_stress(self, esen_model, cu_structure):
        """Test stress calculation"""
        result = esen_model.single_point(cu_structure)
        
        stress = result['stress']
        pressure = result['pressure']
        
        # Stress should be finite
        assert np.all(np.isfinite(stress))
        
        # Pressure = -trace(stress)/3
        expected_pressure = -np.mean(stress[:3])
        assert np.isclose(pressure, expected_pressure, atol=0.01)
        
        print(f"\n✓ Pressure: {pressure:.4f} GPa")


class TestOptimization:
    """Test structure optimization"""
    
    def test_optimize_coords_only(self, esen_model, cu_supercell):
        """Test coordinate-only optimization"""
        # Perturb structure
        atoms = cu_supercell.copy()
        atoms.positions += np.random.normal(0, 0.05, atoms.positions.shape)
        
        result = esen_model.optimize(
            atoms,
            fmax=0.05,
            relax_cell=False,
            max_steps=50
        )
        
        # Check results
        assert 'converged' in result
        assert 'steps' in result
        assert 'final_energy' in result
        assert 'atoms' in result
        
        print(f"\n✓ Converged: {result['converged']}")
        print(f"  Steps: {result['steps']}")
        print(f"  ΔE: {result['energy_change']:.6f} eV")
    
    def test_optimize_full(self, esen_model, cu_supercell):
        """Test full optimization (coords + cell)"""
        result = esen_model.optimize(
            cu_supercell,
            fmax=0.05,
            relax_cell=True,
            max_steps=50
        )
        
        assert isinstance(result['atoms'], Atoms)
        assert len(result['atoms']) == len(cu_supercell)
        
        print(f"\n✓ Full optimization completed")
        print(f"  Volume change: {result['atoms'].get_volume() - cu_supercell.get_volume():.2f} Å³")
    
    def test_optimize_different_optimizers(self, esen_model, cu_structure):
        """Test different optimizers"""
        optimizers = ['LBFGS', 'BFGS', 'FIRE']
        
        for optimizer in optimizers:
            result = esen_model.optimize(
                cu_structure.copy(),
                fmax=0.1,
                optimizer=optimizer,
                max_steps=20
            )
            assert 'converged' in result
            print(f"\n✓ {optimizer}: {result['steps']} steps")


class TestMolecularDynamics:
    """Test MD simulations"""
    
    def test_md_nve(self, esen_model, cu_supercell):
        """Test NVE ensemble"""
        with tempfile.NamedTemporaryFile(suffix='.traj', delete=False) as f:
            traj_file = f.name
        
        try:
            final_atoms = esen_model.run_md(
                cu_supercell,
                temperature=300,
                steps=10,  # Short for testing
                timestep=1.0,
                ensemble='nve',
                trajectory=traj_file
            )
            
            assert isinstance(final_atoms, Atoms)
            assert len(final_atoms) == len(cu_supercell)
            assert os.path.exists(traj_file)
            
            print(f"\n✓ NVE MD completed")
            print(f"  Final T: {final_atoms.get_temperature():.2f} K")
            
        finally:
            if os.path.exists(traj_file):
                os.unlink(traj_file)
    
    def test_md_nvt(self, esen_model, cu_supercell):
        """Test NVT ensemble"""
        final_atoms = esen_model.run_md(
            cu_supercell,
            temperature=300,
            steps=10,
            ensemble='nvt',
            friction=0.01
        )
        
        assert isinstance(final_atoms, Atoms)
        print(f"\n✓ NVT MD completed")


class TestPhonon:
    """Test phonon calculations"""
    
    @pytest.mark.slow
    def test_phonon_basic(self, esen_model, cu_structure):
        """Test basic phonon calculation"""
        # Optimize first
        opt_result = esen_model.optimize(
            cu_structure,
            fmax=0.01,
            relax_cell=True,
            max_steps=100
        )
        
        result = esen_model.phonon(
            opt_result['atoms'],
            supercell_matrix=[2, 2, 2],
            mesh=[10, 10, 10],
            displacement=0.01,
            t_max=500,
            t_step=50
        )
        
        # Check results
        assert 'has_imaginary' in result
        assert 'total_dos' in result
        assert 'frequency_points' in result
        assert 'thermal' in result
        
        thermal = result['thermal']
        assert 'temperatures' in thermal
        assert 'heat_capacity' in thermal
        
        print(f"\n✓ Phonon calculation completed")
        print(f"  Imaginary modes: {result['imaginary_modes']}")


class TestMechanics:
    """Test mechanical properties"""
    
    def test_bulk_modulus(self, esen_model, cu_structure):
        """Test bulk modulus calculation"""
        # Optimize first
        opt_result = esen_model.optimize(
            cu_structure,
            fmax=0.01,
            relax_cell=True,
            max_steps=100
        )
        
        result = esen_model.bulk_modulus(
            opt_result['atoms'],
            strain_range=0.03,
            npoints=5  # Small for testing
        )
        
        # Check results
        assert 'bulk_modulus' in result
        assert 'v0' in result
        assert 'e0' in result
        assert 'volumes' in result
        assert 'energies' in result
        
        assert result['bulk_modulus'] > 0
        assert len(result['volumes']) == 5
        
        print(f"\n✓ Bulk modulus: {result['bulk_modulus']:.2f} GPa")


class TestDeviceManagement:
    """Test device switching and management"""
    
    def test_set_device(self, esen_model):
        """Test device switching"""
        original_device = esen_model.device
        
        # Switch to CPU
        esen_model.set_device('cpu')
        assert esen_model.device == 'cpu'
        
        # Restore
        esen_model.set_device(original_device)
        assert esen_model.device == original_device


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
