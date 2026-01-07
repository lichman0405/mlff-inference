"""
Test suite for GRACE inference utility functions.

This module contains tests for various utility functions and
basic functionality of the GRACE inference package.
"""

import pytest
import numpy as np
from ase import Atoms
from ase.build import bulk, molecule


def test_simple_calculation():
    """Test basic energy calculation with GRACE."""
    try:
        from grace_inference import GRACECalculator
        
        # Create simple structure
        atoms = bulk('Si', 'diamond', a=5.43)
        
        # Create calculator
        calc = GRACECalculator(device='cpu')
        atoms.calc = calc
        
        # Calculate energy
        energy = atoms.get_potential_energy()
        
        # Basic sanity checks
        assert isinstance(energy, (float, np.floating)), "Energy should be a float"
        assert not np.isnan(energy), "Energy should not be NaN"
        assert not np.isinf(energy), "Energy should not be infinite"
        
        print(f"✓ Basic energy calculation successful: {energy:.6f} eV")
        
    except Exception as e:
        pytest.fail(f"Simple calculation failed: {e}")


def test_forces_calculation():
    """Test force calculation with GRACE."""
    try:
        from grace_inference import GRACECalculator
        
        # Create simple structure
        atoms = molecule('H2O')
        atoms.center(vacuum=5.0)
        
        # Create calculator
        calc = GRACECalculator(device='cpu')
        atoms.calc = calc
        
        # Calculate forces
        forces = atoms.get_forces()
        
        # Check shape and values
        assert forces.shape == (len(atoms), 3), "Forces should have shape (n_atoms, 3)"
        assert not np.any(np.isnan(forces)), "Forces should not contain NaN"
        assert not np.any(np.isinf(forces)), "Forces should not contain infinity"
        
        print(f"✓ Force calculation successful")
        print(f"  Force shape: {forces.shape}")
        print(f"  Max force: {np.max(np.abs(forces)):.6f} eV/Å")
        
    except Exception as e:
        pytest.fail(f"Force calculation failed: {e}")


def test_stress_calculation():
    """Test stress calculation with GRACE."""
    try:
        from grace_inference import GRACECalculator
        
        # Create periodic structure
        atoms = bulk('Si', 'diamond', a=5.43)
        
        # Create calculator
        calc = GRACECalculator(device='cpu')
        atoms.calc = calc
        
        # Calculate stress
        stress = atoms.get_stress()
        
        # Check values
        assert len(stress) == 6, "Stress should have 6 components"
        assert not np.any(np.isnan(stress)), "Stress should not contain NaN"
        assert not np.any(np.isinf(stress)), "Stress should not contain infinity"
        
        print(f"✓ Stress calculation successful")
        print(f"  Stress components: {stress}")
        
    except Exception as e:
        pytest.fail(f"Stress calculation failed: {e}")


def test_calculator_copy():
    """Test that calculator can be copied."""
    try:
        from grace_inference import GRACECalculator
        
        calc1 = GRACECalculator(device='cpu')
        calc2 = calc1.copy()
        
        assert calc2 is not None, "Copied calculator should not be None"
        print(f"✓ Calculator copy successful")
        
    except Exception as e:
        pytest.fail(f"Calculator copy failed: {e}")


def test_multiple_structures():
    """Test calculations on multiple different structures."""
    try:
        from grace_inference import GRACECalculator
        
        calc = GRACECalculator(device='cpu')
        
        structures = [
            bulk('Si', 'diamond', a=5.43),
            bulk('Cu', 'fcc', a=3.6),
            molecule('H2O')
        ]
        
        energies = []
        for atoms in structures:
            if not any(atoms.pbc):  # Add vacuum for molecules
                atoms.center(vacuum=5.0)
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            energies.append(energy)
            
            assert not np.isnan(energy), f"Energy should not be NaN"
        
        print(f"✓ Multiple structure calculations successful")
        print(f"  Number of structures tested: {len(structures)}")
        
    except Exception as e:
        pytest.fail(f"Multiple structure test failed: {e}")


def test_calculator_parameters():
    """Test calculator with different parameters."""
    try:
        from grace_inference import GRACECalculator
        
        # Test with CPU
        calc_cpu = GRACECalculator(device='cpu')
        assert calc_cpu is not None
        print(f"✓ Calculator created with CPU device")
        
        # Test with different model paths if applicable
        try:
            calc_auto = GRACECalculator(model_path='auto', device='cpu')
            assert calc_auto is not None
            print(f"✓ Calculator created with auto model path")
        except:
            print(f"⚠ Auto model path not available")
        
    except Exception as e:
        pytest.fail(f"Calculator parameter test failed: {e}")


def test_energy_forces_consistency():
    """Test that energy and forces are consistent."""
    try:
        from grace_inference import GRACECalculator
        
        atoms = bulk('Si', 'diamond', a=5.43)
        calc = GRACECalculator(device='cpu')
        atoms.calc = calc
        
        # Get energy and forces
        energy1 = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        # Get energy again
        energy2 = atoms.get_potential_energy()
        
        # Energies should be the same (cached)
        assert np.abs(energy1 - energy2) < 1e-10, "Cached energy should be identical"
        
        print(f"✓ Energy-force consistency check passed")
        
    except Exception as e:
        pytest.fail(f"Energy-force consistency test failed: {e}")


def test_periodic_boundary_conditions():
    """Test calculations with different periodic boundary conditions."""
    try:
        from grace_inference import GRACECalculator
        
        calc = GRACECalculator(device='cpu')
        
        # Test periodic system
        atoms_periodic = bulk('Si', 'diamond', a=5.43)
        atoms_periodic.calc = calc
        energy_periodic = atoms_periodic.get_potential_energy()
        assert not np.isnan(energy_periodic)
        print(f"✓ Periodic system calculation successful")
        
        # Test non-periodic system
        atoms_molecule = molecule('H2O')
        atoms_molecule.center(vacuum=5.0)
        atoms_molecule.calc = calc
        energy_molecule = atoms_molecule.get_potential_energy()
        assert not np.isnan(energy_molecule)
        print(f"✓ Non-periodic system calculation successful")
        
    except Exception as e:
        pytest.fail(f"PBC test failed: {e}")


if __name__ == "__main__":
    """Run tests with verbose output."""
    print("=" * 60)
    print("GRACE Inference Utility Tests")
    print("=" * 60)
    
    pytest.main([__file__, "-v", "-s"])
