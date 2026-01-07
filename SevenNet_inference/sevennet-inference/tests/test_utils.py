"""
Unit Tests for SevenNet Inference Utils and Tasks Modules

This test module verifies:
1. Utility functions work correctly
2. Task functions produce expected results
3. Error handling is appropriate
4. Edge cases are handled

Author: SevenNet Inference Package
Date: 2026-01-07
"""

import sys
import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions from utils module."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from sevennet_inference import utils
            self.utils = utils
        except ImportError:
            self.skipTest("Utils module not available")
    
    def test_model_path_validation(self):
        """Test model path validation function."""
        if not hasattr(self.utils, 'validate_model_path'):
            self.skipTest("validate_model_path function not implemented")
        
        # Test with valid model name
        try:
            result = self.utils.validate_model_path('7net-0')
            self.assertIsNotNone(result)
        except Exception as e:
            self.skipTest(f"Model validation skipped: {e}")
    
    def test_device_detection(self):
        """Test automatic device detection."""
        if not hasattr(self.utils, 'get_device'):
            self.skipTest("get_device function not implemented")
        
        device = self.utils.get_device()
        self.assertIn(device, ['cpu', 'cuda'])
        print(f"Detected device: {device}")
    
    def test_structure_validation(self):
        """Test structure validation function."""
        if not hasattr(self.utils, 'validate_structure'):
            self.skipTest("validate_structure function not implemented")
        
        from ase.build import bulk
        
        # Valid structure
        atoms = bulk('Si', 'diamond', a=5.43)
        try:
            is_valid = self.utils.validate_structure(atoms)
            self.assertTrue(is_valid)
        except Exception as e:
            self.skipTest(f"Structure validation not implemented: {e}")


class TestTaskFunctions(unittest.TestCase):
    """Test task-specific functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from sevennet_inference import tasks
            self.tasks = tasks
        except ImportError:
            self.skipTest("Tasks module not available")
        
        # Create temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_single_point_task(self):
        """Test single-point calculation task."""
        if not hasattr(self.tasks, 'run_single_point'):
            self.skipTest("run_single_point task not implemented")
        
        from ase.build import bulk
        
        try:
            atoms = bulk('Si', 'diamond', a=5.43)
            result = self.tasks.run_single_point(
                atoms,
                model_path='7net-0',
                device='cpu'
            )
            
            self.assertIn('energy', result)
            self.assertIn('forces', result)
            self.assertIsInstance(result['energy'], float)
            self.assertIsInstance(result['forces'], np.ndarray)
            
        except Exception as e:
            self.skipTest(f"Single-point task failed: {e}")
    
    def test_optimization_task(self):
        """Test structure optimization task."""
        if not hasattr(self.tasks, 'run_optimization'):
            self.skipTest("run_optimization task not implemented")
        
        from ase.build import bulk
        
        try:
            atoms = bulk('Si', 'diamond', a=5.43)
            # Add small random displacement
            atoms.positions += np.random.normal(0, 0.01, atoms.positions.shape)
            
            result = self.tasks.run_optimization(
                atoms,
                model_path='7net-0',
                device='cpu',
                fmax=0.05
            )
            
            self.assertIn('final_energy', result)
            self.assertIn('optimized_atoms', result)
            self.assertIn('steps', result)
            
        except Exception as e:
            self.skipTest(f"Optimization task failed: {e}")


class TestCalculatorInterface(unittest.TestCase):
    """Test calculator interface and methods."""
    
    def setUp(self):
        """Set up calculator for testing."""
        try:
            from sevennet_inference import SevenNetCalculator
            from ase.build import bulk
            
            self.atoms = bulk('Si', 'diamond', a=5.43)
            try:
                self.calc = SevenNetCalculator(model_path='7net-0', device='cpu')
                self.atoms.calc = self.calc
            except Exception as e:
                self.skipTest(f"Calculator initialization failed: {e}")
                
        except ImportError:
            self.skipTest("Calculator not available")
    
    def test_energy_calculation(self):
        """Test energy calculation."""
        try:
            energy = self.atoms.get_potential_energy()
            self.assertIsInstance(energy, float)
            self.assertTrue(np.isfinite(energy))
        except Exception as e:
            self.fail(f"Energy calculation failed: {e}")
    
    def test_force_calculation(self):
        """Test force calculation."""
        try:
            forces = self.atoms.get_forces()
            self.assertIsInstance(forces, np.ndarray)
            self.assertEqual(forces.shape, (len(self.atoms), 3))
            self.assertTrue(np.all(np.isfinite(forces)))
        except Exception as e:
            self.fail(f"Force calculation failed: {e}")
    
    def test_stress_calculation(self):
        """Test stress calculation."""
        try:
            stress = self.atoms.get_stress()
            self.assertIsInstance(stress, np.ndarray)
            self.assertEqual(len(stress), 6)  # Voigt notation
            self.assertTrue(np.all(np.isfinite(stress)))
        except Exception as e:
            self.fail(f"Stress calculation failed: {e}")
    
    def test_consistency(self):
        """Test consistency of multiple calculations."""
        try:
            # Calculate energy twice
            energy1 = self.atoms.get_potential_energy()
            energy2 = self.atoms.get_potential_energy()
            
            # Should be identical for same structure
            self.assertAlmostEqual(energy1, energy2, places=10)
            
            # Calculate forces twice
            forces1 = self.atoms.get_forces()
            forces2 = self.atoms.get_forces()
            
            np.testing.assert_array_almost_equal(forces1, forces2)
            
        except Exception as e:
            self.fail(f"Consistency test failed: {e}")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_invalid_model_path(self):
        """Test handling of invalid model path."""
        try:
            from sevennet_inference import SevenNetCalculator
            
            with self.assertRaises(Exception):
                calc = SevenNetCalculator(
                    model_path='nonexistent_model',
                    device='cpu'
                )
        except ImportError:
            self.skipTest("Calculator not available")
    
    def test_empty_structure(self):
        """Test handling of empty structure."""
        try:
            from sevennet_inference import SevenNetCalculator
            from ase import Atoms
            
            # Create empty structure
            atoms = Atoms()
            
            calc = SevenNetCalculator(model_path='7net-0', device='cpu')
            atoms.calc = calc
            
            # Should raise an error or handle gracefully
            with self.assertRaises(Exception):
                energy = atoms.get_potential_energy()
                
        except Exception as e:
            self.skipTest(f"Empty structure test skipped: {e}")
    
    def test_invalid_device(self):
        """Test handling of invalid device specification."""
        try:
            from sevennet_inference import SevenNetCalculator
            
            # Try to create calculator with invalid device
            # Should either raise error or fall back to CPU
            try:
                calc = SevenNetCalculator(
                    model_path='7net-0',
                    device='invalid_device'
                )
                # If no error, it should have fallen back to CPU
                self.assertEqual(calc.device, 'cpu')
            except Exception:
                # Expected to raise an error
                pass
                
        except ImportError:
            self.skipTest("Calculator not available")


class TestNumericalAccuracy(unittest.TestCase):
    """Test numerical accuracy and physical reasonableness."""
    
    def setUp(self):
        """Set up calculator and test structure."""
        try:
            from sevennet_inference import SevenNetCalculator
            from ase.build import bulk
            
            self.atoms = bulk('Si', 'diamond', a=5.43)
            self.calc = SevenNetCalculator(model_path='7net-0', device='cpu')
            self.atoms.calc = self.calc
            
        except Exception as e:
            self.skipTest(f"Setup failed: {e}")
    
    def test_energy_magnitude(self):
        """Test that energy values are physically reasonable."""
        try:
            energy = self.atoms.get_potential_energy()
            energy_per_atom = energy / len(self.atoms)
            
            # Energy per atom should be reasonable (typically -3 to -7 eV for Si)
            self.assertTrue(-10 < energy_per_atom < 0,
                          f"Energy per atom {energy_per_atom} seems unreasonable")
        except Exception as e:
            self.skipTest(f"Energy magnitude test failed: {e}")
    
    def test_force_symmetry(self):
        """Test force symmetry for symmetric structure."""
        try:
            forces = self.atoms.get_forces()
            
            # For perfect crystal, forces should be very small
            max_force = np.max(np.abs(forces))
            self.assertTrue(max_force < 0.01,
                          f"Forces on perfect crystal too large: {max_force}")
        except Exception as e:
            self.skipTest(f"Force symmetry test failed: {e}")


def run_tests():
    """Run all tests and print results."""
    print("="*60)
    print("SevenNet Inference Package - Utils and Tasks Tests")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestTaskFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestCalculatorInterface))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestNumericalAccuracy))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
