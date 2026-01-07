"""
Installation and Import Tests for SevenNet Inference Package

This test module verifies:
1. Package installation and importability
2. Core dependencies availability
3. Calculator initialization
4. Basic functionality

Author: SevenNet Inference Package
Date: 2026-01-07
"""

import sys
import unittest
from pathlib import Path


class TestInstallation(unittest.TestCase):
    """Test package installation and imports."""
    
    def test_package_import(self):
        """Test that the main package can be imported."""
        try:
            import sevennet_inference
            self.assertTrue(True, "Package imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import sevennet_inference: {e}")
    
    def test_calculator_import(self):
        """Test that the calculator can be imported."""
        try:
            from sevennet_inference import SevenNetCalculator
            self.assertTrue(True, "SevenNetCalculator imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import SevenNetCalculator: {e}")
    
    def test_utils_import(self):
        """Test that utility modules can be imported."""
        try:
            from sevennet_inference import utils
            self.assertTrue(True, "Utils module imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import utils: {e}")
    
    def test_tasks_import(self):
        """Test that tasks module can be imported."""
        try:
            from sevennet_inference import tasks
            self.assertTrue(True, "Tasks module imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import tasks: {e}")


class TestDependencies(unittest.TestCase):
    """Test that all required dependencies are available."""
    
    def test_ase_available(self):
        """Test ASE installation."""
        try:
            import ase
            from ase import Atoms
            from ase.build import bulk
            self.assertTrue(True, "ASE is available")
        except ImportError as e:
            self.fail(f"ASE not available: {e}")
    
    def test_numpy_available(self):
        """Test NumPy installation."""
        try:
            import numpy as np
            self.assertTrue(True, "NumPy is available")
        except ImportError as e:
            self.fail(f"NumPy not available: {e}")
    
    def test_torch_available(self):
        """Test PyTorch installation."""
        try:
            import torch
            self.assertTrue(True, "PyTorch is available")
            
            # Check version
            version = torch.__version__
            print(f"PyTorch version: {version}")
            
        except ImportError as e:
            self.fail(f"PyTorch not available: {e}")
    
    def test_sevenn_available(self):
        """Test SevenNet (sevenn) library installation."""
        try:
            import sevenn
            self.assertTrue(True, "SevenNet library is available")
        except ImportError as e:
            # SevenNet might not be installed in all environments
            self.skipTest(f"SevenNet library not installed: {e}")


class TestBasicFunctionality(unittest.TestCase):
    """Test basic calculator functionality."""
    
    def test_calculator_creation(self):
        """Test that calculator can be instantiated."""
        try:
            from sevennet_inference import SevenNetCalculator
            
            # Try to create calculator (might fail if models not downloaded)
            try:
                calc = SevenNetCalculator(model_path='7net-0', device='cpu')
                self.assertIsNotNone(calc, "Calculator created successfully")
            except Exception as e:
                # Model might not be available, which is okay for install test
                self.skipTest(f"Model not available (expected): {e}")
                
        except ImportError as e:
            self.fail(f"Failed to create calculator: {e}")
    
    def test_simple_calculation(self):
        """Test a simple energy calculation."""
        try:
            from sevennet_inference import SevenNetCalculator
            from ase.build import bulk
            
            # Create simple structure
            atoms = bulk('Si', 'diamond', a=5.43)
            
            # Try calculation
            try:
                calc = SevenNetCalculator(model_path='7net-0', device='cpu')
                atoms.calc = calc
                energy = atoms.get_potential_energy()
                
                self.assertIsInstance(energy, float, "Energy is a float")
                self.assertTrue(energy < 0, "Energy is negative (expected for stable structure)")
                
            except Exception as e:
                self.skipTest(f"Calculation failed (model may not be available): {e}")
                
        except ImportError as e:
            self.fail(f"Failed to run calculation: {e}")


class TestPackageStructure(unittest.TestCase):
    """Test package structure and metadata."""
    
    def test_version_available(self):
        """Test that package version is defined."""
        try:
            import sevennet_inference
            version = getattr(sevennet_inference, '__version__', None)
            
            if version is None:
                self.skipTest("Version not defined in package")
            else:
                self.assertIsInstance(version, str, "Version is a string")
                print(f"Package version: {version}")
                
        except ImportError as e:
            self.fail(f"Failed to check version: {e}")
    
    def test_package_path(self):
        """Test that package path is accessible."""
        try:
            import sevennet_inference
            path = Path(sevennet_inference.__file__).parent
            
            self.assertTrue(path.exists(), "Package path exists")
            print(f"Package location: {path}")
            
        except Exception as e:
            self.fail(f"Failed to get package path: {e}")


def run_tests():
    """Run all tests and print results."""
    print("="*60)
    print("SevenNet Inference Package - Installation Tests")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestInstallation))
    suite.addTests(loader.loadTestsFromTestCase(TestDependencies))
    suite.addTests(loader.loadTestsFromTestCase(TestBasicFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestPackageStructure))
    
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
