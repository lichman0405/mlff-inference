"""Test installation and basic functionality"""

def test_import():
    """Test that package can be imported"""
    print("Testing package import...")
    try:
        import mace_inference
        print(f"✓ mace_inference version {mace_inference.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import mace_inference: {e}")
        return False
    return True


def test_device_utils():
    """Test device utilities"""
    print("\nTesting device utilities...")
    try:
        from mace_inference.utils import get_device
        device = get_device("auto")
        print(f"✓ Device auto-detection: {device}")
    except Exception as e:
        print(f"✗ Device utils failed: {e}")
        return False
    return True


def test_io_utils():
    """Test I/O utilities"""
    print("\nTesting I/O utilities...")
    try:
        from ase.build import bulk
        from mace_inference.utils import create_supercell
        
        atoms = bulk('Cu', 'fcc', a=3.6)
        supercell = create_supercell(atoms, 2)
        print(f"✓ Created supercell: {len(supercell)} atoms")
    except Exception as e:
        print(f"✗ I/O utils failed: {e}")
        return False
    return True


def test_core_init():
    """Test core MACEInference initialization"""
    print("\nTesting MACEInference initialization...")
    try:
        from mace_inference import MACEInference
        calc = MACEInference(model="medium", device="cpu")
        print(f"✓ Created MACEInference: {calc}")
    except ImportError as e:
        print(f"✗ mace-torch not installed: {e}")
        print("  → Install with: pip install mace-torch")
        return False
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False
    return True


def test_cli():
    """Test CLI availability"""
    print("\nTesting CLI...")
    try:
        import subprocess
        result = subprocess.run(
            ["mace-infer", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✓ CLI available: mace-infer")
        else:
            print(f"✗ CLI failed with code {result.returncode}")
            return False
    except FileNotFoundError:
        print("✗ mace-infer command not found")
        print("  → Try reinstalling or check PATH")
        return False
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False
    return True


def main():
    """Run all tests"""
    print("="*60)
    print("MACE Inference Installation Test")
    print("="*60)
    
    tests = [
        ("Package Import", test_import),
        ("Device Utils", test_device_utils),
        ("I/O Utils", test_io_utils),
        ("Core Initialization", test_core_init),
        ("CLI", test_cli),
    ]
    
    results = []
    for name, test_func in tests:
        passed = test_func()
        results.append((name, passed))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:<10} {name}")
    
    print("="*60)
    print(f"Passed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Installation successful.")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed.")
        print("\nNext steps:")
        print("1. Install missing dependencies:")
        print("   pip install mace-torch ase phonopy click")
        print("2. Check the error messages above")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
