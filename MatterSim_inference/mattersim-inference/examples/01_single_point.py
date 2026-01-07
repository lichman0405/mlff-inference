#!/usr/bin/env python
"""
Example 01: Single Point Energy Calculation

Demonstrates how to use MatterSimInference for single point calculations.
"""

from ase.build import bulk
from mattersim_inference import MatterSimInference


def main():
    """Single point calculation example."""
    print("=" * 60)
    print("MatterSim Inference - Example 01: Single Point Energy Calculation")
    print("=" * 60)
    
    # 1. Initialize model
    print("\n1. Initializing MatterSim model...")
    calc = MatterSimInference(
        model_name="MatterSim-v1-5M",
        device="auto"
    )
    print(f"   Model: {calc.model_name}")
    print(f"   Device: {calc.device}")
    
    # 2. Create test structure (copper crystal)
    print("\n2. Creating test structure (Cu bulk)...")
    atoms = bulk("Cu", "fcc", a=3.6)
    print(f"   Number of atoms: {len(atoms)}")
    print(f"   Chemical formula: {atoms.get_chemical_formula()}")
    
    # 3. Single point calculation
    print("\n3. Performing single point calculation...")
    result = calc.single_point(atoms)
    
    # 4. Output results
    print("\n4. Calculation results:")
    print(f"   Total energy: {result['energy']:.6f} eV")
    print(f"   Energy per atom: {result['energy_per_atom']:.6f} eV/atom")
    print(f"   Max force: {result['max_force']:.6f} eV/Å")
    print(f"   RMS force: {result['rms_force']:.6f} eV/Å")
    print(f"   Pressure: {result['pressure']:.4f} GPa")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
