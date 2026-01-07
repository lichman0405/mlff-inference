#!/usr/bin/env python
"""
Example 05: Bulk Modulus Calculation

Demonstrates how to use MatterSimInference to calculate bulk modulus.
"""

from ase.build import bulk
from mattersim_inference import MatterSimInference


def main():
    """Bulk modulus calculation example."""
    print("=" * 60)
    print("MatterSim Inference - Example 05: Bulk Modulus Calculation")
    print("=" * 60)
    
    # 1. Initialize model
    print("\n1. Initializing MatterSim model...")
    calc = MatterSimInference(
        model_name="MatterSim-v1-5M",
        device="auto"
    )
    
    # 2. Create and optimize structure
    print("\n2. Creating and optimizing copper crystal structure...")
    atoms = bulk("Cu", "fcc", a=3.6)
    opt_result = calc.optimize(atoms, fmax=0.01, optimize_cell=True)
    atoms = opt_result['atoms']
    print(f"   Optimized volume: {atoms.get_volume():.4f} Å³")
    
    # 3. Calculate bulk modulus
    print("\n3. Calculating bulk modulus...")
    print("   Strain range: ±5%")
    print("   Number of points: 11")
    
    result = calc.bulk_modulus(
        atoms,
        strain_range=0.05,
        npoints=11,
        eos="birchmurnaghan"
    )
    
    # 4. Display results
    print("\n4. Calculation results:")
    print(f"   Bulk modulus: {result['bulk_modulus']:.2f} GPa")
    print(f"   Equilibrium volume: {result['v0']:.4f} Å³")
    print(f"   Equilibrium energy: {result['e0']:.6f} eV")
    print(f"   Equation of state: {result['eos']}")
    
    # 5. Compare with experimental value
    print("\n5. Comparison with experimental value:")
    exp_bulk_modulus = 140  # Cu experimental value ~140 GPa
    diff = abs(result['bulk_modulus'] - exp_bulk_modulus)
    error_percent = diff / exp_bulk_modulus * 100
    print(f"   Experimental value: ~{exp_bulk_modulus} GPa")
    print(f"   Calculated value: {result['bulk_modulus']:.2f} GPa")
    print(f"   Error: {error_percent:.1f}%")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
