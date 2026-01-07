#!/usr/bin/env python
"""
Example 02: Structure Optimization

Demonstrates how to use MatterSimInference for structure optimization.
"""

from ase.build import bulk
from mattersim_inference import MatterSimInference


def main():
    """Structure optimization example."""
    print("=" * 60)
    print("MatterSim Inference - Example 02: Structure Optimization")
    print("=" * 60)
    
    # 1. Initialize model
    print("\n1. Initializing MatterSim model...")
    calc = MatterSimInference(
        model_name="MatterSim-v1-5M",
        device="auto"
    )
    
    # 2. Create test structure (intentionally using incorrect lattice constant)
    print("\n2. Creating test structure (distorted Cu bulk)...")
    atoms = bulk("Cu", "fcc", a=3.8)  # Slightly larger than equilibrium value
    print(f"   Number of atoms: {len(atoms)}")
    print(f"   Initial volume: {atoms.get_volume():.3f} Å³")
    
    # 3. Optimize atomic positions only
    print("\n3. Optimizing atomic positions only...")
    result1 = calc.optimize(
        atoms,
        fmax=0.01,
        optimize_cell=False,
        max_steps=100
    )
    print(f"   Converged: {result1['converged']}")
    print(f"   Steps: {result1['steps']}")
    print(f"   Energy change: {result1['energy_change']:.6f} eV")
    
    # 4. Optimize cell simultaneously
    print("\n4. Optimizing cell simultaneously...")
    atoms2 = bulk("Cu", "fcc", a=3.8)
    result2 = calc.optimize(
        atoms2,
        fmax=0.01,
        optimize_cell=True,
        max_steps=200
    )
    print(f"   Converged: {result2['converged']}")
    print(f"   Steps: {result2['steps']}")
    print(f"   Initial energy: {result2['initial_energy']:.6f} eV")
    print(f"   Final energy: {result2['final_energy']:.6f} eV")
    print(f"   Final volume: {result2['atoms'].get_volume():.3f} Å³")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
