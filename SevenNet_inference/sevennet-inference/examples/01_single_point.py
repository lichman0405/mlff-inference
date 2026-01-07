"""
Example 01: Single-Point Energy and Force Calculation with SevenNet

This example demonstrates how to:
1. Load a SevenNet model
2. Perform single-point calculations on a structure
3. Extract energy and forces from the results

Author: SevenNet Inference Package
Date: 2026-01-07
"""

import numpy as np
from ase import Atoms
from ase.build import bulk

try:
    from sevennet_inference import SevenNetCalculator
    from sevennet_inference.utils import setup_model, print_results
except ImportError:
    print("Error: sevennet_inference package not found.")
    print("Please install it first using: pip install -e .")
    exit(1)


def single_point_calculation():
    """
    Perform a single-point calculation on a silicon crystal structure.
    """
    print("="*60)
    print("SevenNet Single-Point Calculation Example")
    print("="*60)
    
    # Step 1: Create a test structure (Silicon crystal)
    print("\n[1/3] Creating test structure...")
    atoms = bulk('Si', 'diamond', a=5.43)
    print(f"Created Si structure with {len(atoms)} atoms")
    print(f"Cell parameters: {atoms.cell.cellpar()}")
    
    # Step 2: Initialize SevenNet calculator
    print("\n[2/3] Loading SevenNet model...")
    try:
        # Initialize calculator with default SevenNet model
        calc = SevenNetCalculator(
            model_path='7net-0',  # Use pre-trained SevenNet-0 model
            device='cpu'  # Use 'cuda' for GPU
        )
        atoms.calc = calc
        print("SevenNet calculator initialized successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure SevenNet models are properly installed")
        return
    
    # Step 3: Perform calculation
    print("\n[3/3] Performing single-point calculation...")
    try:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress()
        
        # Print results
        print("\n" + "="*60)
        print("CALCULATION RESULTS")
        print("="*60)
        print(f"\nTotal Energy: {energy:.6f} eV")
        print(f"Energy per atom: {energy/len(atoms):.6f} eV/atom")
        print(f"\nForces (eV/Å):")
        for i, force in enumerate(forces):
            print(f"  Atom {i}: [{force[0]:8.4f}, {force[1]:8.4f}, {force[2]:8.4f}]")
        print(f"\nMax force: {np.max(np.abs(forces)):.6f} eV/Å")
        print(f"RMS force: {np.sqrt(np.mean(forces**2)):.6f} eV/Å")
        print(f"\nStress (eV/Å³):")
        print(f"  {stress}")
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"Error during calculation: {e}")
        return
    
    print("\n✓ Single-point calculation completed successfully!")


if __name__ == "__main__":
    single_point_calculation()
