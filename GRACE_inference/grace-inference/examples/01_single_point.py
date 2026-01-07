"""
Example 01: Single Point Energy Calculation with GRACE

This example demonstrates how to perform a single point energy and force
calculation on a molecular structure using the GRACE model.
"""

from ase import Atoms
from ase.build import bulk
from grace_inference import GRACECalculator


def main():
    """Perform single point energy calculation."""
    
    # Create a simple structure (Silicon diamond)
    atoms = bulk('Si', 'diamond', a=5.43)
    
    # Initialize GRACE calculator
    # You can specify a custom model path if needed
    calc = GRACECalculator(
        model_path='auto',  # 'auto' will download the default model
        device='cpu'  # Use 'cuda' for GPU
    )
    
    # Attach calculator to atoms
    atoms.calc = calc
    
    # Calculate energy and forces
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    
    # Print results
    print("=" * 60)
    print("GRACE Single Point Calculation Results")
    print("=" * 60)
    print(f"Structure: Silicon (diamond)")
    print(f"Number of atoms: {len(atoms)}")
    print(f"\nEnergy: {energy:.6f} eV")
    print(f"\nForces (eV/Å):")
    for i, force in enumerate(forces):
        print(f"  Atom {i}: [{force[0]:8.4f}, {force[1]:8.4f}, {force[2]:8.4f}]")
    print(f"\nStress (eV/Å³):")
    print(f"  {stress}")
    print("=" * 60)


if __name__ == "__main__":
    main()
