"""
Example 02: Structure Optimization with GRACE

This example demonstrates how to perform structure optimization
using the GRACE model as the calculator.
"""

from ase import Atoms
from ase.build import molecule
from ase.optimize import BFGS
from grace_inference import GRACECalculator


def main():
    """Optimize molecular structure using GRACE."""
    
    # Create a water molecule with slightly distorted geometry
    atoms = molecule('H2O')
    atoms.rattle(stdev=0.1)  # Add some random displacement
    
    print("=" * 60)
    print("GRACE Structure Optimization")
    print("=" * 60)
    print(f"Initial positions:")
    print(atoms.get_positions())
    
    # Initialize GRACE calculator
    calc = GRACECalculator(
        model_path='auto',
        device='cpu'
    )
    atoms.calc = calc
    
    # Get initial energy
    initial_energy = atoms.get_potential_energy()
    print(f"\nInitial energy: {initial_energy:.6f} eV")
    
    # Set up optimizer
    optimizer = BFGS(atoms, trajectory='h2o_opt.traj')
    
    # Run optimization
    print("\nStarting optimization...")
    optimizer.run(fmax=0.01)  # Converge when max force < 0.01 eV/Å
    
    # Get final energy
    final_energy = atoms.get_potential_energy()
    
    # Print results
    print("\n" + "=" * 60)
    print("Optimization Results")
    print("=" * 60)
    print(f"Final positions:")
    print(atoms.get_positions())
    print(f"\nFinal energy: {final_energy:.6f} eV")
    print(f"Energy change: {final_energy - initial_energy:.6f} eV")
    print(f"\nOptimization trajectory saved to: h2o_opt.traj")
    print("=" * 60)


if __name__ == "__main__":
    main()
