"""
Example 02: Structure Optimization with SevenNet

This example demonstrates how to:
1. Set up a SevenNet calculator
2. Optimize atomic positions and cell parameters
3. Monitor convergence during optimization

Author: SevenNet Inference Package
Date: 2026-01-07
"""

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.optimize import BFGS, FIRE
from ase.constraints import ExpCellFilter

try:
    from sevennet_inference import SevenNetCalculator
except ImportError:
    print("Error: sevennet_inference package not found.")
    print("Please install it first using: pip install -e .")
    exit(1)


def optimize_positions_only():
    """
    Optimize atomic positions while keeping cell fixed.
    """
    print("="*60)
    print("Position Optimization Example")
    print("="*60)
    
    # Create a slightly distorted Si structure
    atoms = bulk('Si', 'diamond', a=5.43)
    atoms.positions += np.random.normal(0, 0.05, atoms.positions.shape)
    
    print(f"\nInitial structure:")
    print(f"  Number of atoms: {len(atoms)}")
    print(f"  Cell volume: {atoms.get_volume():.3f} Å³")
    
    # Set up calculator
    calc = SevenNetCalculator(model_path='7net-0', device='cpu')
    atoms.calc = calc
    
    # Get initial energy and forces
    E_initial = atoms.get_potential_energy()
    F_initial = atoms.get_forces()
    print(f"\nInitial energy: {E_initial:.6f} eV")
    print(f"Initial max force: {np.max(np.abs(F_initial)):.6f} eV/Å")
    
    # Optimize positions
    print("\nOptimizing positions...")
    optimizer = BFGS(atoms, trajectory='optimization.traj')
    optimizer.run(fmax=0.01)  # Converge when max force < 0.01 eV/Å
    
    # Get final results
    E_final = atoms.get_potential_energy()
    F_final = atoms.get_forces()
    
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Initial energy: {E_initial:.6f} eV")
    print(f"Final energy:   {E_final:.6f} eV")
    print(f"Energy change:  {E_final - E_initial:.6f} eV")
    print(f"\nInitial max force: {np.max(np.abs(F_initial)):.6f} eV/Å")
    print(f"Final max force:   {np.max(np.abs(F_final)):.6f} eV/Å")
    print(f"\nOptimization steps: {optimizer.nsteps}")
    print("="*60)


def optimize_cell_and_positions():
    """
    Optimize both cell parameters and atomic positions.
    """
    print("\n\n" + "="*60)
    print("Cell + Position Optimization Example")
    print("="*60)
    
    # Create Si structure with incorrect lattice parameter
    atoms = bulk('Si', 'diamond', a=5.2)  # Too small
    
    print(f"\nInitial structure:")
    print(f"  Lattice parameter: {atoms.cell.cellpar()[0]:.3f} Å")
    print(f"  Cell volume: {atoms.get_volume():.3f} Å³")
    
    # Set up calculator
    calc = SevenNetCalculator(model_path='7net-0', device='cpu')
    atoms.calc = calc
    
    E_initial = atoms.get_potential_energy()
    V_initial = atoms.get_volume()
    
    print(f"  Initial energy: {E_initial:.6f} eV")
    
    # Use ExpCellFilter for variable cell optimization
    print("\nOptimizing cell and positions...")
    ecf = ExpCellFilter(atoms)
    optimizer = BFGS(ecf, trajectory='cell_opt.traj')
    optimizer.run(fmax=0.01)
    
    # Get final results
    E_final = atoms.get_potential_energy()
    V_final = atoms.get_volume()
    a_final = atoms.cell.cellpar()[0]
    
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Initial lattice parameter: 5.200 Å")
    print(f"Final lattice parameter:   {a_final:.3f} Å")
    print(f"Expected (DFT):            ~5.43 Å")
    print(f"\nInitial volume: {V_initial:.3f} Å³")
    print(f"Final volume:   {V_final:.3f} Å³")
    print(f"Volume change:  {((V_final/V_initial - 1)*100):.2f}%")
    print(f"\nInitial energy: {E_initial:.6f} eV")
    print(f"Final energy:   {E_final:.6f} eV")
    print(f"Energy change:  {E_final - E_initial:.6f} eV")
    print(f"\nOptimization steps: {optimizer.nsteps}")
    print("="*60)


if __name__ == "__main__":
    # Run position optimization
    optimize_positions_only()
    
    # Run cell + position optimization
    optimize_cell_and_positions()
    
    print("\n✓ Structure optimization examples completed successfully!")
