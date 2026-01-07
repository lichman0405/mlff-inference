"""
Example 05: Bulk Modulus Calculation with GRACE

This example demonstrates how to calculate the bulk modulus
of a material using the GRACE model by fitting the
Birch-Murnaghan equation of state.
"""

from ase.build import bulk
from ase.eos import calculate_eos
from grace_inference import GRACECalculator
import numpy as np


def main():
    """Calculate bulk modulus using GRACE."""
    
    # Create primitive cell
    atoms = bulk('Si', 'diamond', a=5.43)
    
    print("=" * 60)
    print("GRACE Bulk Modulus Calculation")
    print("=" * 60)
    print(f"Structure: Silicon (diamond)")
    print(f"Number of atoms: {len(atoms)}")
    
    # Initialize GRACE calculator
    calc = GRACECalculator(
        model_path='auto',
        device='cpu'
    )
    atoms.calc = calc
    
    print("\nCalculating equation of state...")
    print("Scaling volumes from 0.90 to 1.10...")
    
    # Calculate EOS with volume scaling from 0.90 to 1.10
    eos = calculate_eos(atoms, npoints=11, eps=0.05, trajectory='eos.traj')
    
    print("\n" + "=" * 60)
    print("Equation of State Results")
    print("=" * 60)
    
    # Get EOS parameters
    v0 = eos.v0
    e0 = eos.e0
    B = eos.B  # Bulk modulus in eV/Å³
    
    # Convert bulk modulus to GPa
    # 1 eV/Å³ = 160.21766208 GPa
    B_GPa = B * 160.21766208
    
    print(f"\nEquilibrium volume: {v0:.4f} Å³")
    print(f"Equilibrium energy: {e0:.6f} eV")
    print(f"Bulk modulus: {B:.6f} eV/Å³")
    print(f"Bulk modulus: {B_GPa:.2f} GPa")
    
    # Print volumes and energies
    print(f"\nVolume-Energy data:")
    print(f"{'Volume (Å³)':>15} {'Energy (eV)':>15}")
    print("-" * 32)
    for v, e in zip(eos.v, eos.e):
        print(f"{v:15.4f} {e:15.6f}")
    
    print(f"\nEOS fit: {eos.eos_string}")
    print(f"Trajectory saved to: eos.traj")
    
    # You can plot the EOS
    print(f"\nTo plot the equation of state, use:")
    print(f"  eos.plot()")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
