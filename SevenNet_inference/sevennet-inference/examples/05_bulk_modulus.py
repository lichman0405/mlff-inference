"""
Example 05: Bulk Modulus Calculation with SevenNet

This example demonstrates how to:
1. Calculate bulk modulus using equation of state fitting
2. Optimize structure at different volumes
3. Fit Birch-Murnaghan equation of state

Author: SevenNet Inference Package
Date: 2026-01-07
"""

import numpy as np
from ase.build import bulk
from ase.eos import EquationOfState
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

try:
    from sevennet_inference import SevenNetCalculator
except ImportError:
    print("Error: sevennet_inference package not found.")
    print("Please install it first using: pip install -e .")
    exit(1)


def calculate_bulk_modulus():
    """
    Calculate bulk modulus by fitting equation of state.
    """
    print("="*60)
    print("Bulk Modulus Calculation with SevenNet")
    print("="*60)
    
    # Step 1: Create reference structure
    print("\n[1/4] Creating reference structure...")
    a0 = 5.43  # Initial lattice parameter for Si
    atoms = bulk('Si', 'diamond', a=a0)
    print(f"Material: Silicon (diamond structure)")
    print(f"Initial lattice parameter: {a0} Å")
    
    # Step 2: Set up calculator
    print("\n[2/4] Initializing SevenNet calculator...")
    calc = SevenNetCalculator(model_path='7net-0', device='cpu')
    
    # Step 3: Calculate energies at different volumes
    print("\n[3/4] Calculating energies at different volumes...")
    
    # Define volume scaling factors
    scale_factors = np.linspace(0.92, 1.08, 13)  # ±8% volume change
    volumes = []
    energies = []
    
    print(f"\nPerforming {len(scale_factors)} calculations...")
    print(f"Volume range: {scale_factors[0]**3:.3f} to {scale_factors[-1]**3:.3f} (relative)")
    
    for i, scale in enumerate(scale_factors):
        # Scale the cell
        atoms_scaled = atoms.copy()
        atoms_scaled.set_cell(atoms.cell * scale, scale_atoms=True)
        atoms_scaled.calc = calc
        
        # Calculate energy
        energy = atoms_scaled.get_potential_energy()
        volume = atoms_scaled.get_volume()
        
        volumes.append(volume)
        energies.append(energy)
        
        print(f"  {i+1:2d}/{len(scale_factors)}: "
              f"scale={scale:.4f}, V={volume:6.2f} Ų, E={energy:9.5f} eV")
    
    # Step 4: Fit equation of state
    print("\n[4/4] Fitting equation of state...")
    
    eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
    v0, e0, B = eos.fit()
    
    # Convert bulk modulus from eV/Ų to GPa
    B_GPa = B * 160.21766208  # Conversion factor
    
    # Calculate equilibrium lattice parameter
    a_eq = a0 * (v0 / atoms.get_volume())**(1/3)
    
    # Print results
    print("\n" + "="*60)
    print("BULK MODULUS RESULTS")
    print("="*60)
    print(f"\nEquation of State: Birch-Murnaghan")
    print(f"\nEquilibrium properties:")
    print(f"  Lattice parameter: {a_eq:.4f} Å")
    print(f"  Volume: {v0:.3f} Ų")
    print(f"  Energy: {e0:.6f} eV")
    print(f"  Bulk modulus: {B_GPa:.2f} GPa")
    print(f"\nReference values (DFT):")
    print(f"  Lattice parameter: ~5.43 Å")
    print(f"  Bulk modulus: ~98 GPa")
    print(f"\nDeviation:")
    print(f"  Lattice: {((a_eq/5.43 - 1)*100):.2f}%")
    print(f"  Bulk modulus: {((B_GPa/98 - 1)*100):.2f}%")
    print("="*60)
    
    # Step 5: Plot results
    print("\n[5/5] Generating plots...")
    plot_eos(volumes, energies, eos)
    
    print("\n✓ Bulk modulus calculation completed successfully!")
    print("\nOutput file: bulk_modulus.png")
    
    return v0, e0, B_GPa


def plot_eos(volumes, energies, eos):
    """
    Plot equation of state fitting results.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot calculated points
    ax.plot(volumes, energies, 'ro', markersize=8, label='Calculated', zorder=3)
    
    # Plot fitted curve
    v_fit = np.linspace(min(volumes)*0.98, max(volumes)*1.02, 100)
    e_fit = eos.func(v_fit, *[eos.v0, eos.e0, eos.B])
    ax.plot(v_fit, e_fit, 'b-', linewidth=2, label='EOS fit', zorder=2)
    
    # Mark equilibrium point
    ax.plot(eos.v0, eos.e0, 'g*', markersize=15, 
            label=f'Equilibrium\nV₀={eos.v0:.2f} Ų', zorder=4)
    
    # Formatting
    ax.set_xlabel('Volume (Ų)', fontsize=12)
    ax.set_ylabel('Energy (eV)', fontsize=12)
    ax.set_title('Equation of State - Silicon', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    # Add text box with results
    B_GPa = eos.B * 160.21766208
    textstr = f'Birch-Murnaghan EOS\n'
    textstr += f'V₀ = {eos.v0:.2f} Ų\n'
    textstr += f'E₀ = {eos.e0:.4f} eV\n'
    textstr += f'B₀ = {B_GPa:.1f} GPa'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('bulk_modulus.png', dpi=300, bbox_inches='tight')
    print("  Saved: bulk_modulus.png")
    plt.close()


def quick_bulk_modulus():
    """
    Quick bulk modulus calculation with fewer points.
    """
    print("\n\n" + "="*60)
    print("Quick Bulk Modulus Calculation")
    print("="*60)
    
    atoms = bulk('Si', 'diamond', a=5.43)
    calc = SevenNetCalculator(model_path='7net-0', device='cpu')
    
    # Fewer points for quick calculation
    scale_factors = np.linspace(0.95, 1.05, 7)
    volumes = []
    energies = []
    
    print("\nCalculating with 7 volume points...")
    for scale in scale_factors:
        atoms_scaled = atoms.copy()
        atoms_scaled.set_cell(atoms.cell * scale, scale_atoms=True)
        atoms_scaled.calc = calc
        
        energy = atoms_scaled.get_potential_energy()
        volume = atoms_scaled.get_volume()
        
        volumes.append(volume)
        energies.append(energy)
    
    # Fit EOS
    eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
    v0, e0, B = eos.fit()
    B_GPa = B * 160.21766208
    
    print(f"\nQuick result: B₀ = {B_GPa:.2f} GPa")
    print("(Use more points for better accuracy)")


if __name__ == "__main__":
    # Full calculation
    calculate_bulk_modulus()
    
    # Quick calculation
    quick_bulk_modulus()
