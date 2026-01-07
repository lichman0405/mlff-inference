"""
Example 04: Phonon Calculation with GRACE

This example demonstrates how to calculate phonon properties
using the GRACE model with finite displacement method.
"""

from ase.build import bulk
from ase.phonons import Phonons
from grace_inference import GRACECalculator
import numpy as np


def main():
    """Calculate phonon properties using GRACE."""
    
    # Create primitive cell
    atoms = bulk('Si', 'diamond', a=5.43)
    
    print("=" * 60)
    print("GRACE Phonon Calculation")
    print("=" * 60)
    print(f"Structure: Silicon (diamond)")
    print(f"Number of atoms: {len(atoms)}")
    
    # Initialize GRACE calculator
    calc = GRACECalculator(
        model_path='auto',
        device='cpu'
    )
    atoms.calc = calc
    
    # Set up phonon calculation
    # Use 5x5x5 supercell and calculate with 0.05 Å displacement
    N = 5  # Supercell size
    ph = Phonons(atoms, calc, supercell=(N, N, N), delta=0.05)
    
    print(f"\nSupercell size: {N}x{N}x{N}")
    print("Calculating forces for displaced atoms...")
    
    # Run the phonon calculation
    ph.run()
    
    # Read forces and create force constants
    print("Reading forces and building force constant matrix...")
    ph.read(acoustic=True)
    
    # Calculate phonon band structure
    print("\nCalculating phonon band structure...")
    path = atoms.cell.bandpath('GXWLG', npoints=50)
    bs = ph.get_band_structure(path)
    
    # Get phonon DOS
    print("Calculating phonon density of states...")
    dos = ph.get_dos(kpts=(20, 20, 20)).sample_grid(npts=100, width=1e-3)
    
    print("\n" + "=" * 60)
    print("Phonon Calculation Results")
    print("=" * 60)
    
    # Print some phonon frequencies at Gamma point
    omega_kl = ph.band_structure([0, 0, 0])[0]
    print(f"\nPhonon frequencies at Gamma point (THz):")
    for i, freq in enumerate(omega_kl):
        print(f"  Mode {i}: {freq * 1000:.4f}")  # Convert to THz
    
    print(f"\nBand structure can be plotted using:")
    print(f"  bs.plot()")
    print(f"\nDOS can be plotted using:")
    print(f"  dos.plot()")
    
    # Save results
    ph.write('phonon_cache.json')
    print(f"\nPhonon data saved to: phonon_cache.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
