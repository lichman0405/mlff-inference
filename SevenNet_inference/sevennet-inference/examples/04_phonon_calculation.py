"""
Example 04: Phonon Calculation with SevenNet

This example demonstrates how to:
1. Calculate phonon band structure using SevenNet
2. Compute phonon density of states
3. Analyze thermodynamic properties from phonons

Author: SevenNet Inference Package
Date: 2026-01-07
"""

import numpy as np
from ase.build import bulk
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

try:
    from sevennet_inference import SevenNetCalculator
except ImportError:
    print("Error: sevennet_inference package not found.")
    print("Please install it first using: pip install -e .")
    exit(1)

try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
except ImportError:
    print("Error: phonopy package not found.")
    print("Please install it using: pip install phonopy")
    exit(1)


def ase_to_phonopy(atoms):
    """Convert ASE Atoms to PhonopyAtoms"""
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell,
        positions=atoms.positions,
        pbc=True
    )


def calculate_phonons():
    """
    Calculate phonon band structure and density of states for silicon.
    """
    print("="*60)
    print("Phonon Calculation with SevenNet")
    print("="*60)
    
    # Step 1: Create unit cell
    print("\n[1/5] Creating Si unit cell...")
    atoms = bulk('Si', 'diamond', a=5.43)
    print(f"Unit cell: {len(atoms)} atoms")
    print(f"Space group: Fd-3m (diamond structure)")
    
    # Step 2: Set up calculator
    print("\n[2/5] Initializing SevenNet calculator...")
    calc = SevenNetCalculator(model_path='7net-0', device='cpu')
    atoms.calc = calc
    
    # Step 3: Set up Phonopy
    print("\n[3/5] Setting up phonon calculation...")
    unitcell = ase_to_phonopy(atoms)
    phonon = Phonopy(unitcell, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    phonon.generate_displacements(distance=0.01)
    
    print(f"Supercell: 2x2x2")
    print(f"Number of displacements: {len(phonon.supercells_with_displacements)}")
    
    # Step 4: Calculate forces for displaced structures
    print("\n[4/5] Calculating forces for displaced structures...")
    from ase import Atoms as ASEAtoms
    
    forces_set = []
    for i, supercell in enumerate(phonon.supercells_with_displacements):
        # Convert to ASE atoms
        ase_supercell = ASEAtoms(
            symbols=supercell.symbols,
            positions=supercell.positions,
            cell=supercell.cell,
            pbc=True
        )
        ase_supercell.calc = calc
        
        # Calculate forces
        forces = ase_supercell.get_forces()
        forces_set.append(forces)
        
        if (i + 1) % 10 == 0 or i == len(phonon.supercells_with_displacements) - 1:
            print(f"  Progress: {i+1}/{len(phonon.supercells_with_displacements)}")
    
    # Set forces to Phonopy
    phonon.forces = forces_set
    
    # Step 5: Calculate phonon properties
    print("\n[5/5] Computing phonon band structure and DOS...")
    
    # Produce force constants
    phonon.produce_force_constants()
    
    # Band structure
    bands_dict = {
        'band_paths': [
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]],  # Gamma to X
            [[0.5, 0.0, 0.5], [0.5, 0.25, 0.75]],  # X to W
            [[0.5, 0.25, 0.75], [0.375, 0.375, 0.75]],  # W to K
            [[0.375, 0.375, 0.75], [0.0, 0.0, 0.0]],  # K to Gamma
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],  # Gamma to L
        ],
        'labels': ['$\\Gamma$', 'X', 'W', 'K', '$\\Gamma$', 'L']
    }
    
    phonon.run_band_structure(
        bands_dict['band_paths'],
        with_eigenvectors=False,
        is_band_connection=False
    )
    
    # DOS
    phonon.run_mesh([20, 20, 20])
    phonon.run_total_dos()
    
    # Get results
    band_dict = phonon.get_band_structure_dict()
    dos_dict = phonon.get_total_dos_dict()
    
    # Print summary
    print("\n" + "="*60)
    print("PHONON CALCULATION RESULTS")
    print("="*60)
    
    # Extract frequencies
    frequencies = []
    for path in band_dict['frequencies']:
        frequencies.extend(path.flatten())
    
    print(f"\nBand structure:")
    print(f"  Minimum frequency: {np.min(frequencies):.2f} THz")
    print(f"  Maximum frequency: {np.max(frequencies):.2f} THz")
    
    # Check for imaginary modes
    if np.min(frequencies) < -0.1:
        print(f"  WARNING: Imaginary modes detected!")
    else:
        print(f"  No significant imaginary modes (structure is stable)")
    
    print(f"\nDensity of states:")
    print(f"  Frequency range: {dos_dict['frequency_points'][0]:.2f} - {dos_dict['frequency_points'][-1]:.2f} THz")
    
    # Thermodynamic properties at 300K
    phonon.run_thermal_properties(t_min=300, t_max=300, t_step=100)
    tp_dict = phonon.get_thermal_properties_dict()
    
    print(f"\nThermodynamic properties at 300 K:")
    print(f"  Free energy: {tp_dict['free_energy'][0]:.3f} kJ/mol")
    print(f"  Entropy: {tp_dict['entropy'][0]:.3f} J/K/mol")
    print(f"  Heat capacity: {tp_dict['heat_capacity'][0]:.3f} J/K/mol")
    
    print("="*60)
    
    # Step 6: Plot results
    print("\n[6/6] Generating plots...")
    plot_phonon_results(band_dict, dos_dict, bands_dict['labels'])
    
    print("\n✓ Phonon calculation completed successfully!")
    print("\nOutput files:")
    print("  - phonon_band.png")
    print("  - phonon_dos.png")


def plot_phonon_results(band_dict, dos_dict, labels):
    """
    Plot phonon band structure and density of states.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot band structure
    distances = band_dict['distances']
    frequencies = band_dict['frequencies']
    
    for i, (distance, freq) in enumerate(zip(distances, frequencies)):
        ax1.plot(distance, freq, 'b-', linewidth=0.5)
    
    # Add labels
    label_positions = [distances[0][0]]
    for i in range(len(distances) - 1):
        label_positions.append(distances[i][-1])
    label_positions.append(distances[-1][-1])
    
    ax1.set_xticks(label_positions)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Frequency (THz)', fontsize=12)
    ax1.set_title('Phonon Band Structure', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    # Add vertical lines at high-symmetry points
    for pos in label_positions:
        ax1.axvline(x=pos, color='k', linewidth=0.5)
    
    # Plot DOS
    freq_points = dos_dict['frequency_points']
    dos = dos_dict['total_dos']
    
    ax2.plot(dos, freq_points, 'r-', linewidth=1.5)
    ax2.fill_betweenx(freq_points, 0, dos, alpha=0.3, color='red')
    ax2.set_xlabel('DOS (states/THz)', fontsize=12)
    ax2.set_ylabel('Frequency (THz)', fontsize=12)
    ax2.set_title('Phonon Density of States', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('phonon_band.png', dpi=300, bbox_inches='tight')
    print("  Saved: phonon_band.png")
    
    # Separate DOS plot
    fig2, ax = plt.subplots(figsize=(6, 5))
    ax.plot(dos, freq_points, 'r-', linewidth=2)
    ax.fill_betweenx(freq_points, 0, dos, alpha=0.3, color='red')
    ax.set_xlabel('DOS (states/THz)', fontsize=12)
    ax.set_ylabel('Frequency (THz)', fontsize=12)
    ax.set_title('Phonon Density of States - Silicon', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('phonon_dos.png', dpi=300, bbox_inches='tight')
    print("  Saved: phonon_dos.png")
    
    plt.close('all')


if __name__ == "__main__":
    calculate_phonons()
