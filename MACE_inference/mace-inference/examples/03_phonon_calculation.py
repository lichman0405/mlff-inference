"""
Example 3: Phonon Calculation and Thermal Properties

This example demonstrates phonon calculations.
"""

from ase.build import bulk
from mace_inference import MACEInference
import matplotlib.pyplot as plt

# Initialize calculator
print("Initializing MACE calculator...")
calc = MACEInference(model="medium", device="auto")

# Create structure (Si diamond)
print("\nCreating Si diamond structure...")
atoms = bulk('Si', 'diamond', a=5.43)

print(f"Number of atoms in unit cell: {len(atoms)}")

# Calculate phonon with thermal properties
print("\n=== Calculating Phonons ===")
print("This may take a few minutes...")

result = calc.phonon(
    atoms,
    supercell_matrix=[2, 2, 2],  # 2x2x2 supercell
    displacement=0.01,
    mesh=[20, 20, 20],
    temperature_range=(0, 1000, 10),  # 0-1000 K, 10 K steps
    output_dir="phonon_output"
)

print("✓ Phonon calculation completed")

# Extract thermal properties
if 'thermal_properties' in result:
    thermal = result['thermal_properties']
    temperatures = thermal['temperatures']
    heat_capacity = thermal['heat_capacity']
    entropy = thermal['entropy']
    free_energy = thermal['free_energy']
    
    print("\n=== Thermal Properties at Selected Temperatures ===")
    for T_target in [100, 200, 300, 500, 800]:
        if T_target <= temperatures[-1]:
            import numpy as np
            idx = np.argmin(np.abs(temperatures - T_target))
            T = temperatures[idx]
            print(f"\nT = {T:.0f} K:")
            print(f"  C_v = {heat_capacity[idx]:.3f} J/(mol·K)")
            print(f"  S   = {entropy[idx]:.3f} J/(mol·K)")
            print(f"  F   = {free_energy[idx]:.3f} kJ/mol")
    
    # Plot thermal properties
    print("\n=== Generating Plots ===")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(temperatures, free_energy)
    axes[0].set_xlabel('Temperature (K)')
    axes[0].set_ylabel('Free Energy (kJ/mol)')
    axes[0].set_title('Helmholtz Free Energy')
    axes[0].grid(True)
    
    axes[1].plot(temperatures, entropy)
    axes[1].set_xlabel('Temperature (K)')
    axes[1].set_ylabel('Entropy (J/(mol·K))')
    axes[1].set_title('Entropy')
    axes[1].grid(True)
    
    axes[2].plot(temperatures, heat_capacity)
    axes[2].set_xlabel('Temperature (K)')
    axes[2].set_ylabel('C_v (J/(mol·K))')
    axes[2].set_title('Heat Capacity')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('thermal_properties.png', dpi=300)
    print("✓ Plot saved to thermal_properties.png")
else:
    print("No thermal properties calculated")

print("\n✓ Phonon files saved to phonon_output/")
