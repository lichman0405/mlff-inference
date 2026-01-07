"""
Example 3: Phonon and Thermal Properties

This example demonstrates:
1. Phonon calculation using Phonopy
2. Thermal property calculation (heat capacity, entropy, free energy)
3. Plotting phonon DOS and thermal properties
"""

from orb_inference import OrbInference
from orb_inference.tasks.phonon import plot_phonon_dos, plot_thermal_properties
from ase.build import bulk
import numpy as np

print("="*60)
print("Orb Inference - Phonon Calculation Example")
print("="*60)

# Create Si primitive cell
atoms = bulk('Si', 'diamond', a=5.43)

print(f"\nStructure: Si (diamond)")
print(f"  Atoms: {len(atoms)}")
print(f"  Volume: {atoms.get_volume():.3f} Å³")

# Initialize Orb
print("\n1. Initializing Orb model...")
orb = OrbInference(model_name='orb-v3-omat', device='cuda')

# Optimize primitive cell
print("\n2. Optimizing primitive cell...")
opt_result = orb.optimize(atoms, fmax=0.01, relax_cell=True)
optimized = opt_result['atoms']
print(f"   Optimized lattice constant: {optimized.cell[0, 0]:.4f} Å")
print(f"   Final energy: {opt_result['final_energy']:.6f} eV")

# Calculate phonon
print("\n3. Calculating phonon properties...")
print("   (This will take several minutes - calculating forces for displaced supercells)")
result = orb.phonon(
    optimized,
    supercell_matrix=[2, 2, 2],  # 2x2x2 supercell (64 atoms)
    mesh=[20, 20, 20],           # k-point mesh for DOS
    t_min=0,
    t_max=1000,
    t_step=10
)

# Phonon results
freq_points = result['frequency_points']
total_dos = result['total_dos']

print(f"\n4. Phonon Results:")
print(f"   Frequency range: {freq_points.min():.2f} - {freq_points.max():.2f} THz")
print(f"   Number of modes: {len(freq_points)}")

# Check for imaginary modes (negative frequencies)
imaginary_modes = freq_points[freq_points < -0.1]
if len(imaginary_modes) > 0:
    print(f"   WARNING: {len(imaginary_modes)} imaginary modes detected!")
    print(f"   This suggests structural instability or insufficient optimization.")
else:
    print(f"   ✓ No imaginary modes (structure is stable)")

# Plot phonon DOS
plot_phonon_dos(freq_points, total_dos, output='si_phonon_dos.png')
print("\n5. Phonon DOS plot saved to 'si_phonon_dos.png'")

# Thermal properties
thermal = result['thermal']
temperatures = thermal['temperatures']
heat_capacity = thermal['heat_capacity']
entropy = thermal['entropy']
free_energy = thermal['free_energy']

print("\n6. Thermal Properties:")
print(f"\n   Temperature (K)  |  Cv [J/(K·mol)]  |  S [J/(K·mol)]  |  F [kJ/mol]")
print(f"   " + "-"*70)

for T in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    idx = np.argmin(np.abs(temperatures - T))
    T_actual = temperatures[idx]
    Cv = heat_capacity[idx]
    S = entropy[idx]
    F = free_energy[idx]
    print(f"   {T_actual:6.0f}           |  {Cv:15.4f} | {S:14.4f}  | {F:11.4f}")

# Calculate molar mass for Si (28.0855 g/mol)
molar_mass_si = 28.0855

# Plot thermal properties
plot_thermal_properties(
    temperatures, 
    heat_capacity,
    output='si_thermal_properties.png',
    mass_per_formula=molar_mass_si
)
print("\n7. Thermal properties plot saved to 'si_thermal_properties.png'")

# Classical limit check (Dulong-Petit law)
# For Si: Cv → 3R = 24.94 J/(K·mol) at high T
classical_limit = 3 * 8.314  # 3R
Cv_1000K = heat_capacity[np.argmin(np.abs(temperatures - 1000))]
print(f"\n8. High-temperature limit check:")
print(f"   Cv at 1000 K: {Cv_1000K:.2f} J/(K·mol)")
print(f"   Classical limit (3R): {classical_limit:.2f} J/(K·mol)")
print(f"   Ratio: {Cv_1000K/classical_limit:.2f}")

print("\n" + "="*60)
print("Phonon calculation completed!")
print("Output files:")
print("  - si_phonon_dos.png")
print("  - si_thermal_properties.png")
print("="*60)
