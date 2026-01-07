"""
Example 3: Phonon Calculation and Thermodynamic Properties

This example demonstrates:
- Phonon DOS calculation
- Thermodynamic properties (heat capacity, entropy)
- Detection of imaginary modes
- Thermal property plotting

Requirements:
- esen-inference
- phonopy
"""

from esen_inference import ESENInference
from esen_inference.tasks.phonon import plot_phonon_dos, plot_thermal_properties
from ase.build import bulk
import matplotlib.pyplot as plt

print("=" * 60)
print("Example 3: Phonon Calculation with eSEN")
print("=" * 60)

# ====================================
# 1. Initialize
# ====================================
print("\n1. Initializing eSEN model...")
esen = ESENInference(model_name='esen-30m-oam', device='cuda')

# ====================================
# 2. Create Primitive Cell
# ====================================
print("\n2. Creating primitive cell (Cu FCC)...")
primitive = bulk('Cu', 'fcc', a=3.6)
print(f"✓ Primitive cell: {len(primitive)} atoms")
print(f"  Formula: {primitive.get_chemical_formula()}")

# ====================================
# 3. Optimize Primitive Cell
# ====================================
print("\n3. Optimizing primitive cell (high precision)...")
opt_result = esen.optimize(
    primitive,
    fmax=0.001,         # Tight convergence for phonons
    relax_cell=True,
    optimizer='LBFGS',
    max_steps=500
)

if not opt_result['converged']:
    print("⚠ Warning: Optimization did not converge!")
else:
    print(f"✓ Optimized in {opt_result['steps']} steps")
    print(f"  Final fmax: {opt_result['final_fmax']:.6f} eV/Å")

primitive_opt = opt_result['atoms']

# ====================================
# 4. Phonon Calculation
# ====================================
print("\n4. Calculating phonons...")
print("   Supercell: 3×3×3")
print("   k-mesh: 30×30×30")
print("   Displacement: 0.01 Å")
print("   (This may take a few minutes...)")

result = esen.phonon(
    primitive_opt,
    supercell_matrix=[3, 3, 3],  # Larger supercell for better accuracy
    mesh=[30, 30, 30],           # Dense k-mesh
    displacement=0.01,
    t_min=0,
    t_max=1000,
    t_step=10
)

print("✓ Phonon calculation completed")

# ====================================
# 5. Check for Imaginary Modes
# ====================================
print("\n5. Checking for imaginary modes...")
if result['has_imaginary']:
    print(f"⚠ Warning: {result['imaginary_modes']} imaginary modes detected!")
    print("   This suggests the structure may be dynamically unstable.")
    print("   Consider further optimization or checking the cell parameters.")
else:
    print("✓ No imaginary modes detected")
    print("  Structure is dynamically stable")

# ====================================
# 6. Plot Phonon DOS
# ====================================
print("\n6. Plotting phonon DOS...")
plot_phonon_dos(
    result['frequency_points'],
    result['total_dos'],
    output='phonon_dos_cu.png',
    title='Cu Phonon Density of States',
    xlim=(0, 10)  # THz
)
print("✓ Phonon DOS saved to: phonon_dos_cu.png")

# ====================================
# 7. Extract Thermal Properties
# ====================================
print("\n7. Analyzing thermal properties...")
thermal = result['thermal']

# Heat capacity at 300 K
temps = thermal['temperatures']
Cv = thermal['heat_capacity']

idx_300K = (temps >= 300).argmax()
Cv_300K = Cv[idx_300K]

print(f"✓ Heat capacity at 300 K: {Cv_300K:.2f} J/(K·mol)")

# Find Debye temperature (approximate)
# At low T, Cv ~ (12π⁴/5) * N * k_B * (T/θ_D)³
# We use the temperature where Cv reaches ~90% of 3Nk_B
Cv_classical = 3 * 8.314  # 3R per atom
idx_90pct = (Cv >= 0.9 * Cv_classical).argmax()
T_debye_approx = temps[idx_90pct] if idx_90pct > 0 else temps[-1]
print(f"  Approximate Debye temperature: {T_debye_approx:.0f} K")

# ====================================
# 8. Plot Thermal Properties
# ====================================
print("\n8. Plotting thermal properties...")

# Molar mass of Cu
mass_Cu = 63.546  # g/mol

plot_thermal_properties(
    temps,
    Cv,
    output='heat_capacity_cu.png',
    title='Cu Heat Capacity',
    mass_per_formula=mass_Cu
)
print("✓ Heat capacity plot saved to: heat_capacity_cu.png")

# Create comprehensive thermal plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Heat capacity
axes[0, 0].plot(temps, Cv, 'r-', linewidth=2)
axes[0, 0].axhline(Cv_classical, color='k', linestyle='--', 
                   label=f'Classical limit (3R = {Cv_classical:.2f})')
axes[0, 0].set_xlabel('Temperature (K)')
axes[0, 0].set_ylabel('Cv (J/(K·mol))')
axes[0, 0].set_title('Heat Capacity')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Entropy
axes[0, 1].plot(temps, thermal['entropy'], 'g-', linewidth=2)
axes[0, 1].set_xlabel('Temperature (K)')
axes[0, 1].set_ylabel('Entropy (J/(K·mol))')
axes[0, 1].set_title('Entropy')
axes[0, 1].grid(True, alpha=0.3)

# Free energy
axes[1, 0].plot(temps, thermal['free_energy'], 'b-', linewidth=2)
axes[1, 0].set_xlabel('Temperature (K)')
axes[1, 0].set_ylabel('Free Energy (kJ/mol)')
axes[1, 0].set_title('Helmholtz Free Energy')
axes[1, 0].grid(True, alpha=0.3)

# Phonon DOS
axes[1, 1].plot(result['frequency_points'], result['total_dos'], 'k-', linewidth=1.5)
axes[1, 1].fill_between(result['frequency_points'], 0, result['total_dos'], alpha=0.3)
axes[1, 1].set_xlabel('Frequency (THz)')
axes[1, 1].set_ylabel('DOS (states/THz)')
axes[1, 1].set_title('Phonon DOS')
axes[1, 1].set_xlim(0, 10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('thermal_properties_full.png', dpi=300)
print("✓ Full thermal plot saved to: thermal_properties_full.png")

# ====================================
# Summary
# ====================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Structure: Cu FCC (primitive cell)")
print(f"Imaginary modes: {result['imaginary_modes']}")
print(f"Heat capacity at 300 K: {Cv_300K:.2f} J/(K·mol)")
print(f"Approximate Debye temperature: {T_debye_approx:.0f} K")
print(f"\nThermal properties calculated for:")
print(f"  Temperature range: {temps[0]:.0f} - {temps[-1]:.0f} K")
print(f"  Number of points: {len(temps)}")
print("=" * 60)

print("\n✓ Example 3 completed successfully!")
print("\nNext: python 04_mechanical_properties.py")
