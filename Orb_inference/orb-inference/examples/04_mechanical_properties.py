"""
Example 4: Mechanical Properties

This example demonstrates:
1. Bulk modulus calculation via equation of state
2. Plotting EOS curve
3. Estimating other elastic moduli
"""

from orb_inference import OrbInference
from orb_inference.tasks.mechanics import (
    plot_eos,
    estimate_youngs_modulus,
    estimate_shear_modulus
)
from ase.build import bulk

print("="*60)
print("Orb Inference - Mechanical Properties Example")
print("="*60)

# Create Al structure (FCC)
atoms = bulk('Al', 'fcc', a=4.05).repeat((2, 2, 2))

print(f"\nStructure: Al (FCC)")
print(f"  Atoms: {len(atoms)}")
print(f"  Initial volume: {atoms.get_volume():.3f} Å³")

# Initialize Orb
print("\n1. Initializing Orb model...")
orb = OrbInference(model_name='orb-v3-omat', device='cuda')

# Calculate bulk modulus
print("\n2. Calculating bulk modulus (this may take a few minutes)...")
print("   Performing volumetric strain and fitting equation of state...")

result = orb.bulk_modulus(
    atoms,
    strain_range=0.05,    # ±5% volume strain
    n_points=7,           # 7 volume points
    optimize_first=True   # Optimize structure first
)

# Results
B = result['bulk_modulus']
V0 = result['equilibrium_volume']
E0 = result['equilibrium_energy']
eos_type = result['eos_type']

print(f"\n3. Equation of State Results:")
print(f"   EOS type: {eos_type}")
print(f"   Equilibrium volume: {V0:.3f} Å³")
print(f"   Equilibrium energy: {E0:.6f} eV")
print(f"   Bulk modulus: {B:.2f} GPa")

# Experimental comparison (Al: ~76 GPa)
B_exp = 76.0
error_percent = abs(B - B_exp) / B_exp * 100
print(f"\n   Experimental bulk modulus (Al): {B_exp:.2f} GPa")
print(f"   Error: {error_percent:.1f}%")

# Plot EOS
plot_eos(
    result['volumes'],
    result['energies'],
    result['eos'],
    output='al_eos.png'
)
print("\n4. EOS plot saved to 'al_eos.png'")

# Estimate other elastic moduli
print("\n5. Estimating other elastic moduli (assuming Poisson's ratio ν = 0.25):")
print("   Note: These are estimates for isotropic materials.")
print("   For accurate values, use DFT or stress-strain calculations.")

E = estimate_youngs_modulus(B, poisson_ratio=0.25)
G = estimate_shear_modulus(B, poisson_ratio=0.25)

print(f"\n   Young's modulus E: {E:.2f} GPa")
print(f"   Shear modulus G: {G:.2f} GPa")

# Bulk/Shear modulus ratio (for ductility assessment)
# B/G > 1.75 suggests ductile behavior
BG_ratio = B / G
print(f"\n   B/G ratio: {BG_ratio:.2f}")
if BG_ratio > 1.75:
    print(f"   → Material is likely ductile (B/G > 1.75)")
else:
    print(f"   → Material is likely brittle (B/G < 1.75)")

# Volume per atom
V_per_atom = V0 / len(atoms)
print(f"\n6. Additional information:")
print(f"   Volume per atom: {V_per_atom:.3f} Å³/atom")
print(f"   Atoms in cell: {len(atoms)}")

print("\n" + "="*60)
print("Mechanical properties calculation completed!")
print("Output file: al_eos.png")
print("="*60)

# Example for MOF (lower bulk modulus)
print("\n" + "="*60)
print("Example 2: MOF-like structure (hypothetical)")
print("="*60)

# Typical MOF bulk moduli: 5-30 GPa
B_mof = 15.0  # GPa (hypothetical)
print(f"\nAssuming bulk modulus B = {B_mof:.2f} GPa for a MOF:")

E_mof = estimate_youngs_modulus(B_mof, poisson_ratio=0.30)  # MOFs often have ν ~ 0.3
G_mof = estimate_shear_modulus(B_mof, poisson_ratio=0.30)

print(f"  Young's modulus E: {E_mof:.2f} GPa")
print(f"  Shear modulus G: {G_mof:.2f} GPa")
print(f"  B/G ratio: {B_mof/G_mof:.2f}")
print("\nMOFs typically have:")
print("  - Low bulk moduli (5-30 GPa) due to porosity")
print("  - High compressibility")
print("  - Anisotropic mechanical behavior")
