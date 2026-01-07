"""
Example 2: Molecular Dynamics Simulation with eSEN

This example demonstrates:
- NVT molecular dynamics (Langevin thermostat)
- NPT molecular dynamics (Berendsen barostat)
- Trajectory analysis
- Temperature/energy monitoring

Requirements:
- esen-inference
"""

from esen_inference import ESENInference
from esen_inference.tasks.dynamics import analyze_md_trajectory
from ase.build import bulk
from ase.io import read, write
import matplotlib.pyplot as plt

print("=" * 60)
print("Example 2: Molecular Dynamics with eSEN")
print("=" * 60)

# ====================================
# 1. Initialize
# ====================================
print("\n1. Initializing eSEN model...")
esen = ESENInference(model_name='esen-30m-oam', device='cuda')

# Create test structure (Cu crystal, 108 atoms)
print("\n2. Creating Cu structure (108 atoms)...")
atoms = bulk('Cu', 'fcc', a=3.6, cubic=True) * (3, 3, 3)
print(f"✓ Created: {len(atoms)} atoms, V = {atoms.get_volume():.2f} Å³")

# ====================================
# 3. Optimize First
# ====================================
print("\n3. Pre-optimization...")
opt_result = esen.optimize(atoms, fmax=0.01, relax_cell=True, max_steps=200)
atoms_opt = opt_result['atoms']
print(f"✓ Optimized: E = {opt_result['final_energy']:.6f} eV")

# ====================================
# 4. NVT Molecular Dynamics
# ====================================
print("\n4. Running NVT MD (300 K, 10 ps)...")
print("   Ensemble: NVT (canonical)")
print("   Temperature: 300 K")
print("   Steps: 10,000 (1 fs/step = 10 ps)")

final_nvt = esen.run_md(
    atoms_opt.copy(),
    temperature=300.0,
    steps=10000,
    timestep=1.0,
    ensemble='nvt',
    friction=0.01,           # Langevin friction (ps^-1)
    trajectory='nvt_md.traj',
    logfile='nvt_md.log',
    log_interval=100
)

print(f"✓ NVT MD completed")
print(f"  Final T: {final_nvt.get_temperature():.2f} K")
print(f"  Final V: {final_nvt.get_volume():.2f} Å³")

# ====================================
# 5. NPT Molecular Dynamics
# ====================================
print("\n5. Running NPT MD (300 K, 1 atm, 10 ps)...")
print("   Ensemble: NPT (isothermal-isobaric)")
print("   Temperature: 300 K")
print("   Pressure: 0 GPa (1 atm)")

final_npt = esen.run_md(
    atoms_opt.copy(),
    temperature=300.0,
    pressure=0.0,            # 0 GPa = 1 atm
    steps=10000,
    timestep=1.0,
    ensemble='npt',
    taut=100.0,              # T relaxation time (fs)
    taup=1000.0,             # P relaxation time (fs)
    compressibility=4.57e-5,  # Cu: ~1.4e-5 GPa^-1 (approximate)
    trajectory='npt_md.traj',
    logfile='npt_md.log',
    log_interval=100
)

print(f"✓ NPT MD completed")
print(f"  Final T: {final_npt.get_temperature():.2f} K")
print(f"  Final V: {final_npt.get_volume():.2f} Å³")
print(f"  Volume change: {(final_npt.get_volume() - atoms_opt.get_volume())/atoms_opt.get_volume()*100:.2f}%")

# ====================================
# 6. Analyze NVT Trajectory
# ====================================
print("\n6. Analyzing NVT trajectory...")
nvt_traj = read('nvt_md.traj', ':')
nvt_analysis = analyze_md_trajectory(nvt_traj)

print("✓ NVT Analysis:")
print(f"  Avg temperature: {nvt_analysis['avg_temperature']:.2f} ± {nvt_analysis['std_temperature']:.2f} K")
print(f"  Avg energy: {nvt_analysis['avg_energy']:.4f} eV")
print(f"  Energy drift: {nvt_analysis['energy_drift']:.6f} eV")
print(f"  MSD (final): {nvt_analysis['msd'][-1]:.4f} Å²")

# ====================================
# 7. Analyze NPT Trajectory
# ====================================
print("\n7. Analyzing NPT trajectory...")
npt_traj = read('npt_md.traj', ':')
npt_analysis = analyze_md_trajectory(npt_traj)

print("✓ NPT Analysis:")
print(f"  Avg temperature: {npt_analysis['avg_temperature']:.2f} ± {npt_analysis['std_temperature']:.2f} K")
print(f"  Avg volume: {npt_analysis['avg_volume']:.2f} ± {npt_analysis['std_volume']:.2f} Å³")
print(f"  Energy drift: {npt_analysis['energy_drift']:.6f} eV")

# ====================================
# 8. Plot Results
# ====================================
print("\n8. Plotting MD results...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# NVT: Temperature
temps_nvt = [atoms.get_temperature() for atoms in nvt_traj]
axes[0, 0].plot(temps_nvt, 'b-', linewidth=1)
axes[0, 0].axhline(300, color='r', linestyle='--', label='Target')
axes[0, 0].set_xlabel('Frame')
axes[0, 0].set_ylabel('Temperature (K)')
axes[0, 0].set_title('NVT: Temperature Evolution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# NVT: Energy
energies_nvt = [atoms.get_potential_energy() + atoms.get_kinetic_energy() 
                for atoms in nvt_traj]
axes[0, 1].plot(energies_nvt, 'g-', linewidth=1)
axes[0, 1].set_xlabel('Frame')
axes[0, 1].set_ylabel('Total Energy (eV)')
axes[0, 1].set_title('NVT: Energy Conservation')
axes[0, 1].grid(True, alpha=0.3)

# NPT: Temperature
temps_npt = [atoms.get_temperature() for atoms in npt_traj]
axes[1, 0].plot(temps_npt, 'b-', linewidth=1)
axes[1, 0].axhline(300, color='r', linestyle='--', label='Target')
axes[1, 0].set_xlabel('Frame')
axes[1, 0].set_ylabel('Temperature (K)')
axes[1, 0].set_title('NPT: Temperature Evolution')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# NPT: Volume
volumes_npt = [atoms.get_volume() for atoms in npt_traj]
axes[1, 1].plot(volumes_npt, 'm-', linewidth=1)
axes[1, 1].set_xlabel('Frame')
axes[1, 1].set_ylabel('Volume (Å³)')
axes[1, 1].set_title('NPT: Volume Fluctuation')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('md_analysis.png', dpi=300)
print("✓ Plot saved to: md_analysis.png")

# ====================================
# Summary
# ====================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"NVT MD (300 K, 10 ps):")
print(f"  Temperature: {nvt_analysis['avg_temperature']:.2f} ± {nvt_analysis['std_temperature']:.2f} K")
print(f"  Energy drift: {nvt_analysis['energy_drift']:.6f} eV")
print(f"\nNPT MD (300 K, 1 atm, 10 ps):")
print(f"  Temperature: {npt_analysis['avg_temperature']:.2f} ± {npt_analysis['std_temperature']:.2f} K")
print(f"  Volume: {npt_analysis['avg_volume']:.2f} ± {npt_analysis['std_volume']:.2f} Å³")
print("=" * 60)

print("\n✓ Example 2 completed successfully!")
print("\nNext: python 03_phonon_calculation.py")
