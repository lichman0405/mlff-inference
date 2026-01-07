"""
Example 2: Molecular Dynamics Simulation

This example demonstrates:
1. NVT molecular dynamics (constant temperature)
2. NPT molecular dynamics (constant pressure and temperature)
3. Trajectory analysis
"""

from orb_inference import OrbInference
from orb_inference.tasks.dynamics import analyze_md_trajectory
from ase.build import bulk
from ase.io import read

print("="*60)
print("Orb Inference - Molecular Dynamics Example")
print("="*60)

# Create Cu structure
atoms = bulk('Cu', 'fcc', a=3.6).repeat((3, 3, 3))

# Initialize Orb
print("\n1. Initializing Orb model...")
orb = OrbInference(model_name='orb-v3-omat', device='cuda')

# Optimize first
print("\n2. Optimizing structure before MD...")
opt_result = orb.optimize(atoms, fmax=0.05, relax_cell=True)
optimized = opt_result['atoms']
print(f"   Initial volume: {atoms.get_volume():.2f} Å³")
print(f"   Optimized volume: {optimized.get_volume():.2f} Å³")

# NVT molecular dynamics
print("\n3. Running NVT MD (300 K, 5000 steps)...")
nvt_final = orb.run_md(
    optimized,
    temperature=300.0,
    steps=5000,
    timestep=1.0,        # 1 fs
    ensemble='nvt',
    trajectory='nvt_md.traj',
    logfile='nvt_md.log'
)

print(f"   Final temperature: {nvt_final.get_temperature():.2f} K")
print(f"   Final volume: {nvt_final.get_volume():.2f} Å³")
print(f"   Trajectory saved to 'nvt_md.traj'")

# Analyze NVT trajectory
print("\n4. Analyzing NVT trajectory...")
traj_nvt = read('nvt_md.traj', ':')
analysis = analyze_md_trajectory(traj_nvt)

print(f"   Average temperature: {analysis['avg_temperature']:.2f} ± {analysis['std_temperature']:.2f} K")
print(f"   Average volume: {analysis['avg_volume']:.2f} ± {analysis['std_volume']:.2f} Å³")
print(f"   Average energy: {analysis['avg_energy']:.4f} eV")
print(f"   Energy drift: {analysis['energy_drift']:.6f} eV")

# NPT molecular dynamics
print("\n5. Running NPT MD (300 K, 0 GPa, 5000 steps)...")
npt_final = orb.run_md(
    optimized,
    temperature=300.0,
    pressure=0.0,        # 0 GPa (ambient pressure)
    steps=5000,
    timestep=1.0,
    ensemble='npt',
    trajectory='npt_md.traj',
    logfile='npt_md.log'
)

print(f"   Final temperature: {npt_final.get_temperature():.2f} K")
print(f"   Final volume: {npt_final.get_volume():.2f} Å³")
print(f"   Volume change: {(npt_final.get_volume() - optimized.get_volume()):.2f} Å³")

# Analyze NPT trajectory
print("\n6. Analyzing NPT trajectory...")
traj_npt = read('npt_md.traj', ':')
analysis_npt = analyze_md_trajectory(traj_npt)

print(f"   Average temperature: {analysis_npt['avg_temperature']:.2f} ± {analysis_npt['std_temperature']:.2f} K")
print(f"   Average volume: {analysis_npt['avg_volume']:.2f} ± {analysis_npt['std_volume']:.2f} Å³")
print(f"   Volume fluctuation: {(analysis_npt['std_volume']/analysis_npt['avg_volume']*100):.2f}%")
print(f"   Energy drift: {analysis_npt['energy_drift']:.6f} eV")

print("\n" + "="*60)
print("MD simulation completed!")
print("Trajectories: nvt_md.traj, npt_md.traj")
print("Log files: nvt_md.log, npt_md.log")
print("="*60)
