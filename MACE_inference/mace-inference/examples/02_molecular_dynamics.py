"""
Example 2: Molecular Dynamics - NVT and NPT simulations

This example shows how to run MD simulations.
"""

from ase.build import bulk
from mace_inference import MACEInference

# Initialize calculator
print("Initializing MACE calculator...")
calc = MACEInference(model="medium", device="auto")

# Create structure
print("\nCreating Al FCC structure...")
atoms = bulk('Al', 'fcc', a=4.05)
atoms = atoms * (3, 3, 3)  # 3x3x3 supercell for MD

print(f"Number of atoms: {len(atoms)}")
print(f"Initial volume: {atoms.get_volume():.2f} Å³")

# NVT Molecular Dynamics
print("\n=== Running NVT MD (300 K, 1000 steps) ===")
final_nvt = calc.run_md(
    atoms,
    ensemble="nvt",
    temperature_K=300,
    steps=1000,
    timestep=2.0,  # fs
    trajectory="nvt_md.traj",
    logfile="nvt_md.log",
    log_interval=100
)

print("✓ NVT simulation completed")
print(f"Final volume: {final_nvt.get_volume():.2f} Å³")

# NPT Molecular Dynamics
print("\n=== Running NPT MD (300 K, 0 GPa, 2000 steps) ===")
final_npt = calc.run_md(
    atoms,
    ensemble="npt",
    temperature_K=300,
    pressure_GPa=0.0,  # Atmospheric pressure
    steps=2000,
    timestep=2.0,  # fs
    trajectory="npt_md.traj",
    logfile="npt_md.log",
    log_interval=100
)

print("✓ NPT simulation completed")
print(f"Final volume: {final_npt.get_volume():.2f} Å³")
print(f"Volume change: {(final_npt.get_volume() - atoms.get_volume()) / atoms.get_volume() * 100:.2f}%")

# Analyze trajectory
print("\n=== Analyzing Trajectory ===")
from ase.io import read

traj_nvt = read("nvt_md.traj", ":")
print(f"NVT trajectory frames: {len(traj_nvt)}")

# Calculate average volume in NPT
traj_npt = read("npt_md.traj", ":")
volumes = [frame.get_volume() for frame in traj_npt]
import numpy as np
print(f"\nNPT Volume Statistics:")
print(f"  Mean:   {np.mean(volumes):.2f} Å³")
print(f"  Std:    {np.std(volumes):.2f} Å³")
print(f"  Min:    {np.min(volumes):.2f} Å³")
print(f"  Max:    {np.max(volumes):.2f} Å³")

print("\n✓ Trajectories saved:")
print("  - nvt_md.traj")
print("  - npt_md.traj")
print("  - nvt_md.log")
print("  - npt_md.log")
