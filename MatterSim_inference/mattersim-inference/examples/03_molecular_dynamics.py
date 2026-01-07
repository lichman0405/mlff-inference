#!/usr/bin/env python
"""
Example 03: Molecular Dynamics Simulation

Demonstrates how to use MatterSimInference to run MD simulations.
MatterSim ranks #1 in MD stability on MOFSimBench (tied with eSEN).
"""

from ase.build import bulk
from mattersim_inference import MatterSimInference


def main():
    """Molecular dynamics example."""
    print("=" * 60)
    print("MatterSim Inference - Example 03: Molecular Dynamics Simulation")
    print("MOFSimBench MD Stability Ranking: #1 🥇")
    print("=" * 60)
    
    # 1. Initialize model
    print("\n1. Initializing MatterSim model...")
    calc = MatterSimInference(
        model_name="MatterSim-v1-5M",
        device="auto"
    )
    
    # 2. Create supercell
    print("\n2. Creating supercell structure...")
    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
    print(f"   Number of atoms: {len(atoms)}")
    
    # 3. Optimize structure first
    print("\n3. Optimizing initial structure...")
    opt_result = calc.optimize(atoms, fmax=0.05)
    atoms = opt_result['atoms']
    print(f"   Optimization complete, energy: {opt_result['final_energy']:.4f} eV")
    
    # 4. NVT MD simulation
    print("\n4. Running NVT MD simulation...")
    print("   Temperature: 300 K")
    print("   Steps: 1000")
    print("   Timestep: 1 fs")
    
    final_atoms = calc.run_md(
        atoms,
        ensemble="nvt",
        temperature=300,
        steps=1000,
        timestep=1.0,
        trajectory="nvt_md.traj",
        logfile="nvt_md.log",
        log_interval=100
    )
    
    print(f"\n5. MD completed!")
    print(f"   Final number of atoms: {len(final_atoms)}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("Trajectory saved to: nvt_md.traj")
    print("=" * 60)


if __name__ == "__main__":
    main()
