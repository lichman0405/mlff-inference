"""
Example 03: Molecular Dynamics Simulation with SevenNet

This example demonstrates how to:
1. Set up MD simulations with SevenNet
2. Run NVE and NVT ensembles
3. Analyze temperature and energy conservation

Author: SevenNet Inference Package
Date: 2026-01-07
"""

import numpy as np
from ase import Atoms, units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory

try:
    from sevennet_inference import SevenNetCalculator
except ImportError:
    print("Error: sevennet_inference package not found.")
    print("Please install it first using: pip install -e .")
    exit(1)


def run_nve_dynamics():
    """
    Run NVE (microcanonical) molecular dynamics simulation.
    """
    print("="*60)
    print("NVE Molecular Dynamics Example")
    print("="*60)
    
    # Create a 2x2x2 supercell of Si
    atoms = bulk('Si', 'diamond', a=5.43).repeat((2, 2, 2))
    print(f"\nSystem: Si supercell with {len(atoms)} atoms")
    print(f"Volume: {atoms.get_volume():.2f} Å³")
    
    # Set up calculator
    calc = SevenNetCalculator(model_path='7net-0', device='cpu')
    atoms.calc = calc
    
    # Set initial temperature
    temperature = 300  # K
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    
    # Remove center of mass motion
    from ase.md.verlet import VelocityVerlet
    
    print(f"\nInitial temperature: {temperature} K")
    print(f"Time step: 1.0 fs")
    print(f"Total time: 100 fs")
    
    # Set up NVE dynamics
    dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
    
    # Attach trajectory file
    traj = Trajectory('nve_dynamics.traj', 'w', atoms)
    dyn.attach(traj.write, interval=5)
    
    # Storage for analysis
    energies = []
    temperatures = []
    
    def print_status():
        """Print current status during MD"""
        epot = atoms.get_potential_energy() / len(atoms)
        ekin = atoms.get_kinetic_energy() / len(atoms)
        etot = epot + ekin
        temp = ekin / (1.5 * units.kB)
        
        energies.append(etot)
        temperatures.append(temp)
        
        if dyn.nsteps % 20 == 0:
            print(f"Step {dyn.nsteps:3d}: T={temp:6.1f} K  "
                  f"Etot={etot:8.4f} eV/atom  "
                  f"Epot={epot:8.4f}  Ekin={ekin:8.4f}")
    
    dyn.attach(print_status, interval=1)
    
    # Run dynamics
    print("\nRunning NVE dynamics...")
    dyn.run(100)
    
    # Analysis
    print("\n" + "="*60)
    print("NVE DYNAMICS RESULTS")
    print("="*60)
    print(f"Average temperature: {np.mean(temperatures):.2f} ± {np.std(temperatures):.2f} K")
    print(f"Total energy drift: {(energies[-1] - energies[0]):.6f} eV/atom")
    print(f"Energy std dev: {np.std(energies):.6f} eV/atom")
    print("="*60)
    
    traj.close()


def run_nvt_dynamics():
    """
    Run NVT (canonical) molecular dynamics simulation using Langevin thermostat.
    """
    print("\n\n" + "="*60)
    print("NVT Molecular Dynamics Example")
    print("="*60)
    
    # Create structure
    atoms = bulk('Si', 'diamond', a=5.43).repeat((2, 2, 2))
    print(f"\nSystem: Si supercell with {len(atoms)} atoms")
    
    # Set up calculator
    calc = SevenNetCalculator(model_path='7net-0', device='cpu')
    atoms.calc = calc
    
    # Set initial temperature
    temperature = 500  # K
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    
    print(f"\nTarget temperature: {temperature} K")
    print(f"Thermostat: Langevin")
    print(f"Friction: 0.002 fs⁻¹")
    print(f"Time step: 1.0 fs")
    print(f"Total time: 200 fs")
    
    # Set up Langevin thermostat (NVT)
    dyn = Langevin(
        atoms,
        timestep=1.0*units.fs,
        temperature_K=temperature,
        friction=0.002
    )
    
    # Attach trajectory
    traj = Trajectory('nvt_dynamics.traj', 'w', atoms)
    dyn.attach(traj.write, interval=10)
    
    # Storage for analysis
    temperatures = []
    potential_energies = []
    
    def print_status():
        """Print current status during MD"""
        epot = atoms.get_potential_energy() / len(atoms)
        ekin = atoms.get_kinetic_energy() / len(atoms)
        temp = ekin / (1.5 * units.kB)
        
        temperatures.append(temp)
        potential_energies.append(epot)
        
        if dyn.nsteps % 20 == 0:
            print(f"Step {dyn.nsteps:3d}: T={temp:6.1f} K  "
                  f"Epot={epot:8.4f} eV/atom  "
                  f"Ekin={ekin:8.4f} eV/atom")
    
    dyn.attach(print_status, interval=1)
    
    # Run equilibration
    print("\nRunning NVT equilibration...")
    dyn.run(100)
    
    # Reset statistics
    temperatures = []
    potential_energies = []
    
    print("\nRunning NVT production...")
    dyn.run(100)
    
    # Analysis
    print("\n" + "="*60)
    print("NVT DYNAMICS RESULTS")
    print("="*60)
    print(f"Target temperature: {temperature} K")
    print(f"Average temperature: {np.mean(temperatures):.2f} ± {np.std(temperatures):.2f} K")
    print(f"Temperature range: [{np.min(temperatures):.1f}, {np.max(temperatures):.1f}] K")
    print(f"Average potential energy: {np.mean(potential_energies):.4f} ± {np.std(potential_energies):.4f} eV/atom")
    print("="*60)
    
    traj.close()


if __name__ == "__main__":
    # Run NVE dynamics
    run_nve_dynamics()
    
    # Run NVT dynamics
    run_nvt_dynamics()
    
    print("\n✓ Molecular dynamics examples completed successfully!")
    print("\nTrajectory files saved:")
    print("  - nve_dynamics.traj")
    print("  - nvt_dynamics.traj")
