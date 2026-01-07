"""
Example 03: Molecular Dynamics Simulation with GRACE

This example demonstrates how to run a molecular dynamics simulation
using the GRACE model as the calculator.
"""

from ase import Atoms
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from grace_inference import GRACECalculator


def main():
    """Run molecular dynamics simulation using GRACE."""
    
    # Create a 2x2x2 supercell of Silicon
    atoms = bulk('Si', 'diamond', a=5.43).repeat((2, 2, 2))
    
    print("=" * 60)
    print("GRACE Molecular Dynamics Simulation")
    print("=" * 60)
    print(f"Structure: Silicon supercell (2x2x2)")
    print(f"Number of atoms: {len(atoms)}")
    
    # Initialize GRACE calculator
    calc = GRACECalculator(
        model_path='auto',
        device='cpu'
    )
    atoms.calc = calc
    
    # Set up MD parameters
    temperature = 300  # Kelvin
    timestep = 1.0  # fs
    num_steps = 100
    
    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    
    # Create MD object
    dyn = VelocityVerlet(atoms, timestep * units.fs)
    
    # Function to print MD info
    def print_md_info():
        """Print current MD step information."""
        step = dyn.nsteps
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        temp = atoms.get_temperature()
        
        if step % 10 == 0:
            print(f"Step {step:4d}: "
                  f"Epot = {epot:8.3f} eV, "
                  f"Ekin = {ekin:8.3f} eV, "
                  f"T = {temp:6.1f} K")
    
    # Attach observer
    dyn.attach(print_md_info, interval=1)
    
    # Run MD
    print(f"\nRunning MD at {temperature} K for {num_steps} steps...")
    print(f"Timestep: {timestep} fs")
    print("-" * 60)
    dyn.run(num_steps)
    
    print("-" * 60)
    print("MD simulation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
