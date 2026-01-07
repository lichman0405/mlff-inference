"""
Molecular dynamics module.

Provides MD simulations for NVE, NVT, NPT ensembles.
MatterSim ranks #1 in MD stability in MOFSimBench (tied with eSEN).
"""

from typing import Any, Dict, Optional

import numpy as np
from ase import Atoms, units
from ase.io import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.npt import NPT


def run_md(
    atoms: Atoms,
    calculator: Any,
    ensemble: str = "nvt",
    temperature: float = 300.0,
    pressure: Optional[float] = None,
    steps: int = 10000,
    timestep: float = 1.0,
    trajectory: Optional[str] = None,
    logfile: Optional[str] = None,
    log_interval: int = 100
) -> Atoms:
    """
    Run molecular dynamics simulation.
    
    MatterSim ranks #1 in MD stability in MOFSimBench (tied with eSEN).
    
    Args:
        atoms: ASE Atoms object
        calculator: ASE Calculator
        ensemble: Ensemble type
            - "nve": Microcanonical (constant E)
            - "nvt": Canonical (constant T)
            - "npt": Isothermal-isobaric (constant T, P)
        temperature: Temperature (K)
        pressure: Pressure (GPa), only needed for NPT
        steps: Number of simulation steps
        timestep: Time step (fs)
        trajectory: Trajectory file path
        logfile: Log file path
        log_interval: Log recording interval
    
    Returns:
        Atoms: Final structure
    """
    atoms = atoms.copy()
    atoms.calc = calculator
    
    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    
    # Set time step
    dt = timestep * units.fs
    
    # Select integrator
    ensemble = ensemble.lower()
    
    if ensemble == "nve":
        dyn = VelocityVerlet(atoms, timestep=dt, logfile=logfile)
    
    elif ensemble == "nvt":
        # Langevin thermostat
        friction = 0.01 / units.fs
        dyn = Langevin(
            atoms,
            timestep=dt,
            temperature_K=temperature,
            friction=friction,
            logfile=logfile
        )
    
    elif ensemble == "npt":
        # NPT ensemble
        if pressure is None:
            pressure = 0.0
        
        # Convert pressure: GPa -> eV/Å³
        pressure_eV_A3 = pressure / 160.2176634
        
        dyn = NPT(
            atoms,
            timestep=dt,
            temperature_K=temperature,
            externalstress=pressure_eV_A3,
            ttime=25 * units.fs,
            pfactor=None,
            logfile=logfile
        )
    
    else:
        raise ValueError(f"Unknown ensemble: {ensemble}")
    
    # Set up trajectory
    if trajectory:
        traj = Trajectory(trajectory, 'w', atoms)
        dyn.attach(traj.write, interval=log_interval)
    
    # Run MD
    dyn.run(steps)
    
    if trajectory:
        traj.close()
    
    return atoms


def analyze_md_trajectory(
    trajectory_file: str,
    start: int = 0,
    end: Optional[int] = None,
    step: int = 1
) -> Dict[str, Any]:
    """
    Analyze MD trajectory.
    
    Args:
        trajectory_file: Trajectory file path
        start: Starting frame
        end: Ending frame
        step: Frame interval
    
    Returns:
        dict: Analysis results
    """
    from ase.io import read
    
    frames = read(trajectory_file, index=slice(start, end, step))
    
    if isinstance(frames, Atoms):
        frames = [frames]
    
    energies = []
    temperatures = []
    volumes = []
    
    for frame in frames:
        if frame.calc is not None:
            try:
                energies.append(frame.get_potential_energy())
            except:
                pass
        
        # Calculate temperature from kinetic energy
        try:
            kinetic = frame.get_kinetic_energy()
            n_atoms = len(frame)
            temp = 2 * kinetic / (3 * n_atoms * units.kB)
            temperatures.append(temp)
        except:
            pass
        
        volumes.append(frame.get_volume())
    
    return {
        "n_frames": len(frames),
        "energies": np.array(energies) if energies else None,
        "temperatures": np.array(temperatures) if temperatures else None,
        "volumes": np.array(volumes),
        "energy_mean": np.mean(energies) if energies else None,
        "temperature_mean": np.mean(temperatures) if temperatures else None,
        "volume_mean": np.mean(volumes),
    }
