"""
Molecular dynamics simulations for eSEN models
"""

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase import units
from typing import Optional, List
import numpy as np


class DynamicsTask:
    """
    Handler for molecular dynamics simulations.
    
    Supports:
    - NVE (microcanonical)
    - NVT (canonical, Langevin thermostat)
    - NPT (isothermal-isobaric, Berendsen barostat)
    """
    
    def __init__(self, calculator: Calculator):
        """
        Initialize DynamicsTask.
        
        Args:
            calculator: ASE calculator (OCPCalculator for eSEN)
        """
        self.calculator = calculator
    
    def run_md(
        self,
        atoms: Atoms,
        temperature: float = 300.0,
        pressure: Optional[float] = None,
        steps: int = 10000,
        timestep: float = 1.0,
        ensemble: str = 'nvt',
        friction: float = 0.01,
        taut: Optional[float] = None,
        taup: Optional[float] = None,
        compressibility: Optional[float] = None,
        trajectory: Optional[str] = None,
        logfile: Optional[str] = None,
        log_interval: int = 100
    ) -> Atoms:
        """
        Run molecular dynamics simulation.
        
        Args:
            atoms: Initial structure
            temperature: Temperature in K
            pressure: Pressure in GPa (for NPT)
            steps: Number of MD steps
            timestep: Time step in fs
            ensemble: 'nve', 'nvt', or 'npt'
            friction: Friction coefficient in ps^-1 (for NVT Langevin)
            taut: Temperature relaxation time in fs (for NPT, default 100)
            taup: Pressure relaxation time in fs (for NPT, default 1000)
            compressibility: Compressibility in GPa^-1 (for NPT)
            trajectory: Trajectory file path
            logfile: Log file path
            log_interval: Interval for trajectory/log output
        
        Returns:
            Final Atoms object after MD
        """
        # Attach calculator
        atoms.calc = self.calculator
        
        # Initialize velocities
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        
        # Remove center of mass translation
        momentum = atoms.get_momenta().sum(axis=0)
        atoms.set_momenta(atoms.get_momenta() - momentum / len(atoms))
        
        # Setup MD integrator
        timestep_fs = timestep * units.fs
        
        if ensemble.lower() == 'nve':
            # NVE (constant energy)
            dyn = VelocityVerlet(atoms, timestep_fs)
        
        elif ensemble.lower() == 'nvt':
            # NVT with Langevin thermostat
            friction_ase = friction / (1000 * units.fs)  # Convert ps^-1 to ASE units
            dyn = Langevin(
                atoms,
                timestep_fs,
                temperature_K=temperature,
                friction=friction_ase
            )
        
        elif ensemble.lower() == 'npt':
            # NPT with Berendsen barostat
            if pressure is None:
                raise ValueError("Pressure must be specified for NPT ensemble")
            
            # Convert pressure from GPa to eV/Å³
            pressure_eV_A3 = pressure * 0.00624150907
            
            # Default time constants
            if taut is None:
                taut = 100.0  # fs
            if taup is None:
                taup = 1000.0  # fs
            
            # Compressibility
            if compressibility is None:
                # Default for MOFs: ~4.57e-5 GPa^-1
                compressibility = 4.57e-5
            
            # Convert compressibility from GPa^-1 to eV^-1 Å^3
            compressibility_ase = compressibility / 0.00624150907
            
            dyn = NPT(
                atoms,
                timestep_fs,
                temperature_K=temperature,
                externalstress=pressure_eV_A3,
                ttime=taut * units.fs,
                pfactor=(taup * units.fs)**2 * compressibility_ase
            )
        
        else:
            raise ValueError(
                f"Unknown ensemble '{ensemble}'. Choose from: 'nve', 'nvt', 'npt'"
            )
        
        # Attach trajectory writer
        if trajectory is not None:
            from ase.io.trajectory import Trajectory
            traj = Trajectory(trajectory, 'w', atoms)
            dyn.attach(traj.write, interval=log_interval)
        
        # Attach logger
        if logfile is not None:
            from ase.md import MDLogger
            logger = MDLogger(
                dyn,
                atoms,
                logfile,
                header=True,
                stress=atoms.pbc.any(),
                peratom=False,
                mode='w'
            )
            dyn.attach(logger, interval=log_interval)
        
        # Run MD
        dyn.run(steps)
        
        return atoms


def analyze_md_trajectory(trajectory: List[Atoms]) -> dict:
    """
    Analyze MD trajectory.
    
    Args:
        trajectory: List of Atoms objects from MD simulation
    
    Returns:
        Dictionary with analysis results:
        - avg_temperature: Average temperature (K)
        - std_temperature: Temperature std dev (K)
        - avg_volume: Average volume (Å³)
        - std_volume: Volume std dev (Å³)
        - avg_energy: Average total energy (eV)
        - energy_drift: Energy drift (eV)
        - msd: Mean squared displacement (Å²)
    """
    n_frames = len(trajectory)
    
    # Temperature
    temperatures = np.array([atoms.get_temperature() for atoms in trajectory])
    avg_temp = np.mean(temperatures)
    std_temp = np.std(temperatures)
    
    # Volume (for periodic systems)
    if trajectory[0].pbc.any():
        volumes = np.array([atoms.get_volume() for atoms in trajectory])
        avg_vol = np.mean(volumes)
        std_vol = np.std(volumes)
    else:
        avg_vol = None
        std_vol = None
    
    # Energy
    energies = np.array([
        atoms.get_potential_energy() + atoms.get_kinetic_energy()
        for atoms in trajectory
    ])
    avg_energy = np.mean(energies)
    
    # Energy drift (final - initial)
    energy_drift = energies[-1] - energies[0]
    
    # Mean squared displacement
    initial_positions = trajectory[0].get_positions()
    msd = []
    for atoms in trajectory:
        positions = atoms.get_positions()
        displacement = positions - initial_positions
        msd_frame = np.mean(np.sum(displacement**2, axis=1))
        msd.append(msd_frame)
    
    return {
        'avg_temperature': float(avg_temp),
        'std_temperature': float(std_temp),
        'avg_volume': float(avg_vol) if avg_vol is not None else None,
        'std_volume': float(std_vol) if std_vol is not None else None,
        'avg_energy': float(avg_energy),
        'energy_drift': float(energy_drift),
        'msd': np.array(msd)
    }
