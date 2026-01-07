"""Molecular dynamics simulations"""

from typing import Optional
from ase import Atoms
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import Trajectory
from ase import units
import warnings


def run_nvt_md(
    atoms: Atoms,
    calculator,
    temperature_K: float = 300,
    timestep: float = 1.0,
    steps: int = 1000,
    trajectory: Optional[str] = None,
    logfile: Optional[str] = None,
    log_interval: int = 100,
    taut: Optional[float] = None,
    friction: Optional[float] = None
) -> Atoms:
    """
    Run NVT molecular dynamics using Langevin thermostat.
    
    Args:
        atoms: Input ASE Atoms object
        calculator: ASE calculator
        temperature_K: Target temperature (K)
        timestep: Time step (fs)
        steps: Number of MD steps
        trajectory: Trajectory file path
        logfile: Log file path
        log_interval: Logging interval (steps)
        taut: Temperature coupling time (fs) - alternative to friction
        friction: Friction coefficient (1/fs)
        
    Returns:
        Final Atoms object after MD
    """
    atoms = atoms.copy()
    atoms.calc = calculator
    
    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    
    # Calculate friction from taut if provided
    if taut is not None and friction is None:
        friction = 1.0 / taut  # Convert coupling time to friction
    elif friction is None:
        friction = 0.002  # Default friction coefficient
    
    # Create Langevin dynamics
    dyn = Langevin(
        atoms,
        timestep=timestep * units.fs,
        temperature_K=temperature_K,
        friction=friction / units.fs
    )
    
    # Attach trajectory writer
    if trajectory:
        traj = Trajectory(trajectory, 'w', atoms)
        dyn.attach(traj.write, interval=log_interval)
    
    # Attach logger
    if logfile:
        from ase.md import MDLogger
        logger = MDLogger(
            dyn,
            atoms,
            logfile,
            header=True,
            stress=False,
            peratom=False,
            mode='w'
        )
        dyn.attach(logger, interval=log_interval)
    
    # Run MD
    dyn.run(steps)
    
    # Close trajectory
    if trajectory:
        traj.close()
    
    return atoms


def run_npt_md(
    atoms: Atoms,
    calculator,
    temperature_K: float = 300,
    pressure_GPa: float = 0.0,
    timestep: float = 1.0,
    steps: int = 1000,
    trajectory: Optional[str] = None,
    logfile: Optional[str] = None,
    log_interval: int = 100,
    taut: Optional[float] = None,
    taup: Optional[float] = None,
    compressibility_au: Optional[float] = None
) -> Atoms:
    """
    Run NPT molecular dynamics using NPT ensemble.
    
    Args:
        atoms: Input ASE Atoms object
        calculator: ASE calculator
        temperature_K: Target temperature (K)
        pressure_GPa: Target pressure (GPa)
        timestep: Time step (fs)
        steps: Number of MD steps
        trajectory: Trajectory file path
        logfile: Log file path
        log_interval: Logging interval (steps)
        taut: Temperature coupling time (fs)
        taup: Pressure coupling time (fs)
        compressibility_au: Isothermal compressibility (atomic units)
        
    Returns:
        Final Atoms object after MD
    """
    atoms = atoms.copy()
    atoms.calc = calculator
    
    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    
    # Default coupling times
    if taut is None:
        taut = 100.0  # fs
    if taup is None:
        taup = 1000.0  # fs
    
    # Calculate pfactor from taup if compressibility not provided
    if compressibility_au is None:
        # Estimate bulk modulus (typical for MOFs: ~10 GPa = 0.062 eV/Å³)
        B_estimate = 10.0 * units.GPa  # Convert to eV/Å³
        pfactor = (taup * units.fs)**2 * B_estimate
    else:
        pfactor = (taup * units.fs)**2 / compressibility_au
    
    # Convert pressure to atomic units (stress tensor)
    # Note: NPT uses stress (positive in tension), so negate pressure
    externalstress = -pressure_GPa * units.GPa
    
    # Create NPT dynamics
    dyn = NPT(
        atoms,
        timestep=timestep * units.fs,
        temperature_K=temperature_K,
        externalstress=externalstress,
        ttime=taut * units.fs,
        pfactor=pfactor
    )
    
    # Attach trajectory writer
    if trajectory:
        traj = Trajectory(trajectory, 'w', atoms)
        dyn.attach(traj.write, interval=log_interval)
    
    # Attach logger
    if logfile:
        from ase.md import MDLogger
        logger = MDLogger(
            dyn,
            atoms,
            logfile,
            header=True,
            stress=True,
            peratom=False,
            mode='w'
        )
        dyn.attach(logger, interval=log_interval)
    
    # Attach volume monitor
    def log_volume():
        if dyn.nsteps % log_interval == 0:
            vol = atoms.get_volume()
            if hasattr(dyn, '_initial_volume'):
                vol_change = (vol - dyn._initial_volume) / dyn._initial_volume * 100
            else:
                dyn._initial_volume = vol
                vol_change = 0.0
            print(f"Step {dyn.nsteps}: Volume = {vol:.2f} Å³ ({vol_change:+.2f}%)")
    
    dyn.attach(log_volume, interval=log_interval)
    
    # Run MD
    dyn.run(steps)
    
    # Close trajectory
    if trajectory:
        traj.close()
    
    return atoms
