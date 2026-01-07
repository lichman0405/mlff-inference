"""Molecular dynamics simulations"""

from typing import Optional
from ase import Atoms
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import Trajectory
from ase import units


def run_md(
    atoms: Atoms,
    calculator,
    ensemble: str = "nvt",
    temperature_K: float = 300,
    pressure_GPa: Optional[float] = None,
    timestep: float = 1.0,
    steps: int = 1000,
    trajectory: Optional[str] = None,
    logfile: Optional[str] = None,
    log_interval: int = 100,
    taut: Optional[float] = None,
    taup: Optional[float] = None,
    friction: Optional[float] = None
) -> Atoms:
    """
    Run molecular dynamics simulation.
    
    Args:
        atoms: Input ASE Atoms object
        calculator: ASE calculator
        ensemble: MD ensemble ("nve", "nvt", "npt")
        temperature_K: Target temperature (K)
        pressure_GPa: Target pressure for NPT (GPa)
        timestep: Time step (fs)
        steps: Number of MD steps
        trajectory: Trajectory file path
        logfile: Log file path
        log_interval: Logging interval (steps)
        taut: Temperature coupling time for NVT (fs)
        taup: Pressure coupling time for NPT (fs)
        friction: Friction coefficient for Langevin dynamics (1/fs)
        
    Returns:
        Final Atoms object after MD
        
    Raises:
        ValueError: If ensemble is not recognized
        
    Examples:
        >>> # NVT simulation
        >>> final_atoms = run_md(atoms, calc, ensemble="nvt", temperature_K=300, steps=1000)
        >>> 
        >>> # NPT simulation
        >>> final_atoms = run_md(atoms, calc, ensemble="npt", temperature_K=300, 
        ...                      pressure_GPa=0.1, steps=5000)
    """
    ensemble = ensemble.lower()
    
    if ensemble == "nve":
        return run_nve_md(atoms, calculator, timestep, steps, trajectory, logfile, log_interval)
    elif ensemble == "nvt":
        return run_nvt_md(
            atoms, calculator, temperature_K, timestep, steps, 
            trajectory, logfile, log_interval, taut, friction
        )
    elif ensemble == "npt":
        if pressure_GPa is None:
            pressure_GPa = 0.0
        return run_npt_md(
            atoms, calculator, temperature_K, pressure_GPa, timestep, steps,
            trajectory, logfile, log_interval, taut, taup
        )
    else:
        raise ValueError(f"Unknown ensemble: {ensemble}. Choose from 'nve', 'nvt', 'npt'")


def run_nve_md(
    atoms: Atoms,
    calculator,
    timestep: float = 1.0,
    steps: int = 1000,
    trajectory: Optional[str] = None,
    logfile: Optional[str] = None,
    log_interval: int = 100
) -> Atoms:
    """
    Run NVE (microcanonical) molecular dynamics.
    
    Args:
        atoms: Input ASE Atoms object
        calculator: ASE calculator
        timestep: Time step (fs)
        steps: Number of MD steps
        trajectory: Trajectory file path
        logfile: Log file path
        log_interval: Logging interval (steps)
        
    Returns:
        Final Atoms object after MD
    """
    atoms = atoms.copy()
    atoms.calc = calculator
    
    # Create velocity Verlet dynamics
    dyn = VelocityVerlet(atoms, timestep=timestep * units.fs)
    
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
        friction = 0.002  # Default friction coefficient (1/fs)
    
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
    taup: Optional[float] = None
) -> Atoms:
    """
    Run NPT molecular dynamics using Berendsen barostat.
    
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
    
    # Estimate bulk modulus for pfactor calculation
    # Typical value for materials: ~10-100 GPa
    B_estimate = 50.0 * units.GPa  # Convert to eV/Å³
    pfactor = (taup * units.fs)**2 * B_estimate
    
    # Convert pressure to stress (eV/Å³)
    # Note: Stress is negative of pressure
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
    
    # Run MD
    dyn.run(steps)
    
    # Close trajectory
    if trajectory:
        traj.close()
    
    return atoms


def equilibrate_temperature(
    atoms: Atoms,
    calculator,
    target_temperature_K: float,
    equilibration_steps: int = 1000,
    timestep: float = 1.0
) -> Atoms:
    """
    Equilibrate system to target temperature.
    
    Args:
        atoms: Input ASE Atoms object
        calculator: ASE calculator
        target_temperature_K: Target temperature (K)
        equilibration_steps: Number of equilibration steps
        timestep: Time step (fs)
        
    Returns:
        Equilibrated Atoms object
    """
    return run_nvt_md(
        atoms,
        calculator,
        temperature_K=target_temperature_K,
        timestep=timestep,
        steps=equilibration_steps,
        friction=0.01  # Stronger damping for equilibration
    )
