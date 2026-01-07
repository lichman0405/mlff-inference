"""Molecular dynamics simulations."""

from typing import Optional, Literal
from ase import Atoms
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units


def run_nvt_md(
    atoms: Atoms,
    calculator,
    temperature_K: float = 300,
    timestep: float = 1.0,
    steps: int = 1000,
    friction: Optional[float] = None,
    taut: Optional[float] = None,
    trajectory: Optional[str] = None,
    logfile: Optional[str] = None,
    loginterval: int = 100
) -> Atoms:
    """
    Run NVT molecular dynamics simulation (Langevin thermostat).
    
    Args:
        atoms: Initial structure
        calculator: ORBCalculator instance
        temperature_K: Target temperature (K)
        timestep: Time step (fs)
        steps: Number of MD steps
        friction: Friction coefficient (1/fs), default 0.01
        taut: Temperature relaxation time (fs), alternative to friction
        trajectory: Trajectory file path (.traj)
        logfile: Log file path
        loginterval: Logging interval (steps)
        
    Returns:
        Final Atoms object
        
    Note:
        Either friction or taut should be provided, not both.
        Relation: taut = 1 / friction
        
    Examples:
        >>> final_atoms = run_nvt_md(
        ...     atoms, calc,
        ...     temperature_K=300,
        ...     steps=50000,  # 50 ps
        ...     friction=0.01,
        ...     trajectory="nvt_md.traj"
        ... )
    """
    if atoms.calc is None:
        atoms.calc = calculator
    
    # Set friction coefficient
    if friction is None and taut is None:
        friction = 0.01  # default: 100 fs relaxation time
    elif friction is not None and taut is not None:
        raise ValueError("Specify either friction or taut, not both")
    elif taut is not None:
        friction = 1.0 / taut
    
    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    
    # Create NVT dynamics
    dyn = Langevin(
        atoms,
        timestep=timestep * units.fs,
        temperature_K=temperature_K,
        friction=friction,
        trajectory=trajectory,
        logfile=logfile,
        loginterval=loginterval
    )
    
    # Run MD
    dyn.run(steps=steps)
    
    return atoms


def run_npt_md(
    atoms: Atoms,
    calculator,
    temperature_K: float = 300,
    pressure_GPa: Optional[float] = None,
    timestep: float = 1.0,
    steps: int = 1000,
    ttime: Optional[float] = None,
    pfactor: Optional[float] = None,
    trajectory: Optional[str] = None,
    logfile: Optional[str] = None,
    loginterval: int = 100
) -> Atoms:
    """
    Run NPT molecular dynamics simulation (Berendsen barostat).
    
    Args:
        atoms: Initial structure
        calculator: ORBCalculator instance
        temperature_K: Target temperature (K)
        pressure_GPa: Target pressure (GPa), default 0.0 (1 atm ≈ 0.0001 GPa)
        timestep: Time step (fs)
        steps: Number of MD steps
        ttime: Temperature relaxation time (fs), default 100 fs
        pfactor: Pressure coupling factor, auto-estimated if None
        trajectory: Trajectory file path (.traj)
        logfile: Log file path
        loginterval: Logging interval (steps)
        
    Returns:
        Final Atoms object
        
    Note:
        pfactor is estimated as: (timestep^2) * B / V
        where B is bulk modulus (~20 GPa for MOFs) and V is volume.
        
    Examples:
        >>> final_atoms = run_npt_md(
        ...     atoms, calc,
        ...     temperature_K=300,
        ...     pressure_GPa=0.0,  # 1 atm
        ...     steps=50000,
        ...     trajectory="npt_md.traj"
        ... )
    """
    if atoms.calc is None:
        atoms.calc = calculator
    
    # Default values
    if pressure_GPa is None:
        pressure_GPa = 0.0
    if ttime is None:
        ttime = 100.0  # fs
    
    # Convert pressure: GPa -> eV/Å³
    externalstress = pressure_GPa / 160.21766208
    
    # Estimate pfactor if not provided
    if pfactor is None:
        volume = atoms.get_volume()
        bulk_modulus_GPa = 20.0  # Typical for MOFs
        pfactor = (timestep**2) * bulk_modulus_GPa / volume / 160.21766208
    
    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    
    # Create NPT dynamics
    dyn = NPT(
        atoms,
        timestep=timestep * units.fs,
        temperature_K=temperature_K,
        externalstress=externalstress,
        ttime=ttime * units.fs,
        pfactor=pfactor,
        trajectory=trajectory,
        logfile=logfile,
        loginterval=loginterval
    )
    
    # Run MD
    dyn.run(steps=steps)
    
    return atoms


def analyze_md_trajectory(trajectory_file: str) -> dict:
    """
    Analyze MD trajectory.
    
    Args:
        trajectory_file: Path to trajectory file (.traj)
        
    Returns:
        Dictionary with:
            - n_frames: Number of frames
            - volumes: Volume at each frame (Å³)
            - energies: Energy at each frame (eV)
            - temperatures: Temperature at each frame (K)
            
    Examples:
        >>> from ase.io import Trajectory
        >>> result = analyze_md_trajectory("npt_md.traj")
        >>> print(f"Volume drift: {result['volume_drift']:.2f}%")
    """
    from ase.io import Trajectory
    import numpy as np
    
    traj = Trajectory(trajectory_file)
    
    volumes = []
    energies = []
    temperatures = []
    
    for atoms in traj:
        volumes.append(atoms.get_volume())
        try:
            energies.append(atoms.get_potential_energy())
        except:
            energies.append(np.nan)
        
        try:
            # Calculate instantaneous temperature
            ekin = atoms.get_kinetic_energy() / len(atoms)
            temp = 2 * ekin / (3 * units.kB)
            temperatures.append(temp)
        except:
            temperatures.append(np.nan)
    
    volumes = np.array(volumes)
    energies = np.array(energies)
    temperatures = np.array(temperatures)
    
    # Calculate volume drift
    if len(volumes) > 0:
        volume_drift = (volumes[-1] / volumes[0] - 1) * 100
    else:
        volume_drift = 0.0
    
    return {
        "n_frames": len(traj),
        "volumes": volumes,
        "energies": energies,
        "temperatures": temperatures,
        "volume_drift": volume_drift,
        "avg_volume": np.mean(volumes),
        "std_volume": np.std(volumes),
        "avg_temperature": np.nanmean(temperatures),
        "std_temperature": np.nanstd(temperatures),
    }
