"""I/O utilities for structure handling"""

from typing import Union, List, Optional
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import read, write, Trajectory


def read_structure(filepath: Union[str, Path], index: int = -1) -> Atoms:
    """
    Load atomic structure from file.
    
    Args:
        filepath: Path to structure file (CIF, POSCAR, XYZ, etc.)
        index: Frame index for trajectory files (-1 for last frame)
        
    Returns:
        ASE Atoms object
        
    Raises:
        FileNotFoundError: If file does not exist
        
    Examples:
        >>> atoms = read_structure("structure.cif")
        >>> atoms = read_structure("trajectory.traj", index=0)  # First frame
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Structure file not found: {filepath}")
    
    return read(str(filepath), index=index)


def write_structure(
    atoms: Atoms,
    filepath: Union[str, Path],
    format: Optional[str] = None,
    **kwargs
) -> None:
    """
    Save atomic structure to file.
    
    Args:
        atoms: ASE Atoms object
        filepath: Output file path
        format: File format (auto-detected from extension if None)
        **kwargs: Additional arguments passed to ase.io.write
        
    Examples:
        >>> write_structure(atoms, "output.cif")
        >>> write_structure(atoms, "output.xyz", format="xyz")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    write(str(filepath), atoms, format=format, **kwargs)


def validate_structure(atoms: Atoms) -> None:
    """
    Validate atomic structure.
    
    Args:
        atoms: ASE Atoms object to validate
        
    Raises:
        ValueError: If structure is invalid
        
    Examples:
        >>> validate_structure(atoms)  # Raises error if invalid
    """
    if len(atoms) == 0:
        raise ValueError("Structure contains no atoms")
    
    if not atoms.pbc.any() and np.all(atoms.get_cell() == 0):
        raise ValueError("Structure has no cell and no periodic boundary conditions")
    
    # Check for overlapping atoms
    positions = atoms.get_positions()
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 0.5:  # Less than 0.5 Å
                raise ValueError(
                    f"Atoms {i} and {j} are too close (distance: {dist:.3f} Å). "
                    "Check for overlapping atoms."
                )


def get_structure_info(atoms: Atoms) -> dict:
    """
    Get basic information about atomic structure.
    
    Args:
        atoms: ASE Atoms object
        
    Returns:
        Dictionary with structure information
        
    Examples:
        >>> info = get_structure_info(atoms)
        >>> print(info['formula'])
        'C24H12'
    """
    from collections import Counter
    
    symbols = atoms.get_chemical_symbols()
    composition = Counter(symbols)
    
    cell = atoms.get_cell()
    volume = atoms.get_volume()
    
    # Calculate density if cell is defined
    density = None
    if volume > 0:
        mass = sum(atoms.get_masses())  # amu
        density = mass / volume / 0.6022  # g/cm³
    
    return {
        "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "composition": dict(composition),
        "cell_lengths": cell.lengths(),
        "cell_angles": cell.angles(),
        "volume": volume,
        "density": density,
        "pbc": atoms.pbc.tolist(),
    }


def read_trajectory(
    filepath: Union[str, Path],
    start: int = 0,
    stop: Optional[int] = None,
    step: int = 1
) -> List[Atoms]:
    """
    Read trajectory file and return list of Atoms objects.
    
    Args:
        filepath: Path to trajectory file
        start: Starting frame index
        stop: Ending frame index (None for all frames)
        step: Step between frames
        
    Returns:
        List of ASE Atoms objects
        
    Examples:
        >>> frames = read_trajectory("md.traj")
        >>> frames = read_trajectory("md.traj", start=100, stop=200, step=10)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Trajectory file not found: {filepath}")
    
    # Read all frames
    if filepath.suffix == ".traj":
        traj = Trajectory(str(filepath), mode='r')
        frames = [atoms.copy() for atoms in traj[start:stop:step]]
        traj.close()
    else:
        # Use ASE read with slice
        if stop is None:
            frames = read(str(filepath), index=f"{start}:")
        else:
            frames = read(str(filepath), index=f"{start}:{stop}:{step}")
        
        # Ensure frames is a list
        if not isinstance(frames, list):
            frames = [frames]
    
    return frames


def parse_structure_input(
    structure: Union[str, Path, Atoms, List[Atoms]]
) -> Union[Atoms, List[Atoms]]:
    """
    Parse various structure input formats.
    
    Args:
        structure: Structure file path, Atoms object, or list of Atoms
        
    Returns:
        Atoms object or list of Atoms objects
        
    Examples:
        >>> atoms = parse_structure_input("structure.cif")
        >>> atoms = parse_structure_input(existing_atoms)
    """
    if isinstance(structure, (str, Path)):
        return read_structure(structure)
    elif isinstance(structure, Atoms):
        return structure
    elif isinstance(structure, list):
        if not all(isinstance(a, Atoms) for a in structure):
            raise ValueError("All elements in list must be ASE Atoms objects")
        return structure
    else:
        raise TypeError(
            f"Invalid structure input type: {type(structure)}. "
            "Expected str, Path, Atoms, or List[Atoms]"
        )


def create_supercell(
    atoms: Atoms,
    supercell_matrix: Union[List[int], np.ndarray, int]
) -> Atoms:
    """
    Create a supercell from the input structure.
    
    Args:
        atoms: Input ASE Atoms object
        supercell_matrix: Supercell size (e.g., [2, 2, 2] or 2 for isotropic)
        
    Returns:
        Supercell Atoms object
        
    Examples:
        >>> supercell = create_supercell(atoms, [2, 2, 2])
        >>> supercell = create_supercell(atoms, 2)  # Same as [2, 2, 2]
    """
    if isinstance(supercell_matrix, int):
        supercell_matrix = [supercell_matrix] * 3
    
    supercell_matrix = np.array(supercell_matrix)
    
    if supercell_matrix.shape == (3,):
        # Diagonal matrix
        supercell_matrix = np.diag(supercell_matrix)
    
    # Use ASE's repeat method for simple diagonal supercells
    if np.allclose(supercell_matrix, np.diag(np.diag(supercell_matrix))):
        repeats = np.diag(supercell_matrix).astype(int)
        return atoms.repeat(repeats)
    else:
        # For general supercell matrices, use make_supercell
        from ase.build import make_supercell
        return make_supercell(atoms, supercell_matrix)
