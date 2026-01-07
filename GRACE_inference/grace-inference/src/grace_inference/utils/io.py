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
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 0.5:  # Less than 0.5 Å
                raise ValueError(
                    f"Atoms {i} and {j} are too close ({dist:.3f} Å). "
                    "Possible overlapping atoms."
                )


def get_structure_info(atoms: Atoms) -> dict:
    """
    Get information about atomic structure.
    
    Args:
        atoms: ASE Atoms object
        
    Returns:
        Dictionary with structure information
        
    Examples:
        >>> info = get_structure_info(atoms)
        >>> print(f"Formula: {info['formula']}")
        >>> print(f"Number of atoms: {info['natoms']}")
    """
    from collections import Counter
    
    symbols = atoms.get_chemical_symbols()
    symbol_counts = Counter(symbols)
    
    return {
        "formula": atoms.get_chemical_formula(),
        "natoms": len(atoms),
        "symbols": list(set(symbols)),
        "composition": dict(symbol_counts),
        "volume": atoms.get_volume(),
        "cell": atoms.get_cell().array.tolist(),
        "pbc": atoms.pbc.tolist(),
    }


def read_trajectory(filepath: Union[str, Path]) -> List[Atoms]:
    """
    Read trajectory file.
    
    Args:
        filepath: Path to trajectory file
        
    Returns:
        List of ASE Atoms objects
        
    Examples:
        >>> frames = read_trajectory("md.traj")
        >>> print(f"Number of frames: {len(frames)}")
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Trajectory file not found: {filepath}")
    
    return read(str(filepath), index=":")


def parse_structure_input(
    structure: Union[str, Path, Atoms]
) -> Atoms:
    """
    Parse structure input (file path or Atoms object).
    
    Args:
        structure: Structure file path or ASE Atoms object
        
    Returns:
        ASE Atoms object
        
    Examples:
        >>> atoms = parse_structure_input("structure.cif")
        >>> atoms = parse_structure_input(atoms_obj)
    """
    if isinstance(structure, (str, Path)):
        return read_structure(structure)
    elif isinstance(structure, Atoms):
        return structure
    else:
        raise TypeError(
            f"Invalid structure type: {type(structure)}. "
            "Must be file path or ASE Atoms object."
        )


def create_supercell(
    atoms: Atoms,
    size: tuple = (2, 2, 2)
) -> Atoms:
    """
    Create supercell from unit cell.
    
    Args:
        atoms: Unit cell ASE Atoms object
        size: Supercell size (nx, ny, nz)
        
    Returns:
        Supercell ASE Atoms object
        
    Examples:
        >>> supercell = create_supercell(atoms, size=(3, 3, 3))
    """
    from ase.build import make_supercell
    
    P = np.diag(size)
    return make_supercell(atoms, P)
