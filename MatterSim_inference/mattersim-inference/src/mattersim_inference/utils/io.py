"""
I/O utilities module.

Provides structure file read/write and validation functionality.
"""

from pathlib import Path
from typing import Tuple, Union, Optional, List
import warnings

from ase import Atoms
from ase.io import read, write


def read_structure(
    filepath: Union[str, Path],
    format: Optional[str] = None,
    index: int = -1
) -> Atoms:
    """
    Read a structure file.
    
    Supported formats: CIF, POSCAR/VASP, XYZ, PDB, extxyz, etc.
    
    Args:
        filepath: Path to the structure file
        format: File format (optional, auto-detected)
        index: Frame index to read (default: last frame)
    
    Returns:
        Atoms: ASE Atoms object
    
    Raises:
        FileNotFoundError: File does not exist
        ValueError: Unable to parse file
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Structure file not found: {filepath}")
    
    try:
        atoms = read(str(filepath), format=format, index=index)
        return atoms
    except Exception as e:
        raise ValueError(f"Failed to read structure from {filepath}: {e}")


def write_structure(
    atoms: Atoms,
    filepath: Union[str, Path],
    format: Optional[str] = None,
    **kwargs
) -> None:
    """
    Write a structure file.
    
    Args:
        atoms: ASE Atoms object
        filepath: Output file path
        format: File format (optional, auto-detected from extension)
        **kwargs: Additional arguments passed to ase.io.write
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    write(str(filepath), atoms, format=format, **kwargs)


def validate_structure(atoms: Atoms) -> Tuple[bool, str]:
    """
    Validate structure integrity.
    
    Checks:
    - Number of atoms
    - Cell parameters
    - Atomic distances
    - PBC settings
    
    Args:
        atoms: ASE Atoms object
    
    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    issues = []
    
    # Check number of atoms
    if len(atoms) == 0:
        return False, "Structure has no atoms"
    
    # Check cell
    if atoms.cell is None or atoms.cell.volume < 1e-6:
        issues.append("Cell volume is too small or undefined")
    
    # Check PBC
    if not all(atoms.pbc):
        issues.append("Not all PBC directions are enabled")
    
    # Check atomic distances
    try:
        from ase.geometry import get_distances
        positions = atoms.get_positions()
        cell = atoms.get_cell()
        
        # Simple check for minimum distance
        if len(atoms) > 1:
            d_matrix, _ = get_distances(positions, cell=cell, pbc=atoms.pbc)
            min_dist = d_matrix[d_matrix > 0].min() if d_matrix[d_matrix > 0].size > 0 else 0
            
            if min_dist < 0.5:
                issues.append(f"Atoms too close: min distance = {min_dist:.3f} Å")
    except Exception as e:
        issues.append(f"Distance check failed: {e}")
    
    if issues:
        return False, "; ".join(issues)
    
    return True, "Structure is valid"


def get_structure_info(atoms: Atoms) -> dict:
    """
    Get structure information summary.
    
    Args:
        atoms: ASE Atoms object
    
    Returns:
        dict: Structure information
    """
    symbols = atoms.get_chemical_symbols()
    unique_symbols = list(set(symbols))
    
    composition = {}
    for s in unique_symbols:
        composition[s] = symbols.count(s)
    
    cell = atoms.get_cell()
    
    return {
        "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "composition": composition,
        "volume": cell.volume,
        "cell_lengths": list(cell.lengths()),
        "cell_angles": list(cell.angles()),
        "pbc": list(atoms.pbc),
    }


def read_trajectory(
    filepath: Union[str, Path],
    format: Optional[str] = None
) -> List[Atoms]:
    """
    Read a trajectory file.
    
    Args:
        filepath: Path to the trajectory file
        format: File format
    
    Returns:
        List[Atoms]: List of frames
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Trajectory file not found: {filepath}")
    
    frames = read(str(filepath), format=format, index=":")
    
    if isinstance(frames, Atoms):
        frames = [frames]
    
    return frames
