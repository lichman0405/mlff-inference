"""
I/O utilities for eSEN Inference

This module provides functions for reading and writing atomic structures
in various formats (CIF, POSCAR, XYZ, etc.).
"""

from ase import Atoms
from ase.io import read as ase_read, write as ase_write
from typing import Union, Optional, List
from pathlib import Path
import warnings


def read_structure(
    filename: Union[str, Path],
    index: Union[int, str] = -1,
    format: Optional[str] = None,
    **kwargs
) -> Union[Atoms, List[Atoms]]:
    """
    Read atomic structure from file.
    
    Args:
        filename: Path to structure file
        index: Which frame(s) to read:
               - -1: last frame (default)
               - 0: first frame
               - ':': all frames
               - '::2': every 2nd frame
        format: File format (auto-detected if None)
        **kwargs: Additional arguments passed to ase.io.read()
    
    Returns:
        Atoms object or list of Atoms (if index is slice)
    
    Supported formats:
        - CIF (.cif)
        - POSCAR/CONTCAR (.vasp, POSCAR, CONTCAR)
        - XYZ (.xyz)
        - PDB (.pdb)
        - LAMMPS data (.data, .lmp)
        - ASE trajectory (.traj)
        - And many more via ASE
    
    Example:
        >>> atoms = read_structure('MOF-5.cif')
        >>> atoms = read_structure('POSCAR', format='vasp')
        >>> trajectory = read_structure('md.traj', index=':')
        >>> every_10th = read_structure('md.traj', index='::10')
    
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is invalid
    """
    filename = Path(filename)
    
    if not filename.exists():
        raise FileNotFoundError(f"Structure file not found: {filename}")
    
    try:
        atoms = ase_read(str(filename), index=index, format=format, **kwargs)
    except Exception as e:
        raise ValueError(
            f"Failed to read structure from {filename}. "
            f"Error: {e}. "
            f"Please check the file format and content."
        ) from e
    
    return atoms


def write_structure(
    atoms: Atoms,
    filename: Union[str, Path],
    format: Optional[str] = None,
    **kwargs
):
    """
    Write atomic structure to file.
    
    Args:
        atoms: Atoms object to write
        filename: Output file path
        format: File format (auto-detected from extension if None)
        **kwargs: Additional arguments passed to ase.io.write()
    
    Supported formats:
        - CIF (.cif): Crystallographic Information File
        - POSCAR (.vasp, POSCAR): VASP input format
        - XYZ (.xyz): XYZ coordinates
        - PDB (.pdb): Protein Data Bank
        - LAMMPS data (.data, .lmp): LAMMPS input
        - ASE trajectory (.traj): ASE binary format
    
    Example:
        >>> write_structure(atoms, 'optimized.cif')
        >>> write_structure(atoms, 'POSCAR', format='vasp')
        >>> write_structure(atoms, 'output.xyz')
    
    Raises:
        ValueError: If format is unsupported or write fails
    """
    filename = Path(filename)
    
    # Create parent directory if needed
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        ase_write(str(filename), atoms, format=format, **kwargs)
    except Exception as e:
        raise ValueError(
            f"Failed to write structure to {filename}. "
            f"Error: {e}"
        ) from e


def validate_structure(atoms: Atoms, verbose: bool = False) -> bool:
    """
    Validate atomic structure for common issues.
    
    Checks:
    - Atom count > 0
    - Positions are finite
    - Cell is defined (for periodic systems)
    - No overlapping atoms (distance < 0.5 Å)
    
    Args:
        atoms: Atoms object to validate
        verbose: Whether to print warnings
    
    Returns:
        True if structure is valid, False otherwise
    
    Example:
        >>> atoms = read_structure('MOF-5.cif')
        >>> if validate_structure(atoms, verbose=True):
        ...     print("Structure is valid")
    """
    is_valid = True
    
    # Check atom count
    if len(atoms) == 0:
        if verbose:
            warnings.warn("Structure has no atoms")
        return False
    
    # Check positions
    import numpy as np
    positions = atoms.get_positions()
    if not np.all(np.isfinite(positions)):
        if verbose:
            warnings.warn("Structure has non-finite positions")
        is_valid = False
    
    # Check cell for periodic systems
    if atoms.pbc.any():
        cell = atoms.get_cell()
        if not np.all(np.isfinite(cell)):
            if verbose:
                warnings.warn("Periodic structure has invalid cell")
            is_valid = False
        
        # Check cell volume
        volume = atoms.get_volume()
        if volume < 1.0:
            if verbose:
                warnings.warn(f"Cell volume is very small: {volume:.3f} Å³")
            is_valid = False
    
    # Check for overlapping atoms
    from ase.neighborlist import neighbor_list
    try:
        i, j, d = neighbor_list('ijd', atoms, cutoff=0.5)
        if len(i) > 0:
            if verbose:
                warnings.warn(
                    f"Found {len(i)} atom pairs closer than 0.5 Å. "
                    f"Minimum distance: {d.min():.3f} Å"
                )
            is_valid = False
    except Exception:
        # neighbor_list can fail for certain structures
        pass
    
    return is_valid


def convert_structure(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    input_format: Optional[str] = None,
    output_format: Optional[str] = None,
    index: Union[int, str] = -1
):
    """
    Convert structure file from one format to another.
    
    Args:
        input_file: Input structure file
        output_file: Output structure file
        input_format: Input format (auto-detect if None)
        output_format: Output format (auto-detect if None)
        index: Which frame to convert (for trajectories)
    
    Example:
        >>> convert_structure('POSCAR', 'structure.cif')
        >>> convert_structure('md.traj', 'final.xyz', index=-1)
    """
    atoms = read_structure(input_file, index=index, format=input_format)
    write_structure(atoms, output_file, format=output_format)


def get_structure_info(atoms: Atoms) -> dict:
    """
    Get summary information about structure.
    
    Args:
        atoms: Atoms object
    
    Returns:
        Dictionary with structure information:
        - n_atoms: Number of atoms
        - formula: Chemical formula
        - composition: Element counts
        - volume: Cell volume (Å³, if periodic)
        - density: Mass density (g/cm³, if periodic)
        - pbc: Periodic boundary conditions
    
    Example:
        >>> atoms = read_structure('MOF-5.cif')
        >>> info = get_structure_info(atoms)
        >>> print(f"Formula: {info['formula']}")
        >>> print(f"Volume: {info['volume']:.2f} Å³")
    """
    from collections import Counter
    import numpy as np
    
    info = {
        'n_atoms': len(atoms),
        'formula': atoms.get_chemical_formula(),
        'composition': dict(Counter(atoms.get_chemical_symbols())),
        'pbc': tuple(atoms.pbc)
    }
    
    if atoms.pbc.any():
        info['volume'] = atoms.get_volume()
        
        # Calculate density (g/cm³)
        masses = atoms.get_masses()
        total_mass = np.sum(masses)  # amu
        volume_cm3 = info['volume'] * 1e-24  # Å³ to cm³
        
        # 1 amu = 1.66054e-24 g
        density = (total_mass * 1.66054e-24) / volume_cm3
        info['density'] = density
    else:
        info['volume'] = None
        info['density'] = None
    
    return info
