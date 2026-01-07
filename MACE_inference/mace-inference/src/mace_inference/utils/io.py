"""I/O utilities for structure handling"""

from typing import Union, List
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import read, write


def load_structure(filepath: Union[str, Path], index: int = -1) -> Atoms:
    """
    Load atomic structure from file.
    
    Args:
        filepath: Path to structure file (CIF, POSCAR, XYZ, etc.)
        index: Frame index for trajectory files (-1 for last frame)
        
    Returns:
        ASE Atoms object
        
    Examples:
        >>> atoms = load_structure("structure.cif")
        >>> atoms = load_structure("trajectory.traj", index=0)  # First frame
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Structure file not found: {filepath}")
    
    return read(str(filepath), index=index)


def save_structure(
    atoms: Atoms,
    filepath: Union[str, Path],
    format: str = None,
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
        >>> save_structure(atoms, "output.cif")
        >>> save_structure(atoms, "output.xyz", format="xyz")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    write(str(filepath), atoms, format=format, **kwargs)


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
        return load_structure(structure)
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
    
    if len(supercell_matrix) != 3:
        raise ValueError(f"Supercell matrix must have 3 elements, got {len(supercell_matrix)}")
    
    return atoms * tuple(supercell_matrix)


def atoms_to_dict(atoms: Atoms) -> dict:
    """
    Convert Atoms object to dictionary (for JSON serialization).
    
    Args:
        atoms: ASE Atoms object
        
    Returns:
        Dictionary representation
    """
    return {
        "symbols": atoms.get_chemical_symbols(),
        "positions": atoms.get_positions().tolist(),
        "cell": atoms.get_cell().tolist() if atoms.pbc.any() else None,
        "pbc": atoms.pbc.tolist(),
        "numbers": atoms.numbers.tolist(),
    }


def dict_to_atoms(data: dict) -> Atoms:
    """
    Convert dictionary to Atoms object.
    
    Args:
        data: Dictionary with structure data
        
    Returns:
        ASE Atoms object
    """
    return Atoms(
        symbols=data["symbols"],
        positions=data["positions"],
        cell=data.get("cell"),
        pbc=data.get("pbc", [False, False, False])
    )
