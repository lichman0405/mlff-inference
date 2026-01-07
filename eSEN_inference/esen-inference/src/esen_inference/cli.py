"""
Command-line interface for eSEN Inference

Usage:
    esen-infer single-point MOF-5.cif
    esen-infer optimize MOF-5.cif --fmax 0.01 --relax-cell
    esen-infer md MOF-5.cif --temperature 300 --steps 50000
    esen-infer phonon MOF-5_primitive.cif --supercell 2 2 2
    esen-infer bulk-modulus MOF-5.cif --strain-range 0.05
    esen-infer batch-optimize mof_database/*.cif --output-dir optimized/
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np

from esen_inference import ESENInference
from esen_inference.utils.io import read_structure, write_structure


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='eSEN Inference CLI - MOFSimBench #1 Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--version', action='version', version='esen-inference 1.0.0')
    parser.add_argument('--model', default='esen-30m-oam',
                       choices=['esen-30m-oam', 'esen-30m-mp'],
                       help='eSEN model to use')
    parser.add_argument('--device', default='cuda',
                       help='Device (cuda, cpu, mps)')
    parser.add_argument('--precision', default='float32',
                       choices=['float32', 'float64'],
                       help='Numerical precision')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Single-point calculation
    sp_parser = subparsers.add_parser('single-point', help='Single-point energy/forces/stress')
    sp_parser.add_argument('input', help='Input structure file')
    sp_parser.add_argument('--output', help='Output JSON file')
    
    # Optimization
    opt_parser = subparsers.add_parser('optimize', help='Structure optimization')
    opt_parser.add_argument('input', help='Input structure file')
    opt_parser.add_argument('--fmax', type=float, default=0.01, help='Force convergence (eV/Å)')
    opt_parser.add_argument('--relax-cell', action='store_true', help='Optimize cell')
    opt_parser.add_argument('--optimizer', default='LBFGS', choices=['LBFGS', 'BFGS', 'FIRE'])
    opt_parser.add_argument('--max-steps', type=int, default=500, help='Max optimization steps')
    opt_parser.add_argument('--output', help='Output structure file (.cif, .xyz, etc.)')
    opt_parser.add_argument('--trajectory', help='Trajectory file (.traj)')
    
    # MD simulation
    md_parser = subparsers.add_parser('md', help='Molecular dynamics')
    md_parser.add_argument('input', help='Input structure file')
    md_parser.add_argument('--temperature', type=float, default=300.0, help='Temperature (K)')
    md_parser.add_argument('--pressure', type=float, help='Pressure (GPa, for NPT)')
    md_parser.add_argument('--steps', type=int, default=10000, help='Number of MD steps')
    md_parser.add_argument('--timestep', type=float, default=1.0, help='Timestep (fs)')
    md_parser.add_argument('--ensemble', default='nvt', choices=['nve', 'nvt', 'npt'])
    md_parser.add_argument('--trajectory', default='md.traj', help='Trajectory file')
    md_parser.add_argument('--logfile', default='md.log', help='Log file')
    
    # Phonon calculation
    phonon_parser = subparsers.add_parser('phonon', help='Phonon calculation')
    phonon_parser.add_argument('input', help='Input primitive cell file')
    phonon_parser.add_argument('--supercell', nargs=3, type=int, default=[2, 2, 2],
                               help='Supercell size (nx ny nz)')
    phonon_parser.add_argument('--mesh', nargs=3, type=int, default=[20, 20, 20],
                               help='k-point mesh (kx ky kz)')
    phonon_parser.add_argument('--output', default='phonon_dos.png', help='Output plot')
    
    # Bulk modulus
    bulk_parser = subparsers.add_parser('bulk-modulus', help='Bulk modulus calculation')
    bulk_parser.add_argument('input', help='Input structure file')
    bulk_parser.add_argument('--strain-range', type=float, default=0.05, help='Strain range (±)')
    bulk_parser.add_argument('--n-points', type=int, default=7, help='Number of points')
    bulk_parser.add_argument('--output', help='Output JSON file')
    
    # Batch optimization
    batch_parser = subparsers.add_parser('batch-optimize', help='Batch structure optimization')
    batch_parser.add_argument('input', nargs='+', help='Input structure files (glob patterns)')
    batch_parser.add_argument('--output-dir', required=True, help='Output directory')
    batch_parser.add_argument('--fmax', type=float, default=0.05, help='Force convergence')
    batch_parser.add_argument('--relax-cell', action='store_true', help='Optimize cell')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Initialize eSEN
    print(f"Initializing eSEN model: {args.model}")
    esen = ESENInference(
        model_name=args.model,
        device=args.device,
        precision=args.precision
    )
    
    # Execute command
    if args.command == 'single-point':
        cmd_single_point(esen, args)
    elif args.command == 'optimize':
        cmd_optimize(esen, args)
    elif args.command == 'md':
        cmd_md(esen, args)
    elif args.command == 'phonon':
        cmd_phonon(esen, args)
    elif args.command == 'bulk-modulus':
        cmd_bulk_modulus(esen, args)
    elif args.command == 'batch-optimize':
        cmd_batch_optimize(esen, args)


def cmd_single_point(esen, args):
    """Single-point calculation command."""
    print(f"Reading structure: {args.input}")
    atoms = read_structure(args.input)
    
    print("Calculating energy, forces, and stress...")
    result = esen.single_point(atoms)
    
    print("\n=== Results ===")
    print(f"Energy: {result['energy']:.6f} eV")
    print(f"Energy per atom: {result['energy_per_atom']:.6f} eV/atom")
    print(f"Max force: {result['max_force']:.6f} eV/Å")
    print(f"RMS force: {result['rms_force']:.6f} eV/Å")
    if result.get('pressure') is not None:
        print(f"Pressure: {result['pressure']:.4f} GPa")
    
    if args.output:
        # Save to JSON (convert numpy arrays to lists)
        output_data = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                      for k, v in result.items()}
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")


def cmd_optimize(esen, args):
    """Structure optimization command."""
    print(f"Reading structure: {args.input}")
    atoms = read_structure(args.input)
    
    print(f"Optimizing structure (fmax={args.fmax} eV/Å, relax_cell={args.relax_cell})...")
    result = esen.optimize(
        atoms,
        fmax=args.fmax,
        optimizer=args.optimizer,
        relax_cell=args.relax_cell,
        max_steps=args.max_steps,
        trajectory=args.trajectory
    )
    
    print("\n=== Results ===")
    print(f"Converged: {result['converged']}")
    print(f"Steps: {result['steps']}")
    print(f"Initial energy: {result['initial_energy']:.6f} eV")
    print(f"Final energy: {result['final_energy']:.6f} eV")
    print(f"Energy change: {result['energy_change']:.6f} eV")
    print(f"Final fmax: {result['final_fmax']:.6f} eV/Å")
    
    if args.output:
        write_structure(result['atoms'], args.output)
        print(f"\n✓ Optimized structure saved to {args.output}")


def cmd_md(esen, args):
    """Molecular dynamics command."""
    print(f"Reading structure: {args.input}")
    atoms = read_structure(args.input)
    
    print(f"Running MD ({args.ensemble}, T={args.temperature} K, {args.steps} steps)...")
    final_atoms = esen.run_md(
        atoms,
        temperature=args.temperature,
        pressure=args.pressure,
        steps=args.steps,
        timestep=args.timestep,
        ensemble=args.ensemble,
        trajectory=args.trajectory,
        logfile=args.logfile
    )
    
    print("\n=== MD Completed ===")
    print(f"Final temperature: {final_atoms.get_temperature():.2f} K")
    if atoms.pbc.any():
        print(f"Final volume: {final_atoms.get_volume():.2f} Å³")
    print(f"Trajectory saved to: {args.trajectory}")
    print(f"Log saved to: {args.logfile}")


def cmd_phonon(esen, args):
    """Phonon calculation command."""
    print(f"Reading primitive cell: {args.input}")
    atoms = read_structure(args.input)
    
    print(f"Calculating phonons (supercell={args.supercell}, mesh={args.mesh})...")
    result = esen.phonon(
        atoms,
        supercell_matrix=args.supercell,
        mesh=args.mesh
    )
    
    print("\n=== Results ===")
    if result['has_imaginary']:
        print(f"⚠ Warning: {result['imaginary_modes']} imaginary modes detected!")
    else:
        print("✓ No imaginary modes (structure is dynamically stable)")
    
    # Plot DOS
    from esen_inference.tasks.phonon import plot_phonon_dos
    plot_phonon_dos(
        result['frequency_points'],
        result['total_dos'],
        output=args.output
    )
    print(f"\n✓ Phonon DOS plot saved to {args.output}")


def cmd_bulk_modulus(esen, args):
    """Bulk modulus calculation command."""
    print(f"Reading structure: {args.input}")
    atoms = read_structure(args.input)
    
    print(f"Calculating bulk modulus (strain_range=±{args.strain_range})...")
    result = esen.bulk_modulus(
        atoms,
        strain_range=args.strain_range,
        n_points=args.n_points
    )
    
    print("\n=== Results ===")
    print(f"Bulk modulus: {result['bulk_modulus']:.2f} GPa")
    print(f"Equilibrium volume: {result['equilibrium_volume']:.3f} Å³")
    
    if args.output:
        output_data = {
            'bulk_modulus_GPa': result['bulk_modulus'],
            'equilibrium_volume_A3': result['equilibrium_volume'],
            'equilibrium_energy_eV': result['equilibrium_energy']
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")


def cmd_batch_optimize(esen, args):
    """Batch optimization command."""
    from glob import glob
    
    # Expand glob patterns
    input_files = []
    for pattern in args.input:
        input_files.extend(glob(pattern))
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(input_files)} structures to optimize")
    print(f"Output directory: {output_dir}")
    
    results_summary = []
    
    for i, input_file in enumerate(input_files, 1):
        filename = Path(input_file).name
        print(f"\n[{i}/{len(input_files)}] Processing: {filename}")
        
        try:
            atoms = read_structure(input_file)
            result = esen.optimize(
                atoms,
                fmax=args.fmax,
                relax_cell=args.relax_cell,
                max_steps=500
            )
            
            if result['converged']:
                output_file = output_dir / filename
                write_structure(result['atoms'], output_file)
                print(f"  ✓ Converged in {result['steps']} steps")
                print(f"  Energy change: {result['energy_change']:.6f} eV")
                
                results_summary.append({
                    'file': filename,
                    'converged': True,
                    'steps': result['steps'],
                    'energy': result['final_energy'],
                    'fmax': result['final_fmax']
                })
            else:
                print(f"  ✗ Did not converge after {result['steps']} steps")
                results_summary.append({
                    'file': filename,
                    'converged': False,
                    'steps': result['steps']
                })
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results_summary.append({
                'file': filename,
                'converged': False,
                'error': str(e)
            })
    
    # Save summary
    summary_file = output_dir / 'optimization_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    converged_count = sum(1 for r in results_summary if r.get('converged', False))
    print(f"\n=== Batch Optimization Complete ===")
    print(f"Total structures: {len(input_files)}")
    print(f"Converged: {converged_count}")
    print(f"Success rate: {converged_count/len(input_files)*100:.1f}%")
    print(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()
