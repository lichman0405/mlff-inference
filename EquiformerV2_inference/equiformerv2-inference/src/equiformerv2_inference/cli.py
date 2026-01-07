"""Command-line interface for EquiformerV2 inference"""

import argparse
import sys
from pathlib import Path


def create_parser():
    """Create argument parser for EquiformerV2 CLI."""
    parser = argparse.ArgumentParser(
        description="EquiformerV2 Inference - High-level CLI for EquiformerV2 force field calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single-point energy calculation
    sp_parser = subparsers.add_parser(
        'single-point',
        help='Calculate single-point energy, forces, and stress'
    )
    sp_parser.add_argument('structure', type=str, help='Structure file path')
    sp_parser.add_argument('--model', default='equiformer_v2', help='EquiformerV2 model name or path')
    sp_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                          help='Computing device')
    sp_parser.add_argument('--output', type=str, help='Output file for results')
    
    # Structure optimization
    opt_parser = subparsers.add_parser(
        'optimize',
        help='Optimize atomic structure'
    )
    opt_parser.add_argument('structure', type=str, help='Structure file path')
    opt_parser.add_argument('--model', default='equiformer_v2', help='EquiformerV2 model')
    opt_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                           help='Computing device')
    opt_parser.add_argument('--fmax', default=0.05, type=float,
                           help='Force convergence criterion (eV/Å)')
    opt_parser.add_argument('--steps', default=500, type=int,
                           help='Maximum optimization steps')
    opt_parser.add_argument('--optimizer', default='LBFGS', choices=['LBFGS', 'BFGS', 'FIRE'],
                           help='Optimization algorithm')
    opt_parser.add_argument('--cell', action='store_true',
                           help='Optimize cell parameters')
    opt_parser.add_argument('--output', type=str, help='Output structure file')
    opt_parser.add_argument('--trajectory', type=str, help='Trajectory file path')
    
    # Molecular dynamics
    md_parser = subparsers.add_parser(
        'md',
        help='Run molecular dynamics simulation'
    )
    md_parser.add_argument('structure', type=str, help='Structure file path')
    md_parser.add_argument('--model', default='equiformer_v2', help='EquiformerV2 model')
    md_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                          help='Computing device')
    md_parser.add_argument('--ensemble', default='nvt', choices=['nve', 'nvt', 'npt'],
                          help='MD ensemble')
    md_parser.add_argument('--temp', default=300, type=float,
                          help='Temperature (K)')
    md_parser.add_argument('--pressure', type=float,
                          help='Pressure for NPT (GPa)')
    md_parser.add_argument('--steps', default=1000, type=int,
                          help='Number of MD steps')
    md_parser.add_argument('--timestep', default=1.0, type=float,
                          help='Time step (fs)')
    md_parser.add_argument('--trajectory', type=str, help='Trajectory file path')
    md_parser.add_argument('--logfile', type=str, help='Log file path')
    
    # Phonon calculation
    phonon_parser = subparsers.add_parser(
        'phonon',
        help='Calculate phonon properties'
    )
    phonon_parser.add_argument('structure', type=str, help='Structure file path')
    phonon_parser.add_argument('--model', default='equiformer_v2', help='EquiformerV2 model')
    phonon_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                              help='Computing device')
    phonon_parser.add_argument('--supercell', nargs=3, type=int, default=[2, 2, 2],
                              help='Supercell size (nx ny nz)')
    phonon_parser.add_argument('--mesh', nargs=3, type=int, default=[20, 20, 20],
                              help='k-point mesh (nx ny nz)')
    phonon_parser.add_argument('--temp-range', nargs=3, type=float,
                              help='Temperature range (min max step) in K')
    phonon_parser.add_argument('--output-dir', type=str, help='Output directory')
    
    # Bulk modulus calculation
    bulk_parser = subparsers.add_parser(
        'bulk-modulus',
        help='Calculate bulk modulus'
    )
    bulk_parser.add_argument('structure', type=str, help='Structure file path')
    bulk_parser.add_argument('--model', default='equiformer_v2', help='EquiformerV2 model')
    bulk_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                            help='Computing device')
    bulk_parser.add_argument('--points', default=11, type=int,
                            help='Number of volume points')
    bulk_parser.add_argument('--strain-range', default=0.05, type=float,
                            help='Strain range (±)')
    bulk_parser.add_argument('--eos', default='birchmurnaghan',
                            choices=['birchmurnaghan', 'vinet', 'murnaghan', 'sjeos'],
                            help='Equation of state type')
    bulk_parser.add_argument('--output', type=str, help='Output file for results')
    
    # Batch optimization
    batch_parser = subparsers.add_parser(
        'batch-optimize',
        help='Batch optimize multiple structures'
    )
    batch_parser.add_argument('input_dir', type=str,
                             help='Directory containing structure files')
    batch_parser.add_argument('--model', default='equiformer_v2', help='EquiformerV2 model')
    batch_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                             help='Computing device')
    batch_parser.add_argument('--fmax', default=0.05, type=float,
                             help='Force convergence criterion (eV/Å)')
    batch_parser.add_argument('--steps', default=500, type=int,
                             help='Maximum optimization steps')
    batch_parser.add_argument('--optimizer', default='LBFGS',
                             choices=['LBFGS', 'BFGS', 'FIRE'],
                             help='Optimization algorithm')
    batch_parser.add_argument('--cell', action='store_true',
                             help='Optimize cell parameters')
    batch_parser.add_argument('--output-dir', type=str, required=True,
                             help='Output directory')
    batch_parser.add_argument('--pattern', default='*.cif',
                             help='File pattern to match (e.g., *.cif, *.vasp)')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Import here to avoid slow startup
    from equiformerv2_inference import EquiformerV2Inference
    from equiformerv2_inference.utils.io import read_structure, write_structure
    
    # Execute commands
    if args.command == 'single-point':
        execute_single_point(args)
    elif args.command == 'optimize':
        execute_optimize(args)
    elif args.command == 'md':
        execute_md(args)
    elif args.command == 'phonon':
        execute_phonon(args)
    elif args.command == 'bulk-modulus':
        execute_bulk_modulus(args)
    elif args.command == 'batch-optimize':
        execute_batch_optimize(args)


def execute_single_point(args):
    """Execute single-point calculation."""
    from equiformerv2_inference import EquiformerV2Inference
    
    print(f"Loading structure: {args.structure}")
    calc = EquiformerV2Inference(model=args.model, device=args.device)
    
    result = calc.single_point(args.structure)
    
    print("\n" + "=" * 50)
    print("Single-Point Energy Results")
    print("=" * 50)
    print(f"Total Energy:     {result['energy']:.6f} eV")
    print(f"Energy per atom:  {result['energy_per_atom']:.6f} eV/atom")
    print(f"Max Force:        {result['max_force']:.6f} eV/Å")
    print(f"RMS Force:        {result['rms_force']:.6f} eV/Å")
    if result.get('pressure_GPa') is not None:
        print(f"Pressure:         {result['pressure_GPa']:.4f} GPa")
    print("=" * 50)
    
    if args.output:
        import json
        with open(args.output, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            result_json = {k: v.tolist() if hasattr(v, 'tolist') else v 
                          for k, v in result.items()}
            json.dump(result_json, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def execute_optimize(args):
    """Execute structure optimization."""
    from equiformerv2_inference import EquiformerV2Inference
    
    print(f"Optimizing structure: {args.structure}")
    calc = EquiformerV2Inference(model=args.model, device=args.device)
    
    optimized = calc.optimize(
        args.structure,
        fmax=args.fmax,
        steps=args.steps,
        optimizer=args.optimizer,
        optimize_cell=args.cell,
        trajectory=args.trajectory,
        output=args.output
    )
    
    print(f"\n{'=' * 50}")
    print("Optimization completed successfully")
    print("=" * 50)
    if args.output:
        print(f"Optimized structure saved to: {args.output}")
    if args.trajectory:
        print(f"Trajectory saved to: {args.trajectory}")


def execute_md(args):
    """Execute molecular dynamics simulation."""
    from equiformerv2_inference import EquiformerV2Inference
    
    print(f"Running {args.ensemble.upper()} MD: {args.structure}")
    print(f"Temperature: {args.temp} K, Steps: {args.steps}, Timestep: {args.timestep} fs")
    if args.ensemble == 'npt':
        pressure_val = args.pressure if args.pressure is not None else 0.0
        print(f"Pressure: {pressure_val} GPa")
    
    calc = EquiformerV2Inference(model=args.model, device=args.device)
    
    final_atoms = calc.run_md(
        args.structure,
        ensemble=args.ensemble,
        temperature_K=args.temp,
        pressure_GPa=args.pressure,
        steps=args.steps,
        timestep=args.timestep,
        trajectory=args.trajectory,
        logfile=args.logfile
    )
    
    print(f"\n{'=' * 50}")
    print("MD simulation completed successfully")
    print("=" * 50)
    if args.trajectory:
        print(f"Trajectory saved to: {args.trajectory}")
    if args.logfile:
        print(f"Log file saved to: {args.logfile}")


def execute_phonon(args):
    """Execute phonon calculation."""
    from equiformerv2_inference import EquiformerV2Inference
    import numpy as np
    
    print(f"Calculating phonons: {args.structure}")
    print(f"Supercell: {args.supercell}, Mesh: {args.mesh}")
    
    calc = EquiformerV2Inference(model=args.model, device=args.device)
    
    temperature_range = tuple(args.temp_range) if args.temp_range else None
    
    result = calc.phonon(
        args.structure,
        supercell_matrix=list(args.supercell),
        mesh=list(args.mesh),
        temperature_range=temperature_range,
        output_dir=args.output_dir
    )
    
    print(f"\n{'=' * 50}")
    print("Phonon calculation completed successfully")
    print("=" * 50)
    
    if 'thermal_properties' in result:
        thermal = result['thermal_properties']
        print("\nThermal Properties:")
        print(f"Temperature range: {thermal['temperatures'][0]:.1f} - {thermal['temperatures'][-1]:.1f} K")
        
        # Print properties at 300 K if available
        temps = thermal['temperatures']
        idx_300 = int(np.argmin(np.abs(temps - 300)))
        print(f"\nAt {temps[idx_300]:.1f} K:")
        print(f"  Free Energy:   {thermal['free_energy'][idx_300]:.3f} kJ/mol")
        print(f"  Entropy:       {thermal['entropy'][idx_300]:.3f} J/(mol·K)")
        print(f"  Heat Capacity: {thermal['heat_capacity'][idx_300]:.3f} J/(mol·K)")
    
    if args.output_dir:
        print(f"\nPhonon data saved to: {args.output_dir}")


def execute_bulk_modulus(args):
    """Execute bulk modulus calculation."""
    from equiformerv2_inference import EquiformerV2Inference
    
    print(f"Calculating bulk modulus: {args.structure}")
    calc = EquiformerV2Inference(model=args.model, device=args.device)
    
    result = calc.bulk_modulus(
        args.structure,
        strain_range=args.strain_range,
        npoints=args.points,
        eos=args.eos
    )
    
    print(f"\n{'=' * 50}")
    print("Bulk Modulus Results")
    print("=" * 50)
    print(f"Equilibrium Volume: {result['v0']:.3f} Ų")
    print(f"Equilibrium Energy: {result['e0']:.6f} eV")
    print(f"Bulk Modulus:       {result['bulk_modulus']:.2f} GPa")
    print(f"EOS Type:           {result['eos']}")
    print("=" * 50)
    
    if args.output:
        import json
        result_json = {k: v.tolist() if hasattr(v, 'tolist') else v 
                      for k, v in result.items()}
        with open(args.output, 'w') as f:
            json.dump(result_json, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def execute_batch_optimize(args):
    """Execute batch optimization."""
    from equiformerv2_inference import EquiformerV2Inference
    from equiformerv2_inference.utils.io import read_structure
    from pathlib import Path
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Find all structure files matching pattern
    structure_files = list(input_dir.glob(args.pattern))
    
    if not structure_files:
        print(f"Error: No files matching pattern '{args.pattern}' found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(structure_files)} structures to optimize")
    
    # Load all structures
    structures = []
    for f in structure_files:
        try:
            atoms = read_structure(f)
            structures.append(atoms)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
    
    print(f"Successfully loaded {len(structures)} structures")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize calculator
    calc = EquiformerV2Inference(model=args.model, device=args.device)
    
    # Batch optimize
    optimized = calc.batch_optimize(
        structures,
        fmax=args.fmax,
        steps=args.steps,
        optimizer=args.optimizer,
        optimize_cell=args.cell,
        output_dir=str(output_dir)
    )
    
    print(f"\n{'=' * 50}")
    print(f"Batch optimization completed: {len(optimized)} structures")
    print(f"Results saved to: {output_dir}")
    print("=" * 50)


if __name__ == '__main__':
    main()
