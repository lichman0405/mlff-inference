"""Command-line interface for MACE inference"""

import click
import sys
from pathlib import Path


@click.group()
@click.version_option()
def main():
    """MACE Inference - High-level CLI for MACE force field calculations."""
    pass


@main.command()
@click.argument('structure', type=click.Path(exists=True))
@click.option('--model', default='medium', help='MACE model (small/medium/large or path)')
@click.option('--device', default='auto', type=click.Choice(['auto', 'cpu', 'cuda']))
@click.option('--d3/--no-d3', default=False, help='Enable D3 dispersion correction')
def energy(structure, model, device, d3):
    """Calculate single-point energy, forces, and stress."""
    from mace_inference import MACEInference
    
    click.echo(f"Loading structure: {structure}")
    calc = MACEInference(model=model, device=device, enable_d3=d3)
    
    result = calc.single_point(structure)
    
    click.echo("\n=== Single-Point Energy Results ===")
    click.echo(f"Total Energy:     {result['energy']:.6f} eV")
    click.echo(f"Energy per atom:  {result['energy_per_atom']:.6f} eV/atom")
    click.echo(f"Max Force:        {result['max_force']:.6f} eV/Å")
    click.echo(f"RMS Force:        {result['rms_force']:.6f} eV/Å")
    if result['pressure_GPa'] is not None:
        click.echo(f"Pressure:         {result['pressure_GPa']:.4f} GPa")


@main.command()
@click.argument('structure', type=click.Path(exists=True))
@click.option('--model', default='medium', help='MACE model')
@click.option('--device', default='auto', type=click.Choice(['auto', 'cpu', 'cuda']))
@click.option('--d3/--no-d3', default=False, help='Enable D3 correction')
@click.option('--fmax', default=0.05, type=float, help='Force convergence (eV/Å)')
@click.option('--steps', default=500, type=int, help='Max optimization steps')
@click.option('--optimizer', default='LBFGS', type=click.Choice(['LBFGS', 'BFGS', 'FIRE']))
@click.option('--cell/--no-cell', default=False, help='Optimize cell parameters')
@click.option('--output', type=click.Path(), help='Output structure file')
@click.option('--trajectory', type=click.Path(), help='Trajectory file')
def optimize(structure, model, device, d3, fmax, steps, optimizer, cell, output, trajectory):
    """Optimize atomic structure."""
    from mace_inference import MACEInference
    
    click.echo(f"Optimizing structure: {structure}")
    calc = MACEInference(model=model, device=device, enable_d3=d3)
    
    optimized = calc.optimize(
        structure,
        fmax=fmax,
        steps=steps,
        optimizer=optimizer,
        optimize_cell=cell,
        trajectory=trajectory,
        output=output
    )
    
    click.echo(f"\n✓ Optimization completed")
    if output:
        click.echo(f"✓ Saved to: {output}")


@main.command()
@click.argument('structure', type=click.Path(exists=True))
@click.option('--model', default='medium', help='MACE model')
@click.option('--device', default='auto', type=click.Choice(['auto', 'cpu', 'cuda']))
@click.option('--d3/--no-d3', default=False, help='Enable D3 correction')
@click.option('--ensemble', default='nvt', type=click.Choice(['nvt', 'npt']))
@click.option('--temp', default=300, type=float, help='Temperature (K)')
@click.option('--pressure', default=None, type=float, help='Pressure for NPT (GPa)')
@click.option('--steps', default=1000, type=int, help='Number of MD steps')
@click.option('--timestep', default=1.0, type=float, help='Time step (fs)')
@click.option('--trajectory', type=click.Path(), help='Trajectory file')
@click.option('--logfile', type=click.Path(), help='Log file')
def md(structure, model, device, d3, ensemble, temp, pressure, steps, timestep, trajectory, logfile):
    """Run molecular dynamics simulation."""
    from mace_inference import MACEInference
    
    click.echo(f"Running {ensemble.upper()} MD: {structure}")
    click.echo(f"Temperature: {temp} K, Steps: {steps}, Timestep: {timestep} fs")
    if ensemble == 'npt':
        pressure_val = pressure if pressure is not None else 0.0
        click.echo(f"Pressure: {pressure_val} GPa")
    
    calc = MACEInference(model=model, device=device, enable_d3=d3)
    
    final_atoms = calc.run_md(
        structure,
        ensemble=ensemble,
        temperature_K=temp,
        pressure_GPa=pressure,
        steps=steps,
        timestep=timestep,
        trajectory=trajectory,
        logfile=logfile
    )
    
    click.echo(f"\n✓ MD simulation completed")
    if trajectory:
        click.echo(f"✓ Trajectory saved: {trajectory}")


@main.command()
@click.argument('structure', type=click.Path(exists=True))
@click.option('--model', default='medium', help='MACE model')
@click.option('--device', default='auto', type=click.Choice(['auto', 'cpu', 'cuda']))
@click.option('--supercell', nargs=3, type=int, default=[2, 2, 2], help='Supercell size')
@click.option('--temp-range', nargs=3, type=float, help='Temperature range (min max step)')
@click.option('--mesh', nargs=3, type=int, default=[20, 20, 20], help='k-point mesh')
@click.option('--output-dir', type=click.Path(), help='Output directory')
def phonon(structure, model, device, supercell, temp_range, mesh, output_dir):
    """Calculate phonon properties."""
    from mace_inference import MACEInference
    
    click.echo(f"Calculating phonons: {structure}")
    click.echo(f"Supercell: {supercell}, Mesh: {mesh}")
    
    calc = MACEInference(model=model, device=device)
    
    temperature_range = tuple(temp_range) if temp_range else None
    
    result = calc.phonon(
        structure,
        supercell_matrix=list(supercell),
        mesh=list(mesh),
        temperature_range=temperature_range,
        output_dir=output_dir
    )
    
    click.echo(f"\n✓ Phonon calculation completed")
    
    if 'thermal_properties' in result:
        thermal = result['thermal_properties']
        click.echo("\n=== Thermal Properties ===")
        click.echo(f"Temperature range: {thermal['temperatures'][0]:.1f} - {thermal['temperatures'][-1]:.1f} K")
        # Print properties at 300 K if available
        idx_300 = np.argmin(np.abs(thermal['temperatures'] - 300))
        click.echo(f"\nAt 300 K:")
        click.echo(f"  Free Energy: {thermal['free_energy'][idx_300]:.3f} kJ/mol")
        click.echo(f"  Entropy:     {thermal['entropy'][idx_300]:.3f} J/(mol·K)")
        click.echo(f"  Heat Capacity: {thermal['heat_capacity'][idx_300]:.3f} J/(mol·K)")


@main.command(name='bulk-modulus')
@click.argument('structure', type=click.Path(exists=True))
@click.option('--model', default='medium', help='MACE model')
@click.option('--device', default='auto', type=click.Choice(['auto', 'cpu', 'cuda']))
@click.option('--d3/--no-d3', default=False, help='Enable D3 correction')
@click.option('--points', default=11, type=int, help='Number of volume points')
def bulk_modulus(structure, model, device, d3, points):
    """Calculate bulk modulus."""
    from mace_inference import MACEInference
    
    click.echo(f"Calculating bulk modulus: {structure}")
    calc = MACEInference(model=model, device=device, enable_d3=d3)
    
    result = calc.bulk_modulus(structure, n_points=points)
    
    click.echo("\n=== Bulk Modulus Results ===")
    click.echo(f"Equilibrium Volume: {result['v0']:.3f} Å³")
    click.echo(f"Equilibrium Energy: {result['e0']:.6f} eV")
    click.echo(f"Bulk Modulus:       {result['B_GPa']:.2f} GPa")


@main.command()
@click.argument('mof', type=click.Path(exists=True))
@click.option('--model', default='medium', help='MACE model')
@click.option('--device', default='auto', type=click.Choice(['auto', 'cpu', 'cuda']))
@click.option('--d3/--no-d3', default=True, help='Enable D3 correction (recommended)')
@click.option('--gas', required=True, help='Gas molecule (e.g., CO2, H2O, CH4)')
@click.option('--site', nargs=3, type=float, required=True, help='Adsorption site (x y z)')
@click.option('--optimize/--no-optimize', default=True, help='Optimize adsorption complex')
def adsorption(mof, model, device, d3, gas, site, optimize):
    """Calculate gas adsorption energy."""
    from mace_inference import MACEInference
    
    click.echo(f"Calculating {gas} adsorption in: {mof}")
    click.echo(f"Adsorption site: {site}")
    
    calc = MACEInference(model=model, device=device, enable_d3=d3)
    
    result = calc.adsorption_energy(
        mof,
        gas_molecule=gas,
        site_position=list(site),
        optimize_complex=optimize
    )
    
    click.echo("\n=== Adsorption Energy Results ===")
    click.echo(f"E_ads = {result['E_ads']:.4f} eV ({result['E_ads'] * 96.485:.2f} kJ/mol)")
    click.echo(f"E_MOF = {result['E_mof']:.4f} eV")
    click.echo(f"E_gas = {result['E_gas']:.4f} eV")
    click.echo(f"E_complex = {result['E_complex']:.4f} eV")


@main.command()
@click.option('--verbose', is_flag=True, help='Show detailed device information')
def info(verbose):
    """Show MACE inference environment information."""
    from mace_inference.utils.device import get_device_info
    from mace_inference.utils.d3_correction import check_d3_available
    import mace_inference
    
    click.echo("=== MACE Inference Information ===\n")
    click.echo(f"Version: {mace_inference.__version__}")
    
    device_info = get_device_info()
    click.echo(f"\nPyTorch Version: {device_info['pytorch_version']}")
    click.echo(f"CUDA Available: {device_info['cuda_available']}")
    
    if device_info['cuda_available']:
        click.echo(f"CUDA Version: {device_info['cuda_version']}")
        click.echo(f"GPU Count: {device_info['cuda_count']}")
        
        if verbose:
            click.echo("\nGPU Devices:")
            for dev in device_info['devices']:
                click.echo(f"  [{dev['index']}] {dev['name']} ({dev['memory']})")
    
    d3_available = check_d3_available()
    click.echo(f"\nD3 Correction Available: {d3_available}")
    
    if not d3_available:
        click.echo("  → Install with: pip install mace-inference[d3]")


if __name__ == '__main__':
    import numpy as np  # Import here for phonon command
    main()
