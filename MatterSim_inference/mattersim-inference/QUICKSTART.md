# MatterSim Inference Quick Start

> Get started with MatterSim inference in 5 minutes

## 1. Installation

```bash
pip install mattersim-inference
```

## 2. Verify Installation

```bash
mattersim-infer --help
```

## 3. First Calculation

### Python

```python
from mattersim_inference import MatterSimInference

# Initialize
calc = MatterSimInference(model_name="MatterSim-v1-5M", device="auto")

# Single-point calculation
result = calc.single_point("your_structure.cif")
print(f"Energy: {result['energy']:.4f} eV")
```

### Command Line

```bash
mattersim-infer single-point your_structure.cif
```

## 4. Core Features

### 4.1 Single-Point Calculation

```python
result = calc.single_point(atoms)
# Returns: energy, forces, stress, pressure
```

### 4.2 Structure Optimization

```python
result = calc.optimize(atoms, fmax=0.05, optimize_cell=True)
# Returns: converged, atoms, final_energy
```

### 4.3 Molecular Dynamics

```python
final = calc.run_md(atoms, ensemble="nvt", temperature=300, steps=10000)
```

### 4.4 Adsorption Energy (MatterSim's Strength)

```python
result = calc.adsorption_energy(mof, "CO2", [10, 10, 10])
print(f"Adsorption Energy: {result['E_ads']:.4f} eV")
```

## 5. Common Options

| Option | Description |
|------|------|
| `model_name` | MatterSim-v1-1M, MatterSim-v1-5M |
| `device` | auto, cuda, cpu |
| `fmax` | Force convergence threshold (eV/Å) |

## 6. Next Steps

- Check [examples/](examples/) for more examples
- Read [Full Documentation](../MatterSim_inference_tasks.md)
