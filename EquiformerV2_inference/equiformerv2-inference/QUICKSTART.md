# EquiformerV2 Inference Quick Start

> Get started with EquiformerV2 inference in 5 minutes

## 1. Installation

```bash
pip install equiformerv2-inference
```

## 2. Verify Installation

```bash
equiformerv2-infer --help
```

## 3. First Calculation

### Python

```python
from equiformerv2_inference import EquiformerV2Inference

# Initialize
calc = EquiformerV2Inference(model_name="EquiformerV2-31M-S2EF", device="auto")

# Single-point calculation
result = calc.single_point("your_structure.cif")
print(f"Energy: {result['energy']:.4f} eV")
print(f"Max Force: {result['max_force']:.4f} eV/Å")
```

### Command Line

```bash
equiformerv2-infer single-point your_structure.cif
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

### 4.4 Phonon Calculation

```python
result = calc.calculate_phonon(atoms, supercell=[2, 2, 2])
print(f"Zero-point Energy: {result['ZPE']:.4f} eV")
```

## 5. Common Options

| Option | Description |
|------|------|
| `model_name` | EquiformerV2-31M-S2EF, EquiformerV2-153M-S2EF |
| `device` | auto, cuda, cpu |
| `fmax` | Force convergence threshold (eV/Å) |

## 6. Next Steps

- Check [examples/](examples/) for more examples
- Read [Full Documentation](../EquiformerV2_inference_tasks.md)
