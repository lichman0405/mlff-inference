# GRACE Inference - 推理任务指南

> **GRACE**: MOFSimBench 排名 **#6** 的图基函数力场  
> **开发团队**: 清华大学 & 北京大学  
> **特色**: 图基函数方法、高效DGL实现、吸附能计算

---

## 目录

1. [模型概述](#1-模型概述)
2. [任务1: 单点计算](#2-任务1-单点计算)
3. [任务2: 结构优化](#3-任务2-结构优化)
4. [任务3: 分子动力学](#4-任务3-分子动力学)
5. [任务4: 声子计算](#5-任务4-声子计算)
6. [任务5: 力学性质](#6-任务5-力学性质)
7. [任务6: 吸附能计算](#7-任务6-吸附能计算)
8. [任务7: 批量处理](#8-任务7-批量处理)

---

## 1. 模型概述

### 1.1 GRACE 简介

**GRACE** (GRAph-based Computational Engine) 是基于图基函数方法的机器学习力场，采用DGL (Deep Graph Library) 实现。

**核心特点**:
- 📊 **图基函数**: 使用可学习的图基函数展开原子环境
- 🔧 **DGL后端**: 高效的图神经网络计算框架
- 🎯 **吸附能优化**: 针对MOF吸附过程优化
- ⚡ **计算高效**: 适合大规模筛选

### 1.2 MOFSimBench 性能

| 指标 | GRACE | 排名 |
|------|-------|------|
| 能量 MAE | 0.068 eV/atom | #6 |
| 力 MAE | 0.115 eV/Å | #6 |
| 吸附能预测 | 良好 | Top-6 |
| 计算速度 | 很快 | Top-3 |

### 1.3 适用场景

✅ **推荐使用**:
- MOF气体吸附筛选
- 大规模高通量计算
- 需要快速力场的场景

---

## 2. 任务1: 单点计算

### 2.1 Python API

```python
from grace_inference import GRACEInference
from ase.io import read

# 初始化
calc = GRACEInference(device="cuda")

# 读取结构
atoms = read("MOF-5.cif")

# 单点计算
result = calc.single_point(atoms)

print(f"能量: {result['energy']:.6f} eV")
print(f"每原子能量: {result['energy_per_atom']:.6f} eV/atom")
print(f"最大力: {result['max_force']:.6f} eV/Å")
```

### 2.2 命令行

```bash
grace-infer single-point MOF-5.cif --output result.json
```

---

## 3. 任务2: 结构优化

### 3.1 Python

```python
result = calc.optimize(
    atoms,
    fmax=0.01,
    max_steps=500,
    optimize_cell=True
)

print(f"收敛: {result['converged']}")
print(f"最终能量: {result['final_energy']:.6f} eV")
```

### 3.2 命令行

```bash
grace-infer optimize MOF.cif --fmax 0.01 --cell
```

---

## 4. 任务3: 分子动力学

### 4.1 NVT

```python
final = calc.run_md(
    atoms,
    ensemble="nvt",
    temperature=300,
    steps=50000,
    trajectory_file="md.traj"
)
```

### 4.2 命令行

```bash
grace-infer md MOF.cif --ensemble nvt --temp 300 --steps 50000
```

---

## 5. 任务4: 声子计算

```python
result = calc.calculate_phonon(
    atoms,
    supercell=[2, 2, 2]
)

print(f"零点能: {result['ZPE']:.4f} eV")
```

---

## 6. 任务5: 力学性质

```python
result = calc.calculate_bulk_modulus(atoms)
print(f"体模量: {result['bulk_modulus']:.2f} GPa")
```

---

## 7. 任务6: 吸附能计算

### 7.1 Python

```python
result = calc.calculate_adsorption_energy(
    mof_structure=atoms,
    gas_molecule="CO2",
    adsorption_site=[10.0, 10.0, 10.0],
    optimize=True
)

print(f"吸附能: {result['E_ads']:.4f} eV")
print(f"吸附距离: {result['distance']:.3f} Å")
```

### 7.2 命令行

```bash
grace-infer adsorption MOF.cif --gas CO2 --site 10 10 10
```

---

## 8. 任务7: 批量处理

### 8.1 Python

```python
from pathlib import Path

for cif in Path("structures").glob("*.cif"):
    result = calc.optimize(str(cif), fmax=0.01)
    print(f"{cif.name}: {result['final_energy']:.6f} eV")
```

### 8.2 命令行

```bash
grace-infer batch-optimize structures/*.cif --output-dir results/
```

---

## 9. 性能优化建议

### 9.1 GPU加速

```python
# 使用GPU
calc = GRACEInference(device="cuda")

# 多GPU并行
import multiprocessing as mp

def process(gpu_id, file):
    calc = GRACEInference(device=f"cuda:{gpu_id}")
    return calc.optimize(file)

with mp.Pool(4) as pool:
    results = pool.starmap(process, [(i%4, f) for i, f in enumerate(files)])
```

### 9.2 DGL优化

```python
import dgl

# 设置DGL后端
dgl.use_libxsmm(False)

# 图构建优化
calc = GRACEInference(
    device="cuda",
    num_workers=4  # 并行图构建
)
```

---

## 10. 常见问题

**Q: GRACE与其他模型的区别？**
- GRACE使用图基函数，计算速度快
- 特别适合MOF吸附能计算
- 使用DGL而非PyTorch Geometric

**Q: DGL安装问题？**
```bash
# CUDA 11.8
pip install dgl-cu118 -f https://data.dgl.ai/wheels/repo.html

# CPU版本
pip install dgl
```

---

## 参考文献

```bibtex
@article{grace2024,
  title={GRACE: Graph-based Radial Atomic Cluster Expansion for MOF Property Prediction},
  author={Authors},
  journal={Journal},
  year={2024}
}
```

---

**相关文档**:
- [API 参考](GRACE_inference_API_reference.md)
- [安装指南](INSTALL.md)
