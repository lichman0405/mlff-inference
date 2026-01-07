# EquiformerV2 Inference - 推理任务指南

> **EquiformerV2**: MOFSimBench 排名 **#5** 的等变Transformer力场  
> **开发团队**: MIT & Meta AI - Liao & Smidt 2023  
> **特色**: E(3)等变性、Transformer架构、OCP大规模预训练

---

## 目录

1. [模型概述](#1-模型概述)
2. [任务1: 单点计算](#2-任务1-单点计算)
3. [任务2: 结构优化](#3-任务2-结构优化)
4. [任务3: 分子动力学](#4-任务3-分子动力学)
5. [任务4: 声子计算](#5-任务4-声子计算)
6. [任务5: 力学性质](#6-任务5-力学性质)
7. [任务6: 批量处理](#7-任务6-批量处理)
8. [任务7: 高级技巧](#8-任务7-高级技巧)

---

## 1. 模型概述

### 1.1 EquiformerV2 简介

**EquiformerV2** 是基于等变Transformer架构的新一代机器学习力场，由MIT和Meta AI联合开发。

**核心特点**:
- 🔬 **E(3)等变性**: 完全保持旋转和平移对称性
- 🧠 **Transformer架构**: 自注意力机制捕获长程相互作用
- 🌐 **OCP预训练**: 在Open Catalyst Project数据集上训练
- 📈 **可扩展性**: 支持31M到153M参数的多个模型规模
- ⚡ **计算效率**: 优化的实现，适合大规模计算

### 1.2 MOFSimBench 性能

| 指标 | EquiformerV2 | 排名 |
|------|--------------|------|
| 能量 MAE | 0.062 eV/atom | #5 |
| 力 MAE | 0.108 eV/Å | #5 |
| 应力预测 | 良好 | Top-5 |
| 计算速度 | 快速 | Top-5 |
| 可扩展性 | 优秀 | Top-3 |

### 1.3 适用场景

✅ **推荐使用**:
- MOF结构优化和性质预测
- 催化材料的吸附能计算
- 大规模高通量筛选
- 需要长程相互作用的体系

⚠️ **限制**:
- 计算成本高于简单的GNN模型
- 需要较大GPU显存（推荐16GB+）

---

## 2. 任务1: 单点计算

### 2.1 Python API

```python
from equiformerv2_inference import EquiformerV2Inference
from ase.io import read

# 初始化模型
calc = EquiformerV2Inference(
    model_name="EquiformerV2-31M-S2EF",
    device="cuda"
)

# 读取结构
atoms = read("MOF-5.cif")

# 单点计算
result = calc.single_point(atoms)

print(f"能量: {result['energy']:.6f} eV")
print(f"每原子能量: {result['energy_per_atom']:.6f} eV/atom")
print(f"最大力: {result['max_force']:.6f} eV/Å")
print(f"RMS力: {result['rms_force']:.6f} eV/Å")
print(f"压强: {result['pressure']:.4f} GPa")
```

### 2.2 命令行

```bash
# 基础单点计算
equiformerv2-infer single-point MOF-5.cif

# 保存结果到JSON
equiformerv2-infer single-point MOF-5.cif --output result.json

# 使用大模型
equiformerv2-infer single-point MOF-5.cif --model EquiformerV2-153M-S2EF

# 使用CPU
equiformerv2-infer single-point MOF-5.cif --device cpu
```

---

## 3. 任务2: 结构优化

### 3.1 位置优化

```python
# 仅优化原子位置
result = calc.optimize(
    atoms,
    fmax=0.01,           # 力收敛阈值 (eV/Å)
    max_steps=500,       # 最大步数
    optimize_cell=False, # 不优化晶胞
    optimizer="LBFGS"    # 优化器
)

print(f"收敛: {result['converged']}")
print(f"优化步数: {result['steps']}")
print(f"最终能量: {result['final_energy']:.6f} eV")

# 获取优化后的结构
optimized_atoms = result['atoms']
```

### 3.2 晶胞优化

```python
# 同时优化位置和晶胞
result = calc.optimize(
    atoms,
    fmax=0.01,
    optimize_cell=True,  # 优化晶胞
    output_file="optimized.cif"
)

print(f"初始体积: {atoms.get_volume():.2f} Å³")
print(f"最终体积: {result['atoms'].get_volume():.2f} Å³")
```

### 3.3 命令行

```bash
# 位置优化
equiformerv2-infer optimize MOF.cif --fmax 0.01 --output opt.cif

# 晶胞优化
equiformerv2-infer optimize MOF.cif --fmax 0.01 --cell

# 使用FIRE优化器
equiformerv2-infer optimize MOF.cif --fmax 0.05 --optimizer FIRE
```

---

## 4. 任务3: 分子动力学

### 4.1 NVT 系综

```python
# NVT模拟 (恒温恒容)
final_atoms = calc.run_md(
    atoms,
    ensemble="nvt",
    temperature=300,      # K
    timestep=1.0,         # fs
    steps=50000,          # MD步数
    trajectory_file="md.traj",
    logfile="md.log",
    log_interval=100
)
```

### 4.2 NPT 系综

```python
# NPT模拟 (恒温恒压)
final_atoms = calc.run_md(
    atoms,
    ensemble="npt",
    temperature=300,      # K
    pressure=0.0,         # GPa (1 atm ≈ 0.0001 GPa)
    timestep=1.0,
    steps=100000,
    trajectory_file="npt.traj"
)
```

### 4.3 命令行

```bash
# NVT模拟
equiformerv2-infer md MOF.cif --ensemble nvt --temp 300 --steps 50000

# NPT模拟
equiformerv2-infer md MOF.cif --ensemble npt --temp 300 --pressure 0.0001

# 高温稳定性测试
equiformerv2-infer md MOF.cif --ensemble nvt --temp 500 --steps 100000
```

---

## 5. 任务4: 声子计算

### 5.1 声子DOS

```python
result = calc.calculate_phonon(
    atoms,
    supercell=[2, 2, 2],
    mesh=[20, 20, 20],
    temperature_range=(0, 500, 50)
)

print(f"零点能: {result['ZPE']:.4f} eV")
print(f"300K自由能: {result['free_energy'][6]:.4f} eV")
print(f"300K熵: {result['entropy'][6]:.6f} eV/K")
print(f"300K热容: {result['Cv'][6]:.6f} eV/K")
```

### 5.2 命令行

```bash
# 声子计算
equiformerv2-infer phonon MOF.cif --supercell 2 2 2 --output phonon.json

# 大超胞
equiformerv2-infer phonon MOF.cif --supercell 3 3 3 --mesh 30 30 30
```

---

## 6. 任务5: 力学性质

### 6.1 体模量

```python
result = calc.calculate_bulk_modulus(
    atoms,
    strain_range=0.05,  # ±5%应变
    npoints=11          # 应变点数
)

print(f"体模量: {result['bulk_modulus']:.2f} GPa")
print(f"平衡体积: {result['V0']:.2f} Å³")
print(f"平衡能量: {result['E0']:.6f} eV")
```

### 6.2 命令行

```bash
equiformerv2-infer bulk-modulus MOF.cif --output bulk_modulus.json
```

---

## 7. 任务6: 批量处理

### 7.1 Python批量优化

```python
from pathlib import Path

structures = list(Path("structures").glob("*.cif"))

for cif_file in structures:
    print(f"处理: {cif_file.name}")
    
    result = calc.optimize(
        str(cif_file),
        fmax=0.01,
        output_file=f"optimized/{cif_file.name}"
    )
    
    print(f"  收敛: {result['converged']}")
    print(f"  能量: {result['final_energy']:.6f} eV")
```

### 7.2 命令行批量处理

```bash
# 批量优化
equiformerv2-infer batch-optimize structures/*.cif --output-dir optimized/

# 指定参数
equiformerv2-infer batch-optimize *.cif --fmax 0.01 --cell
```

---

## 8. 任务7: 高级技巧

### 8.1 模型选择

```python
# 快速测试：使用31M模型
calc_fast = EquiformerV2Inference(
    model_name="EquiformerV2-31M-S2EF",
    device="cuda"
)

# 高精度：使用153M模型
calc_accurate = EquiformerV2Inference(
    model_name="EquiformerV2-153M-S2EF",
    device="cuda"
)
```

### 8.2 GPU内存优化

```python
# 减小批大小
calc = EquiformerV2Inference(
    model_name="EquiformerV2-31M-S2EF",
    device="cuda",
    batch_size=16  # 默认32
)

# 使用混合精度
import torch
torch.backends.cuda.matmul.allow_tf32 = True
```

### 8.3 多GPU并行

```python
import multiprocessing as mp

def optimize_structure(gpu_id, cif_file):
    calc = EquiformerV2Inference(device=f"cuda:{gpu_id}")
    result = calc.optimize(cif_file, fmax=0.01)
    return result

# 4个GPU并行
with mp.Pool(4) as pool:
    results = pool.starmap(
        optimize_structure,
        [(i % 4, f) for i, f in enumerate(cif_files)]
    )
```

### 8.4 ASE计算器集成

```python
from ase.optimize import BFGS

# 获取ASE计算器
ase_calc = calc.get_calculator()
atoms.calc = ase_calc

# 直接使用ASE功能
opt = BFGS(atoms)
opt.run(fmax=0.01)

# 或用于MD
from ase.md.langevin import Langevin
from ase import units

dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=300, friction=0.01)
dyn.run(10000)
```

---

## 9. 性能对比

| 模型 | 参数量 | 推理速度 | 内存需求 | 适用场景 |
|------|--------|---------|---------|---------|
| EquiformerV2-31M | 31M | 快 | 8-12GB | 快速筛选 |
| EquiformerV2-153M | 153M | 中等 | 16-24GB | 高精度计算 |

---

## 10. 常见问题

**Q1: EquiformerV2与SevenNet有何区别？**
- EquiformerV2使用Transformer架构，SevenNet使用标准GNN
- EquiformerV2在催化材料上训练，SevenNet更通用
- EquiformerV2计算成本更高，但可能更准确

**Q2: 如何选择模型大小？**
- 31M: 快速测试、大规模筛选
- 153M: 需要高精度的生产计算

**Q3: GPU显存不足怎么办？**
- 使用31M模型
- 减小批大小
- 使用CPU（会很慢）

---

## 参考文献

```bibtex
@article{liao2023equiformerv2,
  title={EquiformerV2: Improved Equivariant Transformer for Scalable and Accurate Interatomic Potentials},
  author={Liao, Yi-Lun and Smidt, Tess},
  journal={arXiv preprint arXiv:2306.12059},
  year={2023}
}
```

---

**相关文档**:
- [API 参考](EquiformerV2_inference_API_reference.md)
- [安装指南](INSTALL.md)
