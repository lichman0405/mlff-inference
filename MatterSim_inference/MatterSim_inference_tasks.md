# MatterSim Inference - 推理任务指南

> **MatterSim**: MOFSimBench 排名 **#3** 的通用机器学习力场  
> **开发团队**: Microsoft Research - Yang et al. 2024  
> **特色**: 不确定性估计、吸附能最佳、三体相互作用建模

---

## 目录

1. [模型概述](#1-模型概述)
2. [任务1: 单点计算](#2-任务1-单点计算)
3. [任务2: 结构优化](#3-任务2-结构优化)
4. [任务3: 分子动力学](#4-任务3-分子动力学)
5. [任务4: 声子计算](#5-任务4-声子计算)
6. [任务5: 力学性质](#6-任务5-力学性质)
7. [任务6: 吸附能](#7-任务6-吸附能)
8. [任务7: 配位分析](#8-任务7-配位分析)
9. [任务8: 高通量筛选](#9-任务8-高通量筛选)
10. [性能基准](#10-性能基准)

---

## 1. 模型概述

### 1.1 MatterSim 简介

MatterSim 是 Microsoft Research 开发的通用机器学习力场，基于 M3GNet 架构并引入不确定性感知的主动学习。

### 1.2 可用模型

| 模型名称 | 参数量 | 训练数据 | 推荐用途 |
|----------|--------|----------|----------|
| **MatterSim-v1-1M** | 1M | 专有数据集 | 快速测试 |
| **MatterSim-v1-5M** | 5M | 专有数据集 | **生产推荐** |

### 1.3 MOFSimBench 性能

| 指标 | MatterSim-v1 | 排名 |
|------|--------------|------|
| **能量 MAE** | 0.052 eV/atom | #3 |
| **力 MAE** | 0.095 eV/Å | #3 |
| **体模量 MAE** | 3.8 GPa | #4 |
| **优化成功率** | 85% | #4 |
| **热容 MAE** | 0.028 J/(K·g) | #4 |
| **吸附能** | **最佳** | **#1** 🥇 |
| **MD 稳定性** | **优异** | **#1** 🥇 |

### 1.4 核心特点

- ✅ **吸附能第一**: 主客体相互作用建模最佳
- ✅ **MD 稳定性第一**: 与 eSEN 并列最佳
- ✅ **不确定性估计**: 通过模型集成实现
- ✅ **三体相互作用**: 精确建模角度依赖

---

## 2. 任务1: 单点计算

### 2.1 概述

计算给定结构的能量、力和应力张量。

### 2.2 Python API

```python
from mattersim_inference import MatterSimInference
from ase.io import read

# 初始化模型
calc = MatterSimInference(model_name="MatterSim-v1-5M", device="cuda")

# 读取结构
atoms = read("MOF-5.cif")

# 单点计算
result = calc.single_point(atoms)

print(f"能量: {result['energy']:.6f} eV")
print(f"每原子能量: {result['energy_per_atom']:.6f} eV/atom")
print(f"最大力: {result['max_force']:.6f} eV/Å")
print(f"压强: {result['pressure']:.4f} GPa")
```

### 2.3 返回结果

| 键 | 类型 | 说明 |
|----|------|------|
| `energy` | float | 总能量 (eV) |
| `energy_per_atom` | float | 每原子能量 (eV/atom) |
| `forces` | ndarray | 力 (N, 3) (eV/Å) |
| `stress` | ndarray | 应力张量 (6,) (eV/Å³) |
| `max_force` | float | 最大力分量 (eV/Å) |
| `pressure` | float | 压强 (GPa) |

---

## 3. 任务2: 结构优化

### 3.1 概述

优化原子坐标和/或晶胞参数以最小化能量。

### 3.2 Python API

```python
# 仅优化原子坐标
result = calc.optimize(
    atoms,
    fmax=0.05,
    optimizer='LBFGS',
    max_steps=500
)

# 同时优化晶胞
result = calc.optimize(
    atoms,
    fmax=0.01,
    optimize_cell=True,
    optimizer='LBFGS',
    max_steps=500
)

print(f"收敛: {result['converged']}")
print(f"步数: {result['steps']}")
print(f"最终能量: {result['final_energy']:.6f} eV")
```

### 3.3 优化器选项

| 优化器 | 说明 | 推荐用途 |
|--------|------|----------|
| `LBFGS` | 拟牛顿法 | **默认推荐** |
| `BFGS` | BFGS算法 | 小型结构 |
| `FIRE` | 快速惯性松弛 | 复杂势能面 |

---

## 4. 任务3: 分子动力学

### 4.1 概述

运行分子动力学模拟，支持 NVE、NVT、NPT 系综。

### 4.2 Python API

```python
# NVT 模拟 (恒温)
final_atoms = calc.run_md(
    atoms,
    ensemble='nvt',
    temperature=300,        # K
    steps=50000,            # 50 ps @ 1 fs/step
    timestep=1.0,           # fs
    trajectory='nvt_md.traj',
    logfile='nvt_md.log'
)

# NPT 模拟 (恒温恒压)
final_atoms = calc.run_md(
    atoms,
    ensemble='npt',
    temperature=300,
    pressure=0.0,           # GPa
    steps=50000,
    trajectory='npt_md.traj'
)
```

### 4.3 系综说明

| 系综 | 说明 | 热浴 |
|------|------|------|
| `nve` | 微正则 (恒E) | 无 |
| `nvt` | 正则 (恒T) | Langevin |
| `npt` | 等温等压 | Berendsen |

### 4.4 MOFSimBench MD 稳定性

MatterSim 在 MOFSimBench 中展现出 **#1** 的 MD 稳定性（与 eSEN 并列）：
- 20 ps 模拟稳定运行
- 无能量发散
- 无原子飞离

---

## 5. 任务4: 声子计算

### 5.1 概述

计算声子态密度和热力学性质。

### 5.2 Python API

```python
# 声子计算
result = calc.phonon(
    atoms,
    supercell_matrix=[2, 2, 2],
    mesh=[20, 20, 20],
    t_min=0,
    t_max=1000,
    t_step=10
)

# 检查虚频
if result['has_imaginary']:
    print(f"警告: 发现 {result['imaginary_modes']} 个虚频!")

# 热力学性质
thermal = result['thermal']
print(f"300K 热容: {thermal['heat_capacity'][30]:.3f} J/(mol·K)")
```

### 5.3 返回结果

| 键 | 类型 | 说明 |
|----|------|------|
| `frequency_points` | ndarray | 频率点 (THz) |
| `total_dos` | ndarray | 态密度 |
| `has_imaginary` | bool | 是否有虚频 |
| `thermal` | dict | 热力学性质 |

---

## 6. 任务5: 力学性质

### 6.1 概述

计算体模量和状态方程。

### 6.2 Python API

```python
# 体模量计算
result = calc.bulk_modulus(
    atoms,
    strain_range=0.05,      # ±5% 应变
    npoints=11
)

print(f"体模量: {result['bulk_modulus']:.2f} GPa")
print(f"平衡体积: {result['v0']:.2f} Å³")
```

### 6.3 EOS 模型

| 模型 | 说明 |
|------|------|
| `birchmurnaghan` | Birch-Murnaghan (默认) |
| `vinet` | Vinet EOS |
| `murnaghan` | Murnaghan EOS |

---

## 7. 任务6: 吸附能

### 7.1 概述

计算气体分子在 MOF 中的吸附能。这是 MatterSim 的**最强优势**。

### 7.2 MOFSimBench 吸附能排名

| 模型 | CO₂ 吸附 | H₂O 吸附 | 综合排名 |
|------|----------|----------|----------|
| **MatterSim** | **最佳** | **最佳** | **#1** 🥇 |
| eSEN-OAM | 优异 | 优异 | #2 |
| MACE-DAC-1 | 良好 | 良好 | #3 |

### 7.3 Python API

```python
# 吸附能计算
result = calc.adsorption_energy(
    mof_structure=mof,
    gas_molecule="CO2",
    site_position=[10.0, 10.0, 10.0],
    optimize_complex=True,
    fmax=0.05
)

E_ads_eV = result['E_ads']
E_ads_kJ_mol = E_ads_eV * 96.485

print(f"吸附能: {E_ads_eV:.4f} eV ({E_ads_kJ_mol:.2f} kJ/mol)")
```

### 7.4 支持的气体分子

- CO₂ (二氧化碳)
- H₂O (水)
- CH₄ (甲烷)
- N₂ (氮气)
- H₂ (氢气)
- CO (一氧化碳)
- NH₃ (氨)

---

## 8. 任务7: 配位分析

### 8.1 概述

分析金属中心的配位环境。

### 8.2 Python API

```python
# 配位分析
result = calc.coordination(atoms)

for metal_idx, info in result['coordination'].items():
    print(f"金属 {metal_idx}:")
    print(f"  配位数: {info['coordination_number']}")
    print(f"  平均键长: {info['average_distance']:.3f} Å")
```

---

## 9. 任务8: 高通量筛选

### 9.1 概述

批量处理大量 MOF 结构。

### 9.2 Python API

```python
from pathlib import Path
import json

calc = MatterSimInference(model_name="MatterSim-v1-5M", device="cuda")

# 批量处理
structures = Path("mof_database/").glob("*.cif")
results = {}

for cif_file in structures:
    try:
        opt_result = calc.optimize(cif_file, fmax=0.05, optimize_cell=True)
        results[cif_file.name] = {
            'energy_per_atom': opt_result['final_energy'] / len(opt_result['atoms']),
            'converged': opt_result['converged']
        }
    except Exception as e:
        results[cif_file.name] = {'error': str(e)}

# 保存结果
with open('screening_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## 10. 性能基准

### 10.1 MOFSimBench 综合排名

| 排名 | 模型 | 优势领域 |
|------|------|----------|
| #1 | eSEN-OAM | 能量/体模量/优化成功率 |
| #2 | orb-v3-omat | 综合均衡 |
| **#3** | **MatterSim** | **吸附能/MD稳定性** |
| #4 | SevenNet-ompa | 力场精度 |
| #5 | MACE-MPA | 速度快 |

### 10.2 MatterSim 最佳应用场景

| 场景 | 推荐度 | 说明 |
|------|--------|------|
| **吸附能计算** | ⭐⭐⭐⭐⭐ | 排名第一 |
| **长时间 MD** | ⭐⭐⭐⭐⭐ | 稳定性最佳 |
| 结构优化 | ⭐⭐⭐⭐ | 良好 |
| 力学性质 | ⭐⭐⭐ | 中等 |
| 热力学性质 | ⭐⭐⭐ | 中等 |

### 10.3 计算速度

| 结构大小 | CPU 时间/步 | GPU 时间/步 |
|----------|-------------|-------------|
| 100 atoms | ~50 ms | ~5 ms |
| 500 atoms | ~200 ms | ~20 ms |
| 1000 atoms | ~500 ms | ~50 ms |

---

## 参考文献

1. Yang, H. et al. *MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures.* arXiv:2405.04967 (2024)

2. Chen, C. & Ong, S.P. *A Universal Graph Deep Learning Interatomic Potential for the Periodic Table.* Nature Computational Science (2022)

---

## 下一步

- 查看 [MatterSim_inference_API_reference.md](MatterSim_inference_API_reference.md) 获取完整API
- 运行 [examples/](mattersim-inference/examples/) 中的示例
- 参考 [QUICKSTART.md](mattersim-inference/QUICKSTART.md) 快速上手
