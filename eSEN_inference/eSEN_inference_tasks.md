# eSEN Inference - 8 大推理任务详解

> **eSEN 模型**: 平滑且表达性的等变神经网络 (Smooth & Expressive Equivariant Networks)  
> **架构类型**: E(3)-Equivariant GNN  
> **性能排名**: **#1 (MOFSimBench 整体最佳)**  
> **开发团队**: Meta FAIR (Fundamental AI Research) - Fu et al. 2025  
> **论文**: [arXiv:2502.12147](https://arxiv.org/abs/2502.12147)  
> **代码仓库**: [FAIR-Chem/fairchem](https://github.com/FAIR-Chem/fairchem)

---

## 📌 目录

1. [eSEN 模型概述](#1-esen-模型概述)
2. [任务 1: 单点能量计算](#任务-1-单点能量计算)
3. [任务 2: 结构优化](#任务-2-结构优化)
4. [任务 3: 分子动力学模拟](#任务-3-分子动力学模拟)
5. [任务 4: 声子与热力学性质](#任务-4-声子与热力学性质)
6. [任务 5: 力学性质计算](#任务-5-力学性质计算)
7. [任务 6: 吸附能计算](#任务-6-吸附能计算)
8. [任务 7: 配位环境分析](#任务-7-配位环境分析)
9. [任务 8: 高通量筛选](#任务-8-高通量筛选)
10. [性能基准测试](#性能基准测试)
11. [最佳实践与建议](#最佳实践与建议)

---

## 1. eSEN 模型概述

### 1.1 核心特性

**eSEN (Smooth & Expressive Equivariant Networks)** 是 MOFSimBench 基准测试中 **性能最佳** 的通用机器学习力场，具有以下独特优势：

#### 关键创新

1. **平滑势能面 (Smoothness)**
   - 通过严格的架构设计确保势能面的平滑性
   - 避免传统 GNN 中的能量跳变问题
   - 提高 MD 模拟的数值稳定性

2. **高表达性 (Expressiveness)**
   - 平衡平滑性与表达能力
   - 捕捉复杂的原子间相互作用
   - 保持足够的灵活性拟合多样化数据

3. **等变性 (E(3)-Equivariance)**
   - 严格的 E(3) 等变性（旋转、平移、反射）
   - 保守力（通过能量梯度计算）
   - 物理一致性保证

4. **严格架构评估**
   - 系统评估各种架构选择
   - 基于理论和实验的设计决策
   - 优化计算效率与精度平衡

### 1.2 可用模型

| 模型名称 | 训练数据集 | 参数量 | 推荐用途 |
|----------|-----------|--------|----------|
| **eSEN-30M-OAM** | OMat24 + MPtraj + sAlex | 30M | **通用 MOF 建模 (强烈推荐)** |
| **eSEN-30M-MP** | MPtraj only | 30M | Materials Project 数据专用 |

**推荐**: 使用 **eSEN-30M-OAM** 用于 MOF 材料的所有任务（结构优化、MD、吸附等）

### 1.3 MOFSimBench 性能总览

**综合排名**: **#1** 🥇

| 任务类别 | 排名 | MAE | 说明 |
|---------|------|-----|------|
| **整体性能** | **#1** | - | 所有任务中误差分布最窄 |
| **能量预测** | **#1** | 0.041 eV/atom | 最准确的能量预测 |
| **力预测** | **#2** | 0.084 eV/Å | 仅次于 MACE-OMAT |
| **应力预测** | **#3** | 0.31 GPa | Top 3 性能 |
| **结构优化** | **#1** | 89% 成功率 | 与 orb-v3-omat 并列 |
| **体积模量** | **#1** | 2.64 GPa | 最准确的力学性质预测 |
| **热容** | **#3** | 0.024 J/(K·g) | 接近最佳 (orb-v3-omat) |
| **吸附能** | **#2** | - | 仅次于 MatterSim |

**核心优势**:
- ✅ **最窄误差分布**: 所有任务中表现最稳定
- ✅ **优异的力学性质**: 体积模量预测最佳
- ✅ **高成功率**: 结构优化 89% 成功
- ✅ **长时间 MD 稳定**: 与 MatterSim 并列最佳

### 1.4 技术规格

```python
from fairchem.core import OCPCalculator

# 模型规格
模型参数: 30M (Medium size)
输入: 原子类型 + 坐标 + 周期性边界
输出: 能量 + 保守力 + 应力张量
精度: float32 (default) / float64 (可选)
支持元素: 全周期表 118 个元素
计算设备: CPU / CUDA / ROCm
```

---

## 任务 1: 单点能量计算

### 任务描述

计算给定原子结构的总能量、力和应力，无需结构优化。这是最基础的任务，是所有其他任务的基础。

### 代码示例

```python
from esen_inference import ESENInference
from ase.io import read

# 1. 初始化 eSEN 模型 (OAM 版本)
esen = ESENInference(
    model_name='esen-30m-oam',  # 推荐：OAM 版本
    device='cuda',               # GPU 加速
    precision='float32'          # float32 (默认) 或 float64
)

# 2. 加载 MOF 结构
atoms = read('HKUST-1.cif')

# 3. 单点能量计算
result = esen.single_point(atoms)

# 4. 查看结果
print(f"Energy: {result['energy']:.6f} eV")
print(f"Energy per atom: {result['energy']/len(atoms):.6f} eV/atom")
print(f"Forces shape: {result['forces'].shape}")
print(f"Max force: {result['max_force']:.6f} eV/Å")
print(f"RMS force: {result['rms_force']:.6f} eV/Å")
print(f"Stress (Voigt): {result['stress']}")  # (6,) Voigt 记号
print(f"Pressure: {result['pressure']:.4f} GPa")
```

### 输出格式

```python
result = {
    'energy': float,              # 总能量 (eV)
    'forces': np.ndarray,         # 原子力 (N_atoms, 3) eV/Å
    'stress': np.ndarray,         # 应力张量 (6,) Voigt eV/Å³
    'pressure': float,            # 压力 (GPa)
    'max_force': float,           # 最大力 (eV/Å)
    'rms_force': float,           # RMS 力 (eV/Å)
}
```

### 性能基准 (MOFSimBench)

| 指标 | eSEN-OAM | 排名 | 参考 (MACE-OMAT) |
|------|----------|------|------------------|
| **能量 MAE** | **0.041 eV/atom** | **#1** 🥇 | 0.049 eV/atom |
| **力 MAE** | **0.084 eV/Å** | **#2** 🥈 | 0.081 eV/Å |
| **应力 MAE** | **0.31 GPa** | **#3** 🥉 | 0.31 GPa |

**结论**: eSEN-OAM 在能量预测上达到 **最佳精度**，力预测接近最佳。

---

## 任务 2: 结构优化

### 任务描述

通过最小化总能量来优化原子结构的坐标和/或晶胞参数。eSEN 的保守力确保优化过程稳定高效。

### 代码示例

```python
from esen_inference import ESENInference
from ase.io import read, write

# 初始化模型
esen = ESENInference(model_name='esen-30m-oam', device='cuda')

# 加载初始结构
atoms = read('MOF-5_initial.cif')

# 结构优化 (仅优化原子坐标)
result = esen.optimize(
    atoms,
    fmax=0.01,           # 收敛标准: max(|F|) < 0.01 eV/Å
    optimizer='LBFGS',   # LBFGS / BFGS / FIRE
    relax_cell=False,    # 固定晶胞
    max_steps=500,       # 最大步数
    trajectory='opt.traj'  # 保存轨迹
)

# 全优化 (坐标 + 晶胞)
result_full = esen.optimize(
    atoms,
    fmax=0.01,
    optimizer='LBFGS',
    relax_cell=True,     # 优化晶胞参数
    max_steps=500
)

# 查看结果
print(f"Converged: {result_full['converged']}")
print(f"Steps: {result_full['steps']}")
print(f"Initial energy: {result_full['initial_energy']:.6f} eV")
print(f"Final energy: {result_full['final_energy']:.6f} eV")
print(f"Energy降低: {result_full['final_energy'] - result_full['initial_energy']:.6f} eV")
print(f"Final fmax: {result_full['final_fmax']:.6f} eV/Å")

# 保存优化结构
optimized_atoms = result_full['atoms']
write('MOF-5_optimized.cif', optimized_atoms)
```

### 优化器选择

| 优化器 | 适用场景 | 收敛速度 | 内存需求 |
|--------|----------|----------|----------|
| **LBFGS** | 一般优化 (推荐) | 快 | 中等 |
| **BFGS** | 小体系 | 快 | 高 |
| **FIRE** | 难优化体系 | 中等 | 低 |

### 性能基准

| 指标 | eSEN-OAM | 排名 | 参考 (orb-v3-omat) |
|------|----------|------|-------------------|
| **成功率** | **89%** | **#1** 🥇 | 89% (并列) |
| **平均步数** | ~150 | #2 | ~140 |
| **收敛稳定性** | 优异 | #1 | 优异 |

**结论**: eSEN-OAM 在结构优化任务中达到 **最高成功率**（89%），与 orb-v3-omat 并列第一。

---

## 任务 3: 分子动力学模拟

### 任务描述

使用 eSEN 力场进行 NVT (恒温) 或 NPT (恒温恒压) 分子动力学模拟，研究材料的动力学行为。

### NVT 分子动力学

```python
from esen_inference import ESENInference
from ase.io import read

# 初始化
esen = ESENInference(model_name='esen-30m-oam', device='cuda')

# 加载优化后的结构
atoms = read('MOF-5_optimized.cif')

# NVT MD (300 K, 50 ps)
final_atoms = esen.run_md(
    atoms,
    temperature=300.0,    # K
    steps=50000,          # 50,000 steps
    timestep=1.0,         # 1 fs/step → 50 ps total
    ensemble='nvt',
    friction=0.01,        # Langevin 摩擦系数 (ps^-1)
    trajectory='nvt_md.traj',
    logfile='nvt_md.log',
    log_interval=100      # 每 100 步记录一次
)

print(f"Final temperature: {final_atoms.get_temperature():.2f} K")
```

### NPT 分子动力学

```python
# NPT MD (300 K, 1 atm, 100 ps)
final_atoms = esen.run_md(
    atoms,
    temperature=300.0,
    pressure=0.0,         # GPa (0 = 1 atm)
    steps=100000,         # 100 ps
    timestep=1.0,
    ensemble='npt',
    taut=100.0,           # 温度弛豫时间 (fs)
    taup=1000.0,          # 压力弛豫时间 (fs)
    compressibility=4.57e-5,  # GPa^-1 (MOF 典型值)
    trajectory='npt_md.traj',
    logfile='npt_md.log'
)

print(f"Final temperature: {final_atoms.get_temperature():.2f} K")
print(f"Final volume: {final_atoms.get_volume():.2f} Å³")
print(f"Volume change: {(final_atoms.get_volume() - atoms.get_volume())/atoms.get_volume()*100:.2f}%")
```

### 轨迹分析

```python
from esen_inference.tasks.dynamics import analyze_md_trajectory
from ase.io import read

# 读取轨迹
trajectory = read('npt_md.traj', ':')

# 分析
analysis = analyze_md_trajectory(trajectory)

print(f"平均温度: {analysis['avg_temperature']:.2f} ± {analysis['std_temperature']:.2f} K")
print(f"平均体积: {analysis['avg_volume']:.2f} ± {analysis['std_volume']:.2f} Å³")
print(f"平均能量: {analysis['avg_energy']:.4f} eV")
print(f"能量漂移: {analysis['energy_drift']:.6f} eV")
print(f"MSD: {analysis['msd'][-1]:.4f} Å²")
```

### 性能基准

| 指标 | eSEN-OAM | 排名 | 说明 |
|------|----------|------|------|
| **MD 稳定性 (20 ps)** | **优异** | **#1** | 与 MatterSim 并列 |
| **能量守恒** | **极佳** | #1 | 能量漂移最小 |
| **长时间稳定性** | **优异** | #1 | 无结构坍塌 |

**结论**: eSEN-OAM 在长时间 MD 模拟中表现 **最稳定**，适合研究动力学性质。

---

## 任务 4: 声子与热力学性质

### 任务描述

使用 Phonopy 计算声子谱和热力学性质（热容、熵、自由能）。eSEN 的高精度力预测确保声子计算的准确性。

### 代码示例

```python
from esen_inference import ESENInference
from esen_inference.tasks.phonon import plot_phonon_dos, plot_thermal_properties
from ase.io import read

# 初始化
esen = ESENInference(model_name='esen-30m-oam', device='cuda')

# 加载优化后的原胞
primitive_cell = read('MOF-5_primitive.cif')

# 声子计算 (2x2x2 超胞, 20x20x20 k-mesh)
result = esen.phonon(
    primitive_cell,
    supercell_matrix=[2, 2, 2],  # 超胞大小
    mesh=[20, 20, 20],           # k 点网格
    displacement=0.01,           # 位移大小 (Å)
    t_min=0,                     # 最低温度 (K)
    t_max=1000,                  # 最高温度 (K)
    t_step=10                    # 温度步长 (K)
)

# 声子结果
phonon = result['phonon']
freq_points = result['frequency_points']
total_dos = result['total_dos']

# 检查虚频 (负频率)
imaginary_modes = freq_points[freq_points < -0.1]
if len(imaginary_modes) > 0:
    print(f"警告: 检测到 {len(imaginary_modes)} 个虚频模式！")
    print("可能原因: 结构未充分优化或动力学不稳定")
else:
    print("✓ 无虚频模式，结构动力学稳定")

# 绘制声子 DOS
plot_phonon_dos(freq_points, total_dos, output='phonon_dos.png')

# 热力学性质
thermal = result['thermal']
temperatures = thermal['temperatures']
heat_capacity = thermal['heat_capacity']

# 300 K 处的热容
idx_300K = (temperatures >= 300).argmax()
Cv_300K = heat_capacity[idx_300K]
print(f"Heat capacity at 300 K: {Cv_300K:.2f} J/(K·mol)")

# 绘制热容曲线
plot_thermal_properties(
    temperatures,
    heat_capacity,
    output='thermal_properties.png',
    mass_per_formula=1000.0  # MOF 摩尔质量 (g/mol)
)
```

### 性能基准

| 指标 | eSEN-OAM | 排名 | 参考 (orb-v3-omat) |
|------|----------|------|-------------------|
| **热容 MAE** | **0.024 J/(K·g)** | **#3** 🥉 | 0.018 J/(K·g) (#1) |
| **热容 MAPE** | **2.9%** | #3 | 2.3% (#1) |
| **声子准确性** | 优异 | #2 | 最佳 |

**结论**: eSEN-OAM 在热力学性质预测中表现 **优异**，仅次于 orb-v3-omat 和 MACE-MP-MOF0。

---

## 任务 5: 力学性质计算

### 任务描述

计算材料的力学性质，包括体积模量 (Bulk Modulus)、弹性常数等。eSEN 在体积模量预测上达到 **最佳精度**。

### 体积模量 (Bulk Modulus)

```python
from esen_inference import ESENInference
from esen_inference.tasks.mechanics import plot_eos
from ase.io import read

# 初始化
esen = ESENInference(model_name='esen-30m-oam', device='cuda')

# 加载优化结构
atoms = read('MOF-5_optimized.cif')

# 计算体积模量 (EOS 拟合)
result = esen.bulk_modulus(
    atoms,
    strain_range=0.05,    # ±5% 体积应变
    n_points=7,           # 7 个体积点
    eos_type='birchmurnaghan',  # BM / murnaghan / vinet
    optimize_first=True,  # 先优化结构
    fmax=0.01
)

# 结果
B = result['bulk_modulus']       # GPa
V0 = result['equilibrium_volume']  # Å³
E0 = result['equilibrium_energy']  # eV

print(f"Bulk modulus: {B:.2f} GPa")
print(f"Equilibrium volume: {V0:.3f} Å³")
print(f"Equilibrium energy: {E0:.6f} eV")

# 绘制 EOS 曲线
plot_eos(
    result['volumes'],
    result['energies'],
    result['eos'],
    output='eos_curve.png'
)
```

### 性能基准

| 指标 | eSEN-OAM | 排名 | 参考 (MACE-MP-MOF0) |
|------|----------|------|---------------------|
| **体积模量 MAE** | **2.64 GPa** | **#1** 🥇 | 3.14 GPa (#2) |
| **EOS 拟合质量** | 优异 | #1 | 优异 |

**结论**: eSEN-OAM 在体积模量预测中达到 **最佳精度** (MAE 2.64 GPa)，优于所有其他模型。

### 弹性常数 (Elastic Constants)

```python
# 注意: 弹性常数计算需要应变-应力完整映射
# eSEN 支持应力计算，可用于弹性常数

from esen_inference.tasks.mechanics import calculate_elastic_constants

# 计算 6x6 弹性常数张量 (Voigt 记号)
try:
    result = calculate_elastic_constants(
        atoms,
        esen.calculator,
        delta=0.01,      # 应变幅度
        voigt=True       # 使用 Voigt 记号
    )
    
    C = result['elastic_tensor']  # (6, 6) GPa
    B_vrh = result['bulk_modulus_vrh']  # GPa
    G_vrh = result['shear_modulus_vrh']  # GPa
    
    print(f"Bulk modulus (VRH): {B_vrh:.2f} GPa")
    print(f"Shear modulus (VRH): {G_vrh:.2f} GPa")
    
except NotImplementedError:
    print("弹性常数完整计算需要高级实现，建议使用 DFT 验证")
```

---

## 任务 6: 吸附能计算

### 任务描述

计算客体分子（如 CO₂、H₂O、H₂）在 MOF 中的吸附能。eSEN 在主客体相互作用建模中表现优异。

### 代码示例

```python
from esen_inference import ESENInference
from ase.io import read

# 初始化
esen = ESENInference(model_name='esen-30m-oam', device='cuda')

# 加载结构
mof = read('HKUST-1.cif')                # 主体 MOF
co2 = read('CO2.xyz')                     # 客体 CO₂
mof_with_co2 = read('HKUST-1_CO2.cif')   # MOF + CO₂ 复合物

# 计算吸附能
result = esen.adsorption_energy(
    host=mof,
    guest=co2,
    complex_atoms=mof_with_co2,
    optimize_complex=True,  # 优化复合物
    fmax=0.05
)

# 结果
E_ads = result['E_ads']  # eV (负值表示稳定吸附)
E_ads_per_atom = result['E_ads_per_atom']  # eV/atom

print(f"Adsorption energy: {E_ads:.6f} eV")
print(f"E_ads per guest atom: {E_ads_per_atom:.6f} eV/atom")

if E_ads < 0:
    print("→ Stable adsorption (E_ads < 0)")
    # 转换为常用单位
    E_ads_kJ_mol = E_ads * 96.485  # kJ/mol
    print(f"E_ads: {E_ads_kJ_mol:.2f} kJ/mol")
else:
    print("→ Unstable adsorption (E_ads > 0)")

# MOF 吸附能参考范围:
# - CO₂: -10 to -40 kJ/mol (物理吸附)
# - H₂O: -40 to -80 kJ/mol (较强相互作用)
# - H₂: -5 to -15 kJ/mol (弱相互作用)
```

### 性能基准

| 指标 | eSEN-OAM | 排名 | 参考 (MatterSim) |
|------|----------|------|------------------|
| **CO₂ 吸附能** | 优异 | **#2** 🥈 | 最佳 (#1) |
| **H₂O 吸附能** | 优异 | #2 | 最佳 |
| **主客体相互作用** | 准确 | #2 | 最准确 |

**结论**: eSEN-OAM 在吸附能计算中表现 **优异**，仅次于 MatterSim，优于微调的 MACE-DAC-1。

---

## 任务 7: 配位环境分析

### 任务描述

分析 MOF 中金属中心的配位环境，包括配位数、配位原子类型、配位距离等。

### 代码示例

```python
from esen_inference import ESENInference
from ase.io import read

# 初始化
esen = ESENInference(model_name='esen-30m-oam', device='cuda')

# 加载 MOF 结构
atoms = read('HKUST-1.cif')

# 找到 Cu 原子索引
cu_indices = [i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == 'Cu']

# 配位分析
result = esen.coordination(
    atoms,
    center_indices=cu_indices,  # Cu 金属中心
    cutoff_scale=1.3,           # 1.3 × 天然截断半径
    neighbor_indices=None       # 考虑所有原子
)

# 查看结果
cn = result['coordination_numbers']
neighbor_lists = result['neighbor_lists']
distances = result['distances']

for cu_idx in cu_indices[:3]:  # 显示前 3 个 Cu
    print(f"\nCu atom {cu_idx}:")
    print(f"  Coordination number: {cn[cu_idx]}")
    print(f"  Neighbors: {neighbor_lists[cu_idx]}")
    print(f"  Distances (Å): {[f'{d:.3f}' for d in distances[cu_idx]]}")
    
    # 配位原子类型统计
    neighbor_symbols = [atoms[i].symbol for i in neighbor_lists[cu_idx]]
    from collections import Counter
    coord_types = Counter(neighbor_symbols)
    print(f"  Coordination types: {dict(coord_types)}")
```

### 寻找吸附位点

```python
from esen_inference.tasks.adsorption import find_adsorption_sites

# 网格法寻找潜在吸附位点
sites = find_adsorption_sites(
    atoms,
    guest_symbol='C',      # 探针原子 (CO₂ 的 C)
    min_distance=2.5,      # 与框架最小距离 (Å)
    grid_spacing=0.5       # 网格间距 (Å)
)

print(f"Found {len(sites)} potential adsorption sites")

# 可视化前 10 个位点
from ase import Atoms
from ase.io import write

site_atoms = Atoms('He' * len(sites[:10]), positions=sites[:10])
combined = atoms + site_atoms
write('adsorption_sites.cif', combined)
```

---

## 任务 8: 高通量筛选

### 任务描述

批量处理多个 MOF 结构，计算能量、优化、体积模量等性质，用于高通量材料筛选。

### 代码示例

```python
from esen_inference import ESENInference
from ase.io import read, write
from pathlib import Path
import numpy as np

# 初始化 (一次初始化，多次使用)
esen = ESENInference(model_name='esen-30m-oam', device='cuda')

# MOF 数据库
mof_files = Path('mof_database').glob('*.cif')

results = {}

for mof_file in mof_files:
    mof_name = mof_file.stem
    print(f"Processing: {mof_name}")
    
    try:
        # 加载结构
        atoms = read(mof_file)
        
        # 1. 优化
        opt_result = esen.optimize(atoms, fmax=0.05, relax_cell=True, max_steps=500)
        
        if not opt_result['converged']:
            print(f"  Warning: {mof_name} did not converge")
            continue
        
        # 2. 单点能量
        sp_result = esen.single_point(opt_result['atoms'])
        
        # 3. 体积模量 (快速估算: 3 个点)
        bulk_result = esen.bulk_modulus(
            opt_result['atoms'],
            strain_range=0.03,
            n_points=5,
            optimize_first=False
        )
        
        # 存储结果
        results[mof_name] = {
            'energy': sp_result['energy'],
            'energy_per_atom': sp_result['energy'] / len(atoms),
            'volume': opt_result['atoms'].get_volume(),
            'bulk_modulus': bulk_result['bulk_modulus'],
            'max_force': sp_result['max_force'],
            'converged': True
        }
        
        # 保存优化结构
        write(f'optimized/{mof_name}_opt.cif', opt_result['atoms'])
        
        print(f"  ✓ Completed: B = {bulk_result['bulk_modulus']:.2f} GPa")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results[mof_name] = {'converged': False, 'error': str(e)}

# 结果分析
converged_mofs = {k: v for k, v in results.items() if v.get('converged', False)}

if converged_mofs:
    bulk_moduli = [v['bulk_modulus'] for v in converged_mofs.values()]
    
    print(f"\n=== High-Throughput Screening Results ===")
    print(f"Total MOFs: {len(results)}")
    print(f"Converged: {len(converged_mofs)}")
    print(f"Success rate: {len(converged_mofs)/len(results)*100:.1f}%")
    print(f"\nBulk modulus statistics:")
    print(f"  Mean: {np.mean(bulk_moduli):.2f} GPa")
    print(f"  Std: {np.std(bulk_moduli):.2f} GPa")
    print(f"  Min: {np.min(bulk_moduli):.2f} GPa")
    print(f"  Max: {np.max(bulk_moduli):.2f} GPa")
    
    # 找到最硬和最软的 MOF
    hardest = max(converged_mofs.items(), key=lambda x: x[1]['bulk_modulus'])
    softest = min(converged_mofs.items(), key=lambda x: x[1]['bulk_modulus'])
    
    print(f"\nHardest MOF: {hardest[0]} (B = {hardest[1]['bulk_modulus']:.2f} GPa)")
    print(f"Softest MOF: {softest[0]} (B = {softest[1]['bulk_modulus']:.2f} GPa)")
```

### 性能优势

eSEN-30M-OAM 在高通量筛选中的优势:
- ✅ **高成功率**: 89% 优化成功率
- ✅ **稳定预测**: 误差分布最窄，结果可靠
- ✅ **GPU 加速**: 支持批量并行计算
- ✅ **全元素支持**: 118 个元素全覆盖

---

## 性能基准测试

### MOFSimBench 综合排名

| 排名 | 模型 | 整体表现 | 核心优势 |
|------|------|----------|----------|
| **#1** 🥇 | **eSEN-30M-OAM** | 最窄误差分布 | 能量/体积模量/MD 稳定性 |
| #2 | orb-v3-omat | 热容最佳 | 热力学性质 |
| #3 | MACE-OMAT-0 | 力预测最佳 | 精确力场 |
| #4 | MatterSim | 吸附最佳 | 主客体相互作用 |

### 各任务详细性能

#### 1. 能量 & 力 & 应力

| 指标 | eSEN-OAM | 排名 | 最佳模型 |
|------|----------|------|----------|
| 能量 MAE | 0.041 eV/atom | **#1** 🥇 | eSEN-OAM |
| 力 MAE | 0.084 eV/Å | #2 | MACE-OMAT-0 (0.081) |
| 应力 MAE | 0.31 GPa | #3 | SevenNet-ompa (0.28) |

#### 2. 结构优化

| 指标 | eSEN-OAM | 排名 | 最佳模型 |
|------|----------|------|----------|
| 成功率 | 89% | **#1** 🥇 | eSEN-OAM / orb-v3-omat |
| 平均步数 | ~150 | #2 | orb-v3-omat (~140) |

#### 3. 力学性质

| 指标 | eSEN-OAM | 排名 | 最佳模型 |
|------|----------|------|----------|
| 体积模量 MAE | 2.64 GPa | **#1** 🥇 | eSEN-OAM |
| EOS 拟合 R² | 0.98+ | #1 | eSEN-OAM |

#### 4. 热力学性质

| 指标 | eSEN-OAM | 排名 | 最佳模型 |
|------|----------|------|----------|
| 热容 MAE | 0.024 J/(K·g) | #3 | orb-v3-omat (0.018) |
| 热容 MAPE | 2.9% | #3 | orb-v3-omat (2.3%) |

#### 5. 吸附能

| 指标 | eSEN-OAM | 排名 | 最佳模型 |
|------|----------|------|----------|
| CO₂ 吸附 | 优异 | #2 | MatterSim |
| H₂O 吸附 | 优异 | #2 | MatterSim |

#### 6. MD 稳定性

| 指标 | eSEN-OAM | 排名 | 最佳模型 |
|------|----------|------|----------|
| 20 ps 稳定性 | 优异 | **#1** 🥇 | eSEN-OAM / MatterSim |
| 能量守恒 | 极佳 | #1 | eSEN-OAM |

---

## 最佳实践与建议

### 1. 模型选择

```python
# 通用 MOF 建模 (强烈推荐)
esen = ESENInference(model_name='esen-30m-oam', device='cuda')

# Materials Project 数据专用
esen = ESENInference(model_name='esen-30m-mp', device='cuda')
```

**推荐**: 所有 MOF 任务都使用 **eSEN-30M-OAM**

### 2. 精度 vs 速度

```python
# 生产环境 (平衡精度与速度)
esen = ESENInference(
    model_name='esen-30m-oam',
    device='cuda',
    precision='float32'  # 默认，推荐
)

# 高精度基准 (牺牲速度)
esen = ESENInference(
    model_name='esen-30m-oam',
    device='cuda',
    precision='float64'  # 双精度
)
```

### 3. 结构优化建议

```python
# 两阶段优化策略
# 第 1 阶段: 粗优化 (快速)
result1 = esen.optimize(atoms, fmax=0.05, relax_cell=True, max_steps=500)

# 第 2 阶段: 精优化 (高精度)
result2 = esen.optimize(result1['atoms'], fmax=0.01, relax_cell=True, max_steps=300)
```

### 4. MD 模拟建议

```python
# 1. 先优化结构
opt_result = esen.optimize(atoms, fmax=0.01, relax_cell=True)

# 2. 预平衡 (NVT, 10 ps)
pre_equilibrate = esen.run_md(
    opt_result['atoms'],
    temperature=300,
    steps=10000,
    ensemble='nvt'
)

# 3. 生产 MD (NPT, 100 ps)
production = esen.run_md(
    pre_equilibrate,
    temperature=300,
    pressure=0.0,
    steps=100000,
    ensemble='npt',
    trajectory='production.traj'
)
```

### 5. 声子计算建议

```python
# 1. 使用充分优化的原胞
opt_primitive = esen.optimize(primitive_cell, fmax=0.001, relax_cell=True)

# 2. 选择合适的超胞大小 (至少 10 Å 每个方向)
# 对于小原胞: [3, 3, 3]
# 对于大原胞: [2, 2, 2]
result = esen.phonon(
    opt_primitive['atoms'],
    supercell_matrix=[2, 2, 2],
    mesh=[20, 20, 20],
    displacement=0.01  # 小位移，避免非谐效应
)
```

### 6. GPU 内存优化

```python
# 大体系 (> 500 atoms) 内存优化
import torch

# 清理 GPU 缓存
torch.cuda.empty_cache()

# 使用 float32 精度
esen = ESENInference(model_name='esen-30m-oam', device='cuda', precision='float32')

# 或者使用 CPU (内存更大)
esen_cpu = ESENInference(model_name='esen-30m-oam', device='cpu')
```

### 7. 批量计算优化

```python
# 复用模型实例，避免重复加载
esen = ESENInference(model_name='esen-30m-oam', device='cuda')

for mof_file in mof_files:
    atoms = read(mof_file)
    result = esen.single_point(atoms)  # 复用同一个 esen 实例
    # ...
```

---

## 总结

**eSEN-30M-OAM** 是 MOFSimBench 基准测试中 **性能最佳** 的通用机器学习力场:

### 核心优势

1. ✅ **整体最佳**: 所有任务中误差分布最窄，性能最稳定
2. ✅ **能量预测第一**: MAE 0.041 eV/atom，精度最高
3. ✅ **体积模量第一**: MAE 2.64 GPa，力学性质最准
4. ✅ **MD 稳定性第一**: 长时间模拟无坍塌，能量守恒极佳
5. ✅ **优化成功率第一**: 89% 成功率，与 orb-v3-omat 并列
6. ✅ **吸附能第二**: 仅次于 MatterSim，优于所有微调模型

### 推荐应用场景

- **通用 MOF 建模**: 结构优化、能量计算、性质预测
- **力学性质研究**: 体积模量、弹性常数（最佳精度）
- **长时间 MD**: 稳定性最佳，适合动力学研究
- **高通量筛选**: 高成功率 + 稳定预测
- **吸附研究**: 主客体相互作用准确

### 与其他模型对比

| 场景 | 推荐模型 | 原因 |
|------|----------|------|
| 通用 MOF 建模 | **eSEN-OAM** | 整体最佳 |
| 热容预测 | orb-v3-omat | 热容第一 (0.018 vs 0.024) |
| 精确力场 | MACE-OMAT-0 | 力预测第一 |
| 吸附研究 | MatterSim | 吸附能第一 |

**结论**: **eSEN-30M-OAM** 是 MOF 材料计算的 **首选模型**，适用于绝大多数场景。

---

**文档版本**: v1.0  
**更新日期**: 2026-01-07  
**模型版本**: eSEN-30M-OAM (30M parameters)  
**性能排名**: #1 (MOFSimBench)
