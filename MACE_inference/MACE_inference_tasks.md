# MACE 系列模型推理任务文档

> **模型类别**: 等变图神经网络 (Equivariant GNN)
> 
> **文档版本**: v1.1
> 
> **最后更新**: 2026年1月7日

**相关文档**：
- [MACE 环境安装指南](INSTALL.md)：CPU/GPU 环境配置说明
- [MACE API 接口参考](MACE_inference_API_reference.md)：详细的输入输出规范和接口验证

---

## 目录

1. [模型概述](#1-模型概述)
2. [推理任务详解](#2-推理任务详解)
   - [2.1 静态建模与结构优化](#21-静态建模与结构优化)
   - [2.2 动力学建模](#22-动力学建模)
   - [2.3 体相性质预测](#23-体相性质预测)
   - [2.4 主客体相互作用](#24-主客体相互作用)
3. [基于ASE/Phonopy的可扩展推理任务](#3-基于asephonopy的可扩展推理任务)
4. [任务可行性总结](#4-任务可行性总结)

---

## 1. 模型概述

### 1.1 MACE 系列模型列表

| 模型名称 | 训练数据集 | 特点 | GitHub |
|----------|------------|------|--------|
| **MACE-MP-0a** | MPtraj | 第一个通用MACE模型 | [ACEsuit/mace](https://github.com/ACEsuit/mace) |
| **MACE-MP-0b3 (medium)** | MPtraj | 改进的对排斥、正确的孤立原子、更好的高压稳定性 | [ACEsuit/mace](https://github.com/ACEsuit/mace) |
| **MACE-MPA-0** | MPtraj + sAlex | 扩展训练数据，更好的化学多样性 | [ACEsuit/mace](https://github.com/ACEsuit/mace) |
| **MACE-OMAT-0** | OMat24 | 使用OMat24数据集训练，包含非平衡构型 | [ACEsuit/mace](https://github.com/ACEsuit/mace) |
| **MACE-MATPES-r2SCAN-0** | MATPES-r2SCAN | 使用r2SCAN泛函数据训练，更高精度 | [ACEsuit/mace](https://github.com/ACEsuit/mace) |

### 1.2 MACE 架构特点

- **高阶等变消息传递神经网络**：基于E(3)等变性设计
- **原子集群展开（ACE）理论**：系统地捕获多体相互作用
- **保守力计算**：通过能量对原子位置的梯度计算力，保证物理一致性
- **全元素覆盖**：支持元素周期表大部分元素
- **预训练权重可用**：可直接加载使用，无需从头训练

### 1.3 安装与基本使用

详细安装说明请参考 [INSTALL.md](INSTALL.md)。

```bash
# 安装MACE
pip install mace-torch

# 或从源码安装
git clone https://github.com/ACEsuit/mace.git
cd mace
pip install -e .
```

#### 设备选择 (CPU/GPU)

```python
import torch
from mace.calculators import mace_mp

# 自动检测设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# 加载预训练模型
calc = mace_mp(model="medium", device=DEVICE)  # MACE-MP-0b3

# 设置到ASE Atoms对象
atoms.calc = calc
```

| 设备 | 参数 | 适用场景 | 依赖文件 |
|------|------|----------|----------|
| GPU | `device="cuda"` | 大型结构、长时间MD、批量计算 | `requirements-gpu.txt` |
| CPU | `device="cpu"` | 小型测试、无GPU环境 | `requirements-cpu.txt` |

> ⚠️ **注意**：本文档后续代码示例默认使用 `device="cuda"`。如使用CPU版本，请将所有 `device="cuda"` 替换为 `device="cpu"`。CPU版本计算速度较慢（约10-100倍），不建议用于长时间MD或大型声子计算。

### 1.4 D3色散校正

#### 为什么需要D3校正？

MACE模型的训练数据主要基于以下泛函：

| MACE模型 | 训练泛函 | 色散校正情况 |
|----------|----------|--------------|
| MACE-MP-0a/0b3 | PBE | ❌ PBE本身不含色散 |
| MACE-MPA-0 | PBE (混合数据) | ⚠️ 部分 |
| MACE-OMAT-0 | PBE | ❌ 不含色散 |
| MACE-MATPES-r2SCAN-0 | r2SCAN | ⚠️ r2SCAN有一定色散能力但不完整 |

**DFT-D3**（Grimme色散校正）对MOF模拟非常重要：

1. **主客体相互作用**：气体分子吸附主要依赖范德华力/色散力
2. **孔道内分子行为**：气体在孔道中的扩散和吸附位点识别
3. **层间/链间相互作用**：某些MOF拓扑需要准确的弱相互作用描述
4. **吸附热预测**：色散力对吸附热贡献显著

#### D3校正的使用方法

MACE本身不包含D3校正，但可以通过ASE的`SumCalculator`将MACE与D3校正组合：

```bash
# 安装torch-dftd（PyTorch实现的DFT-D3）
pip install torch-dftd
```

```python
import torch
from ase.io import read
from ase.calculators.mixing import SumCalculator
from mace.calculators import mace_mp
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

# 设备选择
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 加载结构
atoms = read("structure.cif")

# 创建MACE计算器
mace_calc = mace_mp(model="medium", device=DEVICE)

# 创建D3校正计算器
# damping='bj' 使用Becke-Johnson阻尼函数（推荐）
# xc='pbe' 指定与训练数据一致的泛函
d3_calc = TorchDFTD3Calculator(
    device=DEVICE,
    damping="bj",
    xc="pbe"
)

# 组合计算器：E_total = E_MACE + E_D3
combined_calc = SumCalculator([mace_calc, d3_calc])

# 设置组合计算器
atoms.calc = combined_calc

# 现在可以正常使用
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

#### 何时使用D3校正？

| 任务 | 是否建议使用D3 | 原因 |
|------|----------------|------|
| 结构优化 | ⚠️ 可选 | 对平衡体积有轻微影响 |
| 单点能量 | ⚠️ 可选 | 取决于比较对象 |
| MD稳定性 | ⚠️ 可选 | 对框架影响较小 |
| **主客体相互作用** | ✅ **强烈建议** | 色散力是主要贡献 |
| **吸附力预测** | ✅ **强烈建议** | 色散力是主要贡献 |
| 体积模量 | ❌ 影响较小 | 框架弹性主要由共价键决定 |
| 热容 | ❌ 影响较小 | 声子主要由强键决定 |

#### 注意事项

1. **计算速度**：添加D3校正会略微增加计算时间（约5-10%）
2. **一致性**：如果训练数据已包含D3校正，不要重复添加
3. **泛函匹配**：D3参数应与MACE训练数据的泛函匹配（通常为PBE）
4. **r2SCAN模型**：MACE-MATPES-r2SCAN-0可能不需要额外D3，因为r2SCAN本身有一定色散能力

---

## 2. 推理任务详解

### 2.1 静态建模与结构优化

#### 2.1.1 结构优化 (Structural Optimization)

| 项目 | 详细信息 |
|------|----------|
| **任务描述** | 优化原子位置和晶胞参数以找到最小能量构型 |
| **MACE支持** | ✅ 完全支持 |
| **使用的库** | ASE |
| **推荐方法** | FrechetCellFilter + LBFGS/FIRE优化器 |
| **收敛标准** | fmax < 0.01 eV/Å（力收敛），stress收敛可选 |

**ASE实现代码**：

```python
from ase.io import read
from ase.optimize import LBFGS, FIRE
from ase.filters import FrechetCellFilter
from mace.calculators import mace_mp

# 加载结构
atoms = read("structure.cif")

# 设置MACE计算器
calc = mace_mp(model="medium", device="cuda")
atoms.calc = calc

# 使用FrechetCellFilter同时优化原子位置和晶胞
filtered_atoms = FrechetCellFilter(atoms)

# 结构优化
optimizer = LBFGS(filtered_atoms, logfile="opt.log", trajectory="opt.traj")
optimizer.run(fmax=0.01)  # 收敛标准: 最大力 < 0.01 eV/Å

# 保存优化后的结构
atoms.write("optimized.cif")
```

---

#### 2.1.2 单点能量计算 (Single-Point Energy Calculation)

| 项目 | 详细信息 |
|------|----------|
| **任务描述** | 计算给定构型的势能、力和应力 |
| **MACE支持** | ✅ 完全支持 |
| **使用的库** | ASE |

**计算意义**：

单点能量计算是最基础的推理任务，具有以下重要意义：

1. **能量基准比较**：将MLIP计算的能量与DFT参考值对比，评估模型精度
2. **构型筛选**：快速评估大量候选结构的相对稳定性，无需完整优化
3. **力场验证**：检验力和应力的预测是否合理，验证模型的可靠性
4. **势能面采样**：用于NEB、过渡态搜索等需要势能面信息的计算
5. **热力学积分**：自由能计算中需要大量单点能量评估
6. **训练数据生成**：为进一步微调模型生成能量标签

**ASE实现代码**：

```python
from ase.io import read
from mace.calculators import mace_mp

# 加载结构
atoms = read("structure.cif")

# 设置MACE计算器
calc = mace_mp(model="medium", device="cuda")
atoms.calc = calc

# 单点计算
energy = atoms.get_potential_energy()  # 总能量 (eV)
forces = atoms.get_forces()            # 力 (eV/Å)
stress = atoms.get_stress()            # 应力 (eV/Å³)

print(f"Total Energy: {energy:.4f} eV")
print(f"Energy per atom: {energy/len(atoms):.4f} eV/atom")
print(f"Max force: {abs(forces).max():.4f} eV/Å")
```

---

### 2.2 动力学建模

#### 2.2.1 NPT分子动力学稳定性

| 项目 | 详细信息 |
|------|----------|
| **任务描述** | 在NpT系综下进行分子动力学模拟，评估结构稳定性 |
| **MACE支持** | ✅ 完全支持 |
| **使用的库** | ASE |
| **模拟条件** | 300K, 1 bar, 时间步长1-2 fs |
| **评估指标** | 体积漂移（阈值±10%）、能量守恒 |

> ⚠️ **CPU版本警告**：50ps MD模拟在CPU上可能需要数小时，建议先用短时间测试（如1ps）验证代码正确性。

**ASE实现代码**：

```python
from ase.io import read, Trajectory
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from mace.calculators import mace_mp

# 加载结构（建议使用supercell）
atoms = read("structure.cif") * (2, 2, 2)

# 设置MACE计算器
calc = mace_mp(model="medium", device="cuda")
atoms.calc = calc

# 初始化速度
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# NPT系综设置
temperature = 300  # K
pressure = 1.01325 * units.bar  # 1 atm
timestep = 1 * units.fs

dyn = NPT(
    atoms,
    timestep=timestep,
    temperature_K=temperature,
    externalstress=pressure,
    ttime=25 * units.fs,      # 温度耦合时间
    pfactor=75 * units.fs**2  # 压力耦合时间
)

# 记录轨迹
traj = Trajectory("npt_md.traj", "w", atoms)
dyn.attach(traj.write, interval=100)

# 记录体积变化
initial_volume = atoms.get_volume()
def print_status():
    current_volume = atoms.get_volume()
    volume_change = (current_volume - initial_volume) / initial_volume * 100
    print(f"Step: {dyn.nsteps}, Volume change: {volume_change:.2f}%")
dyn.attach(print_status, interval=1000)

# 运行50ps
dyn.run(50000)  # 50000 steps * 1fs = 50ps
```

---

#### 2.2.2 配位环境稳定性（多种金属）

| 项目 | 详细信息 |
|------|----------|
| **任务描述** | 评估不同金属的配位数在MD过程中的保持情况 |
| **MACE支持** | ✅ 完全支持 |
| **使用的库** | ASE |
| **测试金属** | Cu, Zn, Fe, Co, Ni, Cr, Mn, Mg, Ca, Al, Ti, V, Zr 等 |
| **模拟协议** | 10ps@300K → 10ps@400K → 10ps@300K |

**金属配位环境说明**：

| 金属类型 | 常见配位数 | 代表性MOF |
|----------|------------|-----------|
| Cu | 4 (正方形), 5 (三角双锥), 6 (八面体) | HKUST-1, MOF-2 |
| Zn | 4 (四面体), 6 (八面体) | ZIF-8, MOF-5 |
| Fe | 4, 5, 6 | MIL-53(Fe), MIL-100(Fe) |
| Co | 4, 6 | ZIF-67, MOF-74(Co) |
| Ni | 4, 6 | MOF-74(Ni), Ni-MOF |
| Cr | 6 | MIL-101(Cr) |
| Zr | 8 | UiO-66, UiO-67 |
| Al | 6 | MIL-53(Al), CAU-10 |
| Mg | 6 | MOF-74(Mg) |

**ASE实现代码**：

```python
from ase.io import read, Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import NeighborList
from ase import units
from mace.calculators import mace_mp
import numpy as np

def get_coordination_number(atoms, metal_symbol, cutoff=2.5):
    """计算金属原子的配位数"""
    metal_indices = [i for i, s in enumerate(atoms.symbols) if s == metal_symbol]
    
    # 创建邻居列表
    nl = NeighborList([cutoff/2] * len(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)
    
    coord_numbers = []
    for idx in metal_indices:
        neighbors, _ = nl.get_neighbors(idx)
        # 排除金属-金属配位，只计算金属-配体
        non_metal_neighbors = [n for n in neighbors if atoms[n].symbol not in ['Cu', 'Zn', 'Fe', 'Co', 'Ni', 'Cr', 'Mn', 'Mg', 'Al', 'Ti', 'Zr']]
        coord_numbers.append(len(non_metal_neighbors))
    
    return coord_numbers

# 加载结构
atoms = read("metal_mof.cif") * (2, 2, 2)
metal_symbol = "Cu"  # 可更换为其他金属: Zn, Fe, Co, Ni 等

# 设置计算器
calc = mace_mp(model="medium", device="cuda")
atoms.calc = calc

# 记录初始配位数
initial_cn = get_coordination_number(atoms, metal_symbol)
print(f"Initial coordination numbers for {metal_symbol}: {initial_cn}")

MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# 阶段1: 300K, 10ps
dyn = Langevin(atoms, 1*units.fs, temperature_K=300, friction=0.02)
dyn.run(10000)

# 阶段2: 升温到400K, 10ps
dyn.set_temperature(temperature_K=400)
dyn.run(10000)

# 阶段3: 降温回300K, 10ps
dyn.set_temperature(temperature_K=300)
dyn.run(10000)

# 记录最终配位数
final_cn = get_coordination_number(atoms, metal_symbol)
print(f"Final coordination numbers for {metal_symbol}: {final_cn}")

# 评估配位稳定性
cn_change = np.array(final_cn) - np.array(initial_cn)
print(f"Coordination number changes: {cn_change}")
print(f"Coordination stable: {np.allclose(cn_change, 0)}")
```

---

#### 2.2.3 高温稳定性测试

| 项目 | 详细信息 |
|------|----------|
| **任务描述** | 渐进升温模拟评估热稳定性极限 |
| **MACE支持** | ✅ 完全支持 |
| **使用的库** | ASE |
| **模拟条件** | 从300K到1000K，100K步进，每步20ps |

**ASE实现代码**：

```python
from ase.io import read, Trajectory
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from mace.calculators import mace_mp
import numpy as np

# 加载结构
atoms = read("structure.cif") * (2, 2, 2)
calc = mace_mp(model="medium", device="cuda")
atoms.calc = calc

MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# 温度序列
temperatures = [300, 400, 500, 600, 700, 800, 900, 1000]

results = []
for temp in temperatures:
    print(f"\n=== Temperature: {temp} K ===")
    
    dyn = NPT(
        atoms,
        timestep=1 * units.fs,
        temperature_K=temp,
        externalstress=1.01325 * units.bar,
        ttime=25 * units.fs,
        pfactor=75 * units.fs**2
    )
    
    # 记录初始体积
    initial_volume = atoms.get_volume()
    
    # 运行20ps
    dyn.run(20000)
    
    # 记录最终体积和能量
    final_volume = atoms.get_volume()
    final_energy = atoms.get_potential_energy()
    volume_change = (final_volume - initial_volume) / initial_volume * 100
    
    results.append({
        'temperature': temp,
        'volume_change': volume_change,
        'final_energy': final_energy,
        'stable': abs(volume_change) < 20  # 稳定性阈值
    })
    
    print(f"Volume change: {volume_change:.2f}%")
    print(f"Stable: {results[-1]['stable']}")
    
    # 如果结构严重变形，停止测试
    if abs(volume_change) > 50:
        print("Structure collapsed! Stopping test.")
        break

# 确定热稳定性极限
stable_temps = [r['temperature'] for r in results if r['stable']]
print(f"\nThermal stability limit: {max(stable_temps) if stable_temps else 'N/A'} K")
```

---

### 2.3 体相性质预测

#### 2.3.1 体积模量 (Bulk Modulus)

| 项目 | 详细信息 |
|------|----------|
| **任务描述** | 通过Birch-Murnaghan状态方程计算体积模量 |
| **MACE支持** | ✅ 完全支持 |
| **使用的库** | ASE |
| **计算方法** | 体积应变±4%（11步），拟合EOS |

**计算意义**：

体积模量（Bulk Modulus, K）是材料抵抗均匀压缩的能力度量，具有以下重要意义：

1. **机械稳定性评估**：体积模量反映MOF框架的刚性，低体积模量表示结构容易变形
2. **压力响应预测**：预测材料在高压条件下（如工业吸附过程）的行为
3. **柔性MOF识别**：异常低的体积模量可能指示呼吸效应（breathing effect）
4. **结构完整性**：筛选能够在实际应用压力条件下保持结构的候选材料
5. **比较不同拓扑结构**：评估不同MOF拓扑对机械性能的影响
6. **与DFT基准对比**：验证MLIP对势能面曲率的预测准确性

**ASE实现代码**：

```python
from ase.io import read
from ase.eos import EquationOfState
from mace.calculators import mace_mp
import numpy as np

# 加载优化后的结构
atoms = read("optimized.cif")
calc = mace_mp(model="medium", device="cuda")
atoms.calc = calc

# 获取初始体积和晶胞
initial_volume = atoms.get_volume()
initial_cell = atoms.get_cell()

# 体积应变范围: ±4%, 11步
strains = np.linspace(-0.04, 0.04, 11)
volumes = []
energies = []

for strain in strains:
    # 均匀缩放晶胞
    strained_atoms = atoms.copy()
    scale_factor = (1 + strain) ** (1/3)
    strained_atoms.set_cell(initial_cell * scale_factor, scale_atoms=True)
    strained_atoms.calc = calc
    
    # 计算能量
    energy = strained_atoms.get_potential_energy()
    volume = strained_atoms.get_volume()
    
    volumes.append(volume)
    energies.append(energy)
    print(f"Strain: {strain:+.3f}, Volume: {volume:.2f} Å³, Energy: {energy:.4f} eV")

# 拟合Birch-Murnaghan状态方程
eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
v0, e0, B = eos.fit()

# 体积模量转换: eV/Å³ → GPa
B_GPa = B * 160.21766208  # 1 eV/Å³ = 160.21766208 GPa

print(f"\n=== Results ===")
print(f"Equilibrium volume: {v0:.2f} Å³")
print(f"Minimum energy: {e0:.4f} eV")
print(f"Bulk modulus: {B_GPa:.2f} GPa")

# 绘图（可选）
eos.plot("bulk_modulus.png")
```

---

#### 2.3.2 热容 (Heat Capacity)

| 项目 | 详细信息 |
|------|----------|
| **任务描述** | 通过声子计算获得定容热容 |
| **MACE支持** | ✅ 完全支持 |
| **使用的库** | ASE + Phonopy |
| **计算方法** | 有限差分法计算力常数，声子态密度积分 |

> ⚠️ **CPU版本警告**：声子计算需要评估大量位移构型的力，在CPU上可能非常耗时（数小时至数天）。强烈建议使用GPU版本进行声子计算。

**ASE + Phonopy实现代码**：

```python
from ase.io import read, write
from mace.calculators import mace_mp
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import numpy as np

def ase_to_phonopy(atoms):
    """ASE Atoms转换为PhonopyAtoms"""
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.get_cell()
    )

# 加载优化后的结构
atoms = read("optimized.cif")
calc = mace_mp(model="medium", device="cuda")

# 创建Phonopy对象
phonopy_atoms = ase_to_phonopy(atoms)
phonopy = Phonopy(
    phonopy_atoms,
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    primitive_matrix='auto'
)

# 生成位移
phonopy.generate_displacements(distance=0.01)  # 0.01 Å位移

# 计算力
supercells = phonopy.supercells_with_displacements
force_sets = []

for i, sc in enumerate(supercells):
    print(f"Calculating forces for displacement {i+1}/{len(supercells)}")
    
    # 转换回ASE
    from ase import Atoms
    ase_sc = Atoms(
        symbols=sc.symbols,
        scaled_positions=sc.scaled_positions,
        cell=sc.cell,
        pbc=True
    )
    ase_sc.calc = calc
    
    forces = ase_sc.get_forces()
    force_sets.append(forces)

# 设置力常数
phonopy.forces = force_sets
phonopy.produce_force_constants()

# 计算热力学性质
phonopy.run_mesh([20, 20, 20])
phonopy.run_thermal_properties(t_min=0, t_max=1000, t_step=10)

# 获取300K的热容
thermal_props = phonopy.get_thermal_properties_dict()
temperatures = thermal_props['temperatures']
heat_capacities = thermal_props['heat_capacity']  # J/K/mol

# 找到300K对应的热容
idx_300K = np.argmin(np.abs(temperatures - 300))
Cv_300K = heat_capacities[idx_300K]

# 转换为J/K/g
molar_mass = sum(atoms.get_masses())  # g/mol
Cv_per_gram = Cv_300K / molar_mass

print(f"\n=== Heat Capacity at 300K ===")
print(f"Cv: {Cv_300K:.2f} J/K/mol")
print(f"Cv: {Cv_per_gram:.4f} J/K/g")
```

---

### 2.4 主客体相互作用

#### 2.4.1 主客体相互作用能预测（多种气体分子）

| 项目 | 详细信息 |
|------|----------|
| **任务描述** | 预测不同气体分子与MOF框架的相互作用能 |
| **MACE支持** | ✅ 完全支持 |
| **使用的库** | ASE |
| **测试气体** | CO₂, H₂O, CH₄, N₂, H₂, O₂, NH₃, H₂S, SO₂, NO₂ 等 |

**气体分子参数**：

| 气体分子 | 动力学直径(Å) | 主要相互作用类型 | 应用场景 |
|----------|---------------|------------------|----------|
| CO₂ | 3.30 | 四极矩-电场梯度 | 碳捕获 |
| H₂O | 2.65 | 氢键、偶极 | 除湿、水稳定性 |
| CH₄ | 3.80 | 弱范德华力 | 天然气存储 |
| N₂ | 3.64 | 四极矩 | 空气分离 |
| H₂ | 2.89 | 弱范德华力 | 氢气存储 |
| O₂ | 3.46 | 顺磁性 | 氧气富集 |
| NH₃ | 2.60 | 氢键、配位 | 氨捕获 |
| H₂S | 3.60 | 偶极、氢键 | 气体脱硫 |
| SO₂ | 4.11 | 偶极 | 烟气脱硫 |

> ⚠️ **重要提示**：对于主客体相互作用计算，**强烈建议使用D3色散校正**，因为色散力是气体分子与MOF框架相互作用的主要贡献。详见[1.4节 D3色散校正](#14-d3色散校正)。

**ASE实现代码（含D3校正）**：

```python
from ase.io import read
from ase.build import molecule
from ase import Atoms
from ase.calculators.mixing import SumCalculator
from mace.calculators import mace_mp
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
import numpy as np

def create_mace_d3_calculator(device="cuda"):
    """创建MACE+D3组合计算器"""
    mace_calc = mace_mp(model="medium", device=device)
    d3_calc = TorchDFTD3Calculator(device=device, damping="bj", xc="pbe")
    return SumCalculator([mace_calc, d3_calc])

def calculate_interaction_energy(mof_atoms, guest_molecule, calc, insertion_point):
    """
    计算主客体相互作用能
    E_interaction = E_MOF+guest - E_MOF - E_guest
    """
    # 1. 计算MOF能量
    mof_atoms.calc = calc
    E_mof = mof_atoms.get_potential_energy()
    
    # 2. 计算孤立客体分子能量
    guest = molecule(guest_molecule)
    guest.calc = calc
    guest.center(vacuum=10)  # 加真空层
    E_guest = guest.get_potential_energy()
    
    # 3. 将客体分子放入MOF
    combined = mof_atoms.copy()
    guest_in_mof = molecule(guest_molecule)
    guest_in_mof.translate(insertion_point)
    combined += guest_in_mof
    combined.calc = calc
    E_combined = combined.get_potential_energy()
    
    # 4. 计算相互作用能
    E_interaction = E_combined - E_mof - E_guest
    
    return E_interaction * 1000  # 转换为meV

# 加载MOF结构
mof = read("mof_structure.cif")

# 使用MACE+D3组合计算器（推荐用于主客体相互作用）
calc = create_mace_d3_calculator(device="cuda")

# 测试不同气体分子
gas_molecules = ['CO2', 'H2O', 'CH4', 'N2', 'H2']

# 在孔道中心插入
pore_center = mof.get_cell().sum(axis=0) / 2  # 简化：晶胞中心

results = []
for gas in gas_molecules:
    try:
        E_int = calculate_interaction_energy(mof, gas, calc, pore_center)
        results.append({'gas': gas, 'E_interaction': E_int})
        print(f"{gas}: Interaction energy = {E_int:.2f} meV")
    except Exception as e:
        print(f"{gas}: Failed - {e}")

# 按相互作用能排序
results.sort(key=lambda x: x['E_interaction'])
print("\n=== Adsorption Affinity Ranking ===")
for r in results:
    print(f"{r['gas']}: {r['E_interaction']:.2f} meV")
```

**不使用D3校正的版本**（如需对比）：

```python
# 仅使用MACE（不含D3校正）
calc = mace_mp(model="medium", device="cuda")
# 其余代码相同
```

---

#### 2.4.2 吸附力预测（多种分子）

| 项目 | 详细信息 |
|------|----------|
| **任务描述** | 预测不同分子在MOF孔道中的受力情况 |
| **MACE支持** | ✅ 完全支持 |
| **使用的库** | ASE |
| **评估指标** | 客体分子上的力 (meV/Å) |

> ⚠️ **重要提示**：与相互作用能计算一样，吸附力预测也**强烈建议使用D3色散校正**。

**ASE实现代码（含D3校正）**：

```python
from ase.io import read
from ase.build import molecule
from ase.calculators.mixing import SumCalculator
from mace.calculators import mace_mp
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
import numpy as np

def create_mace_d3_calculator(device="cuda"):
    """创建MACE+D3组合计算器"""
    mace_calc = mace_mp(model="medium", device=device)
    d3_calc = TorchDFTD3Calculator(device=device, damping="bj", xc="pbe")
    return SumCalculator([mace_calc, d3_calc])

def calculate_adsorption_forces(mof_atoms, guest_molecule, calc, positions_grid):
    """
    计算客体分子在不同位置的受力
    """
    results = []
    
    for pos in positions_grid:
        # 创建组合结构
        combined = mof_atoms.copy()
        guest = molecule(guest_molecule)
        guest.translate(pos)
        
        guest_start_idx = len(combined)
        combined += guest
        combined.calc = calc
        
        # 计算力
        forces = combined.get_forces()
        
        # 提取客体分子上的力
        guest_forces = forces[guest_start_idx:]
        
        # 计算客体分子的合力
        total_force = np.sum(guest_forces, axis=0)
        force_magnitude = np.linalg.norm(total_force)
        
        results.append({
            'position': pos,
            'force_vector': total_force,
            'force_magnitude': force_magnitude * 1000  # meV/Å
        })
    
    return results

# 加载MOF
mof = read("mof_structure.cif")

# 使用MACE+D3组合计算器（推荐）
calc = create_mace_d3_calculator(device="cuda")

# 在孔道内生成采样点
cell = mof.get_cell()
# 生成3x3x3网格
grid_points = []
for i in np.linspace(0.3, 0.7, 3):
    for j in np.linspace(0.3, 0.7, 3):
        for k in np.linspace(0.3, 0.7, 3):
            pos = i * cell[0] + j * cell[1] + k * cell[2]
            grid_points.append(pos)

# 测试不同分子
molecules_to_test = ['CO2', 'H2O', 'CH4', 'N2']

for mol in molecules_to_test:
    print(f"\n=== {mol} Adsorption Forces ===")
    try:
        results = calculate_adsorption_forces(mof, mol, calc, grid_points)
        
        # 统计
        force_mags = [r['force_magnitude'] for r in results]
        print(f"Mean force: {np.mean(force_mags):.2f} meV/Å")
        print(f"Max force: {np.max(force_mags):.2f} meV/Å")
        print(f"Min force: {np.min(force_mags):.2f} meV/Å")
        
    except Exception as e:
        print(f"Failed: {e}")
```

---

## 3. 基于ASE/Phonopy的可扩展推理任务

以下是可以使用ASE和Phonopy库（无需额外软件）执行的扩展推理任务：

### 3.1 ASE支持的扩展任务

| 任务 | 描述 | MACE支持 | 实现难度 |
|------|------|----------|----------|
| **弹性张量计算** | 计算完整的6×6弹性常数矩阵 | ✅ | 中等 |
| **应力-应变曲线** | 评估材料的非线性力学响应 | ✅ | 简单 |
| **表面能计算** | 计算不同晶面的表面能 | ✅ | 中等 |
| **NEB反应路径** | 过渡态搜索和反应能垒计算 | ✅ | 中等 |
| **自扩散系数** | 通过MD轨迹计算MSD和扩散系数 | ✅ | 简单 |
| **径向分布函数** | 结构分析 | ✅ | 简单 |
| **能量-体积曲线** | 状态方程拟合 | ✅ | 简单 |

### 3.2 Phonopy支持的扩展任务

| 任务 | 描述 | MACE支持 | 实现难度 |
|------|------|----------|----------|
| **声子色散关系** | 计算声子能带结构 | ✅ | 中等 |
| **声子态密度** | 计算振动态密度 | ✅ | 中等 |
| **热膨胀系数** | 准谐近似下的热膨胀 | ✅ | 较高 |
| **Grüneisen参数** | 评估非谐效应 | ✅ | 较高 |
| **自由能计算** | Helmholtz振动自由能 | ✅ | 中等 |
| **熵计算** | 振动熵 | ✅ | 中等 |
| **零点能** | 量子零点振动能 | ✅ | 中等 |

### 3.3 弹性张量计算示例

```python
from ase.io import read
from mace.calculators import mace_mp
import numpy as np

def calculate_elastic_tensor(atoms, calc, delta=0.01):
    """
    计算6x6弹性张量
    使用应力-应变关系: σ = C · ε
    """
    atoms.calc = calc
    
    # Voigt notation: 1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy
    C = np.zeros((6, 6))
    
    # 6种独立应变
    strain_patterns = [
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),  # ε_xx
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),  # ε_yy
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),  # ε_zz
        np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),  # ε_yz
        np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),  # ε_xz
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),  # ε_xy
    ]
    
    cell0 = atoms.get_cell()
    
    for i, strain_pattern in enumerate(strain_patterns):
        # 正向应变
        strained_cell_plus = cell0 + delta * strain_pattern @ cell0
        atoms_plus = atoms.copy()
        atoms_plus.set_cell(strained_cell_plus, scale_atoms=True)
        atoms_plus.calc = calc
        stress_plus = atoms_plus.get_stress(voigt=True)  # 6-component Voigt
        
        # 负向应变
        strained_cell_minus = cell0 - delta * strain_pattern @ cell0
        atoms_minus = atoms.copy()
        atoms_minus.set_cell(strained_cell_minus, scale_atoms=True)
        atoms_minus.calc = calc
        stress_minus = atoms_minus.get_stress(voigt=True)
        
        # 中心差分
        C[:, i] = (stress_plus - stress_minus) / (2 * delta)
    
    # 转换单位: eV/Å³ → GPa
    C_GPa = C * 160.21766208
    
    return C_GPa

# 使用
atoms = read("optimized.cif")
calc = mace_mp(model="medium", device="cuda")
C = calculate_elastic_tensor(atoms, calc)

print("Elastic tensor (GPa):")
print(C)

# 计算导出量
# Voigt平均体积模量
K_V = (C[0,0] + C[1,1] + C[2,2] + 2*(C[0,1] + C[1,2] + C[0,2])) / 9
print(f"Bulk modulus (Voigt): {K_V:.2f} GPa")

# Voigt平均剪切模量
G_V = (C[0,0] + C[1,1] + C[2,2] - C[0,1] - C[1,2] - C[0,2] + 3*(C[3,3] + C[4,4] + C[5,5])) / 15
print(f"Shear modulus (Voigt): {G_V:.2f} GPa")
```

---

## 4. 任务可行性总结

### 4.1 论文基准任务可行性

| 任务类别 | 任务名称 | MACE支持 | 所需库 | 备注 |
|----------|----------|----------|--------|------|
| **静态建模** | 结构优化 | ✅ | ASE | 完全支持 |
| | 单点能量计算 | ✅ | ASE | 完全支持 |
| **动力学建模** | NPT分子动力学稳定性 | ✅ | ASE | 完全支持 |
| | 配位环境稳定性（多种金属） | ✅ | ASE | 完全支持 |
| | 高温稳定性测试 | ✅ | ASE | 完全支持 |
| **体相性质** | 体积模量 | ✅ | ASE | 完全支持 |
| | 热容 | ✅ | ASE + Phonopy | 完全支持 |
| **主客体相互作用** | 相互作用能（多种气体） | ✅ | ASE | 完全支持 |
| | 吸附力预测（多种分子） | ✅ | ASE | 完全支持 |

### 4.2 扩展任务可行性

| 任务 | MACE支持 | 所需库 | 备注 |
|------|----------|--------|------|
| 弹性张量计算 | ✅ | ASE | 完全支持 |
| 声子色散关系 | ✅ | Phonopy | 完全支持 |
| 声子态密度 | ✅ | Phonopy | 完全支持 |
| 自由能计算 | ✅ | Phonopy | 完全支持 |
| 熵计算 | ✅ | Phonopy | 完全支持 |
| NEB反应路径 | ✅ | ASE | 完全支持 |
| 自扩散系数 | ✅ | ASE | 完全支持 |
| 热膨胀系数 | ✅ | Phonopy (准谐近似) | 完全支持 |

### 4.3 MACE系列模型推荐

| 应用场景 | 推荐模型 | 理由 |
|----------|----------|------|
| **通用MOF模拟** | MACE-OMAT-0 | 包含非平衡数据，MD稳定性好 |
| **高精度计算** | MACE-MATPES-r2SCAN-0 | r2SCAN泛函精度更高 |
| **快速筛选** | MACE-MP-0b3 (medium) | 速度与精度平衡 |
| **扩展化学覆盖** | MACE-MPA-0 | Alexandria数据增加化学多样性 |

---

## 附录：CPU/GPU代码快速切换

在所有代码示例中，可以通过以下方式快速切换CPU/GPU：

```python
import torch

# 方法1：自动检测
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 方法2：强制使用CPU
DEVICE = "cpu"

# 方法3：强制使用GPU（需要CUDA可用）
DEVICE = "cuda"

# 然后在创建计算器时使用
from mace.calculators import mace_mp
calc = mace_mp(model="medium", device=DEVICE)
```

---

*文档生成时间: 2026年1月7日*

*基于MOFSimBench论文分析和MACE官方文档整理*
