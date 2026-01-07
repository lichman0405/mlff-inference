# MOFSimBench 论文分析：机器学习力场与推理任务

> **论文标题**: MOFSimBench: Evaluating Universal Machine Learning Interatomic Potentials In Metal–Organic Framework Molecular Modeling
> 
> **作者**: Hendrik Kraß, Ju Huang, Seyed Mohamad Moosavi
> 
> **论文链接**: [arXiv:2507.11806](https://arxiv.org/abs/2507.11806)
> 
> **代码仓库**: [https://github.com/AI4ChemS/mofsim-bench](https://github.com/AI4ChemS/mofsim-bench)

---

## 目录

1. [论文概述](#1-论文概述)
2. [机器学习力场分类与详细信息](#2-机器学习力场分类与详细信息)
   - [2.1 等变图神经网络 (Equivariant GNN)](#21-等变图神经网络-equivariant-gnn)
   - [2.2 图Transformer (Graph Transformer)](#22-图transformer-graph-transformer)
   - [2.3 图网络模拟器 (GNS)](#23-图网络模拟器-gns)
   - [2.4 图基函数方法 (Graph-Basis Functions)](#24-图基函数方法-graph-basis-functions)
   - [2.5 经典力场基线](#25-经典力场基线)
   - [2.6 其他引用的机器学习力场](#26-其他引用的机器学习力场)
3. [训练数据集说明](#3-训练数据集说明)
4. [论文中评估的推理任务](#4-论文中评估的推理任务)
5. [基于相同库可扩展的推理任务](#5-基于相同库可扩展的推理任务)
6. [关键发现与结论](#6-关键发现与结论)
7. [参考文献](#7-参考文献)

---

## 1. 论文概述

### 1.1 研究背景

通用机器学习原子间势能（Universal Machine Learning Interatomic Potentials, uMLIPs）已成为加速原子模拟的强大工具，提供接近量子计算精度的可扩展高效建模。然而，它们在实际应用中的可靠性和有效性仍然是一个开放问题。

金属有机框架（Metal-Organic Frameworks, MOFs）及相关纳米多孔材料是高度多孔的晶体，在碳捕获、能源存储和催化应用中具有重要意义。由于其化学多样性、结构复杂性（包括孔隙率和配位键）以及在现有训练数据集中的缺失，建模纳米多孔材料对uMLIPs提出了独特挑战。

### 1.2 研究目标

本论文引入了**MOFSimBench**，一个评估uMLIPs在纳米多孔材料关键材料建模任务上的基准，包括：
- 结构优化
- 分子动力学（MD）稳定性
- 体相性质预测（如体积模量和热容）
- 主客体相互作用

### 1.3 主要发现

- 评估了超过20个来自不同架构的模型
- 表现最佳的uMLIPs在所有任务上持续优于经典力场和微调的机器学习势能
- **数据质量**（特别是训练集多样性和非平衡构型的包含）比模型架构在决定性能方面起着更关键的作用

---

## 2. 机器学习力场分类与详细信息

### 2.1 等变图神经网络 (Equivariant GNN)

#### MACE 系列

| 模型名称 | 训练数据集 | 特点 | 论文链接 | GitHub |
|----------|------------|------|----------|--------|
| **MACE-MP-0a** | MPtraj | 第一个通用MACE模型 | [arXiv:2206.07697](https://arxiv.org/abs/2206.07697) | [ACEsuit/mace](https://github.com/ACEsuit/mace) |
| **MACE-MP-0b3 (medium)** | MPtraj | 改进的对排斥、正确的孤立原子、更好的高压稳定性 | [arXiv:2401.00096](https://arxiv.org/abs/2401.00096) | [ACEsuit/mace](https://github.com/ACEsuit/mace) |
| **MACE-MPA-0** | MPtraj + sAlex | 扩展训练数据 | 同上 | [ACEsuit/mace](https://github.com/ACEsuit/mace) |
| **MACE-OMAT-0** | OMat24 | 使用OMat24数据集训练 | 同上 | [ACEsuit/mace](https://github.com/ACEsuit/mace) |
| **MACE-MATPES-r2SCAN-0** | MATPES-r2SCAN | 使用r2SCAN泛函数据训练 | [arXiv:2503.04070](http://arxiv.org/abs/2503.04070) | [ACEsuit/mace](https://github.com/ACEsuit/mace) |
| **MACE-MP-MOF0** (微调) | 127个MOF结构 + 4764 DFT计算 | MOF特定微调，支持元素受限 | Elena et al. 2025 | - |
| **MACE-DAC-1** (微调) | 主客体相互作用数据 | 用于CO₂和H₂O吸附模拟 | Lim et al. 2025 | - |

**MACE架构特点**：
- 高阶等变消息传递神经网络
- 基于原子集群展开（ACE）理论
- 通过能量梯度计算力（保守力）

#### MatterSim

| 模型名称 | 训练数据集 | 特点 | 论文链接 | GitHub |
|----------|------------|------|----------|--------|
| **MatterSim-v1 (5M)** | 专有数据集 | 基于M3GNet架构，使用不确定性感知的主动学习管道 | [arXiv:2405.04967](http://arxiv.org/abs/2405.04967) | [microsoft/mattersim](https://github.com/microsoft/mattersim) |

**MatterSim架构特点**：
- 基于M3GNet架构
- 使用模型集成进行不确定性估计
- 三体相互作用建模

#### SevenNet 系列

| 模型名称 | 训练数据集 | 特点 | 论文链接 | GitHub |
|----------|------------|------|----------|--------|
| **SevenNet-0** | MPtraj | 基础模型 | Park et al. 2024 | [MDIL-SNU/SevenNet](https://github.com/MDIL-SNU/SevenNet) |
| **SevenNet-l3i5** | MPtraj | 增加复杂度 | 同上 | [MDIL-SNU/SevenNet](https://github.com/MDIL-SNU/SevenNet) |
| **SevenNet-ompa** | OMat24 + sAlex + MPtraj | 最佳性能版本 | 同上 | [MDIL-SNU/SevenNet](https://github.com/MDIL-SNU/SevenNet) |

**SevenNet架构特点**：
- 基于NequIP架构
- 可扩展的并行算法用于分子动力学模拟

---

### 2.2 图Transformer (Graph Transformer)

#### EquiformerV2 系列

| 模型名称 | 训练数据集 | 特点 | 论文链接 | GitHub |
|----------|------------|------|----------|--------|
| **eqV2-M OMat MPtrj-sAlex** | OMat24 → MPtraj + sAlex微调 | 非保守力输出 | [arXiv:2306.12059](https://arxiv.org/abs/2306.12059) | [atomicarchitects/equiformer_v2](https://github.com/atomicarchitects/equiformer_v2) |
| **eqV2-M-DeNS** | MPtraj | 仅MPtraj训练 | 同上 | [atomicarchitects/equiformer_v2](https://github.com/atomicarchitects/equiformer_v2) |

**EquiformerV2架构特点**：
- 改进的等变Transformer架构
- 可扩展到更高阶表示
- 直接输出力（非保守力），计算效率高但稳定性较差

#### eSEN 系列

| 模型名称 | 训练数据集 | 特点 | 论文链接 | GitHub |
|----------|------------|------|----------|--------|
| **eSEN-30M-OAM** | OMat24 + MPtraj + sAlex | 最佳性能模型之一 | [arXiv:2502.12147](https://arxiv.org/abs/2502.12147) | [FAIR-Chem/fairchem](https://github.com/FAIR-Chem/fairchem) |
| **eSEN-30M-MP** | MPtraj | 仅MPtraj训练 | 同上 | [FAIR-Chem/fairchem](https://github.com/FAIR-Chem/fairchem) |

**eSEN架构特点**：
- 设计确保平滑和表达性的势能面
- 通过严格的架构决策评估开发
- 保守力（通过能量梯度计算）
- **在本基准中表现最佳的模型之一**

---

### 2.3 图网络模拟器 (GNS)

#### Orb 系列

| 模型名称 | 训练数据集 | 力类型 | 特点 | 论文链接 | GitHub |
|----------|------------|--------|------|----------|--------|
| **orb-d3-v2** | MPtraj + Alexandria (D3校正) | 非保守力 | 直接预测D3校正输出 | [arXiv:2410.22570](https://arxiv.org/abs/2410.22570) | [orbital-materials/orb-models](https://github.com/orbital-materials/orb-models) |
| **orb-mptraj-only-v2** | MPtraj | 非保守力 | 不预测D3校正 | 同上 | [orbital-materials/orb-models](https://github.com/orbital-materials/orb-models) |
| **orb-v3-con-inf-omat** | OMat24 | **保守力** | 无上限邻居限制 | [arXiv:2504.06231](http://arxiv.org/abs/2504.06231) | [orbital-materials/orb-models](https://github.com/orbital-materials/orb-models) |
| **orb-v3-con-inf-mpa** | MPtraj + Alexandria | **保守力** | 无上限邻居限制 | 同上 | [orbital-materials/orb-models](https://github.com/orbital-materials/orb-models) |

**Orb架构特点**：
- 基于图网络模拟器（GNS）架构
- 通过正则化和数据增强学习等变性（而非预定义）
- v3版本引入保守力，显著提高稳定性
- **orb-v3-omat是本基准中表现最佳的模型之一**

---

### 2.4 图基函数方法 (Graph-Basis Functions)

#### GRACE 系列

| 模型名称 | 训练数据集 | 特点 | 论文链接 | GitHub |
|----------|------------|------|----------|--------|
| **GRACE-2L-MP (r6)** | MPtraj | 基础模型 | Bochkarev et al. 2024 | [ICAMS/grace-tensorpotential](https://github.com/ICAMS/grace-tensorpotential) |
| **GRACE-2L-OMAT** | OMat24 | 使用OMat24训练 | 同上 | [ICAMS/grace-tensorpotential](https://github.com/ICAMS/grace-tensorpotential) |
| **GRACE-2L-OAM (r6)** | OMat24预训练 → sAlex + MPtraj微调 | 综合训练策略 | 同上 | [ICAMS/grace-tensorpotential](https://github.com/ICAMS/grace-tensorpotential) |

**GRACE架构特点**：
- 将原子集群展开（ACE）扩展到图基函数
- 半局域相互作用建模
- 高效且准确

---

### 2.5 经典力场基线

| 力场名称 | 类型 | 描述 | 参考文献 | 软件集成 |
|----------|------|------|----------|----------|
| **UFF** | 通用力场 | 1992年引入，覆盖全元素周期表 | Rappe et al. 1992 | LAMMPS内置 |
| **UFF4MOF** | UFF扩展 | 针对MOF的配位环境扩展 | Coupry et al. 2016 | [lammps_interface](https://github.com/peteboyd/lammps_interface) |
| **MOF-FF** | MOF专用 | 第一性原理衍生的MOF力场 | Bureekaew et al. 2013 | - |

**经典力场局限性**：
- 刚性函数形式
- 依赖固定参数
- 无法适应新的配位环境
- 在本基准中表现较差

---

### 2.6 其他引用的机器学习力场

| 模型名称 | 架构类型 | 描述 | 论文链接 | GitHub |
|----------|----------|------|----------|--------|
| **M3GNet** | GNN | 通用图深度学习原子间势能 | Chen and Ong 2022 | [materialsvirtuallab/m3gnet](https://github.com/materialsvirtuallab/m3gnet) |
| **CHGNet** | GNN | 电荷感知的预训练通用神经网络势能 | Deng et al. 2023 | [CederGroupHub/chgnet](https://github.com/CederGroupHub/chgnet) |
| **NequIP** | 等变GNN | E(3)等变图神经网络 | Batzner et al. 2022 | [mir-group/nequip](https://github.com/mir-group/nequip) |
| **SchNet** | 连续卷积GNN | 连续滤波器卷积网络 | Schütt et al. 2017 | [atomistic-machine-learning/schnetpack](https://github.com/atomistic-machine-learning/schnetpack) |
| **PACE/ACE** | 原子集群展开 | 高效的原子集群展开实现 | Lysogorskiy et al. 2021, Drautz 2019 | [ACEsuit](https://github.com/ACEsuit) |
| **GAP** | 高斯逼近势 | 基于高斯过程的机器学习势能 | Deringer and Csányi 2017 | [libAtoms/GAP](https://github.com/libAtoms/GAP) |

---

## 3. 训练数据集说明

### 3.1 主要训练数据集对比

| 数据集 | 数据规模 | 内容特点 | 对模型性能的影响 | 参考文献 |
|--------|----------|----------|------------------|----------|
| **MPtraj** | 大规模 | 主要包含平衡构型（最小能量附近） | 基础性能，但在非平衡区域存在"软化"问题 | Deng et al. 2023 |
| **Alexandria/sAlex** | 大规模 | 扩展化学覆盖范围 | 提高化学多样性 | Schmidt et al. 2023 |
| **OMat24** | 超大规模 | **包含大量非平衡构型** | 显著提高MD稳定性和性质预测准确性 | Barroso-Luque et al. 2024 |
| **MATPES** | 大规模 | r2SCAN泛函计算 | 更高精度的参考数据 | Kaplan et al. 2025 |

### 3.2 数据集对性能的关键影响

论文的核心发现之一是**训练数据质量比模型架构更重要**：

1. **非平衡构型的重要性**：
   - 仅在平衡构型上训练会导致势能面"软化"
   - 表现为体积模量低估、热容高估
   - OMat24包含非平衡数据，显著改善这一问题

2. **化学多样性**：
   - 覆盖更多元素和配位环境
   - 提高对新材料的泛化能力

3. **性能提升时间线**：
   ```
   MPtraj-only → MPtraj + Alexandria → OMat24-based
   （性能逐步提升，与数据集扩展强相关）
   ```

---

## 4. 论文中评估的推理任务

### 4.1 静态建模与结构优化

#### 4.1.1 结构优化 (Structural Optimization)

| 项目 | 详细信息 |
|------|----------|
| **任务描述** | 优化原子位置和晶胞参数以找到最小能量构型 |
| **评估指标** | 收敛率、体积偏差（与DFT参考的差异<10%） |
| **方法** | FrechetCellFilter + LBFGS优化器，收敛标准10⁻³ eV/Å |
| **最佳模型** | eSEN-OAM, orb-v3-omat（89%成功率） |
| **关键发现** | 非保守力模型（orb-d3-v2, eqV2-OMsA）收敛率最低 |

#### 4.1.2 单点能量计算

| 项目 | 详细信息 |
|------|----------|
| **任务描述** | 计算给定构型的势能 |
| **评估指标** | MAE (eV/atom) |
| **参考数据** | QMOF数据集的DFT能量 |

### 4.2 动力学建模

#### 4.2.1 NpT分子动力学稳定性

| 项目 | 详细信息 |
|------|----------|
| **任务描述** | 在NpT系综下进行50ps的MD模拟 |
| **评估指标** | 体积漂移（初始vs最终，阈值±10%） |
| **模拟条件** | 300K, 1 bar, 时间步长1fs |
| **最佳模型** | eSEN-OAM, orb-v3-omat, MatterSim |
| **关键发现** | 在非平衡数据上训练的模型更稳定 |

#### 4.2.2 配位环境稳定性

| 项目 | 详细信息 |
|------|----------|
| **任务描述** | 评估金属配位数在MD过程中的变化 |
| **评估指标** | 配位数变化（初始vs最终） |
| **测试材料** | 13个Cu-MOF，配位数从2到6 |
| **模拟条件** | 10ps@300K → 10ps@400K → 10ps@300K |
| **最佳模型** | SevenNet-ompa, GRACE-OMAT, MatterSim, eSEN-OAM, orb-v3-omat |
| **关键发现** | uMLIPs无需预定义配位即可保持局部环境 |

#### 4.2.3 高温稳定性测试

| 项目 | 详细信息 |
|------|----------|
| **任务描述** | 渐进升温模拟（300K到1000K） |
| **模拟条件** | 100K步进，每步20ps，共160ps NpT |

### 4.3 体相性质预测

#### 4.3.1 体积模量 (Bulk Modulus)

| 项目 | 详细信息 |
|------|----------|
| **任务描述** | 通过Birch-Murnaghan状态方程计算体积模量 |
| **评估指标** | MAE (GPa), MAPE (%) |
| **计算方法** | 体积应变±4%（11步），拟合EOS |
| **最佳模型** | eSEN-OAM (MAE 2.64 GPa), MACE-MP-MOF0 (MAE 3.14 GPa), SevenNet-ompa (MAE 3.35 GPa) |
| **关键发现** | 非保守力模型误差最大（orb-d3-v2: MAE 72.29 GPa） |

#### 4.3.2 热容 (Heat Capacity)

| 项目 | 详细信息 |
|------|----------|
| **任务描述** | 通过声子计算获得300K的定容热容 |
| **评估指标** | MAE (J/K/g), MAPE (%) |
| **计算方法** | Phonopy，有限差分法（位移0.01Å） |
| **测试数据** | 231个MOF、COF和沸石结构 |
| **最佳模型** | orb-v3-omat (MAE 0.018, MAPE 2.3%), MACE-MP-MOF0 (MAE 0.020), eSEN-OAM (MAE 0.024) |
| **关键发现** | 所有模型存在系统性高估，与势能面软化相关 |

### 4.4 主客体相互作用

#### 4.4.1 相互作用能预测

| 项目 | 详细信息 |
|------|----------|
| **任务描述** | 预测CO₂和H₂O与MOF框架的相互作用能 |
| **评估指标** | 能量误差 (meV) |
| **测试数据** | GoldDAC数据集，312个结构，26个MOF |
| **相互作用类型** | 排斥区(R)、平衡区(E)、弱吸引区(W) |
| **最佳模型** | MatterSim, eSEN-OAM（优于微调的MACE-DAC-1） |

#### 4.4.2 吸附力预测

| 项目 | 详细信息 |
|------|----------|
| **任务描述** | 预测气体分子在MOF中的受力 |
| **评估指标** | 力误差 (meV/Å) |
| **关键发现** | 架构在此任务中比数据更重要 |

---

## 5. 基于相同库可扩展的推理任务

以下任务可使用相同的工具链（ASE、Phonopy、LAMMPS等）和uMLIP模型执行：

### 5.1 热力学性质计算

| 任务 | 方法/库 | 描述 | 科学应用 |
|------|---------|------|----------|
| **声子谱计算** | Phonopy | 计算声子色散关系和态密度 | 热学性质、振动光谱预测 |
| **热导率** | Phono3py / MD | 通过声子-声子散射或Green-Kubo方法 | 热管理材料设计 |
| **自由能计算** | 热力学积分 | Helmholtz/Gibbs自由能 | 相稳定性预测 |
| **熵计算** | 声子方法 | 振动熵贡献 | 完整热力学描述 |
| **热膨胀系数** | 准谐近似 / MD | 温度依赖的晶格参数变化 | 热应力分析 |

### 5.2 力学性质计算

| 任务 | 方法/库 | 描述 | 科学应用 |
|------|---------|------|----------|
| **弹性张量** | ASE + 应力-应变分析 | 完整的Cᵢⱼ弹性常数 | 各向异性力学性质 |
| **杨氏模量、剪切模量** | 从弹性张量导出 | Voigt-Reuss-Hill平均 | 机械强度评估 |
| **泊松比** | 从弹性张量导出 | 横向与纵向应变比 | 材料变形特性 |
| **应力-应变曲线** | MD或静态变形 | 非线性力学响应 | 材料强度极限 |
| **断裂韧性** | MD模拟 | 裂纹扩展行为 | 结构完整性 |

### 5.3 动力学与输运性质

| 任务 | 方法/库 | 描述 | 科学应用 |
|------|---------|------|----------|
| **自扩散系数** | MD + MSD分析 | 框架原子的扩散 | 框架稳定性 |
| **客体分子扩散** | NVE/NVT MD | 气体/溶剂分子的扩散系数 | 气体分离性能 |
| **粘度** | 平衡MD / NEMD | 流体在孔道中的输运 | 流体动力学 |
| **离子电导率** | MD + 电流自相关 | 离子输运 | 电化学应用 |

### 5.4 吸附与分离性质

| 任务 | 方法/库 | 描述 | 科学应用 |
|------|---------|------|----------|
| **吸附等温线** | GCMC (需集成RASPA) | 压力-吸附量关系 | 气体存储容量 |
| **等量吸附热** | Widom插入 / MD | 吸附焓 | 吸附强度评估 |
| **选择性吸附** | 多组分GCMC | 混合气体分离因子 | 气体分离应用 |
| **吸附位点识别** | 能量扫描 | 优先吸附位置 | 活性位点定位 |
| **Henry常数** | Widom插入 | 低压吸附行为 | 初始筛选 |

### 5.5 相变与稳定性

| 任务 | 方法/库 | 描述 | 科学应用 |
|------|---------|------|----------|
| **负热膨胀(NTE)预测** | 变温MD / 准谐近似 | 异常热膨胀行为 | 特殊功能材料 |
| **相变温度** | 自由能计算 | 相转变点确定 | 相图构建 |
| **化学稳定性** | 缺陷形成能计算 | 点缺陷的热力学稳定性 | 材料耐久性 |
| **水稳定性** | 显式水分子MD | 水合/水解行为 | 实际应用条件 |
| **框架柔性(呼吸效应)** | 变压MD | 压力诱导的相变 | 柔性MOF研究 |

### 5.6 表面与界面性质

| 任务 | 方法/库 | 描述 | 科学应用 |
|------|---------|------|----------|
| **表面能** | 切片模型计算 | 不同晶面的表面能 | 形貌预测 |
| **功函数** | 静电势分析 | 电子逸出功 | 电化学界面 |
| **吸附构型优化** | 表面吸附计算 | 分子在表面的最优取向 | 催化应用 |

### 5.7 催化与反应性质

| 任务 | 方法/库 | 描述 | 科学应用 |
|------|---------|------|----------|
| **反应能垒** | NEB方法 | 过渡态搜索 | 催化机理 |
| **反应路径** | Dimer方法 / MD | 最小能量路径 | 反应动力学 |
| **活化能** | Arrhenius分析 | 温度依赖的反应速率 | 催化效率 |

### 5.8 高通量筛选应用

| 任务 | 方法/库 | 描述 | 科学应用 |
|------|---------|------|----------|
| **大规模结构优化** | 自动化工作流 (AiiDA) | 批量结构弛豫 | 材料数据库构建 |
| **稳定性预筛选** | 快速MD测试 | 排除不稳定候选 | 高效筛选 |
| **多目标优化** | 贝叶斯优化 + MLIP | 同时优化多个性质 | 逆向设计 |
| **性质预测** | ML + MLIP | 快速性质估算 | 加速发现 |

---

## 6. 关键发现与结论

### 6.1 模型性能排名

基于论文中的综合评估，各模型的相对性能如下：

| 排名 | 模型 | 优势 | 劣势 |
|------|------|------|------|
| 1 | **eSEN-OAM** | 所有任务表现优异，最窄误差分布 | - |
| 2 | **orb-v3-omat** | 热容预测最佳，整体稳定 | - |
| 3 | **MatterSim** | 主客体相互作用最佳 | 专有数据集 |
| 4 | **SevenNet-ompa** | 配位环境稳定性好 | - |
| 5 | **MACE-MP-MOF0** (微调) | MOF特定任务准确 | 元素覆盖受限 |
| ... | **MACE-OMAT-0** | 比MP-0版本改进明显 | - |
| 较差 | **orb-d3-v2** | - | 非保守力导致不稳定 |
| 最差 | **eqV2-OMsA** | - | 非保守力导致不稳定 |

### 6.2 核心发现

1. **数据质量 > 模型架构**
   - 训练数据的多样性和非平衡构型的包含比架构选择更重要
   - OMat24数据集的发布带来了显著的性能提升

2. **保守力的重要性**
   - 非保守力模型（直接预测力）在优化和MD任务中表现最差
   - 通过能量梯度计算力确保物理一致性

3. **uMLIPs已超越经典力场**
   - 最佳uMLIPs在所有任务上优于UFF/UFF4MOF
   - 标志着MOF建模方法的范式转变

4. **微调的权衡**
   - 微调可提高特定任务性能但牺牲通用性
   - 随着通用模型改进，微调的必要性降低

### 6.3 实践建议

| 应用场景 | 推荐模型 | 理由 |
|----------|----------|------|
| **通用MOF建模** | eSEN-OAM, orb-v3-omat | 最佳综合性能 |
| **吸附模拟** | MatterSim, eSEN-OAM | 主客体相互作用最佳 |
| **力学性质** | eSEN-OAM, MACE-OMAT-0 | 体积模量预测准确 |
| **热学性质** | orb-v3-omat, eSEN-OAM | 热容预测准确 |
| **长时间MD** | eSEN-OAM, MatterSim | 稳定性最好 |
| **特定MOF深入研究** | MACE-MP-MOF0 (如适用) | 特定系统最准确 |

---

## 7. 参考文献

### 核心论文

1. **MOFSimBench**: Kraß, H.; Huang, J.; Moosavi, S.M. *MOFSimBench: Evaluating Universal Machine Learning Interatomic Potentials In Metal–Organic Framework Molecular Modeling.* arXiv:2507.11806 (2025)

### 模型论文

2. **MACE**: Batatia, I. et al. *MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields.* arXiv:2206.07697 (2023)

3. **MACE Foundation**: Batatia, I. et al. *A foundation model for atomistic materials chemistry.* arXiv:2401.00096 (2024)

4. **MatterSim**: Yang, H. et al. *MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures.* arXiv:2405.04967 (2024)

5. **EquiformerV2**: Liao, Y.-L. et al. *EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations.* arXiv:2306.12059 (2024)

6. **eSEN**: Fu, X. et al. *Learning Smooth and Expressive Interatomic Potentials for Physical Property Prediction.* arXiv:2502.12147 (2025)

7. **Orb**: Neumann, M. et al. *Orb: A Fast, Scalable Neural Network Potential.* arXiv:2410.22570 (2024)

8. **Orb-v3**: Rhodes, B. et al. *Orb-v3: atomistic simulation at scale.* arXiv:2504.06231 (2025)

9. **SevenNet**: Park, Y. et al. *Scalable Parallel Algorithm for Graph Neural Network Interatomic Potentials in Molecular Dynamics Simulations.* J. Chem. Theory Comput. (2024)

10. **GRACE**: Bochkarev, A. et al. *Graph Atomic Cluster Expansion for Semilocal Interactions beyond Equivariant Message Passing.* Phys. Rev. X (2024)

### 数据集论文

11. **OMat24**: Barroso-Luque, L. et al. *Open Materials 2024 (OMat24) Inorganic Materials Dataset and Models.* arXiv:2410.12771 (2024)

12. **MPtraj/CHGNet**: Deng, B. et al. *CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling.* Nature Machine Intelligence (2023)

13. **Alexandria**: Schmidt, J. et al. *Machine-Learning-Assisted Determination of the Global Zero-Temperature Phase Diagram of Materials.* Advanced Materials (2023)

### 工具与方法

14. **ASE**: Hjorth Larsen, A. et al. *The atomic simulation environment—a Python library for working with atoms.* J. Phys.: Condens. Matter (2017)

15. **Phonopy**: Togo, A. et al. *Implementation strategies in phonopy and phono3py.* J. Phys. Condens. Matter (2023)

16. **LAMMPS**: Thompson, A.P. et al. *LAMMPS - A flexible simulation tool for particle-based materials modeling.* Computer Physics Communications (2022)

---

*文档生成时间: 2026年1月7日*

*基于论文 arXiv:2507.11806v1 分析整理*
