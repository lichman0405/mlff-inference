# MLFF-inference

**机器学习力场推理工具包**

一个全面的机器学习力场（MLFF）推理包集合，包含最先进的模型，专为材料科学和分子模拟优化。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📋 项目概述

本项目为 MOFSimBench 基准测试中的 7 个领先机器学习力场模型提供统一、易用的推理接口。每个模型包提供：

- 🚀 **统一 API**：所有模型采用一致的接口
- 🔧 **丰富功能**：单点计算、结构优化、分子动力学、声子计算、力学性质
- 💻 **CPU/GPU 支持**：灵活的部署选项
- 📦 **独立包装**：可独立安装和使用
- 🌐 **完整文档**：中英文双语文档

## 🎯 支持的模型

| 排名 | 模型 | 软件包 | 关键特性 | 主要应用场景 |
|------|------|--------|----------|-------------|
| 2 | **MACE** | `MACE_inference` | 等变消息传递，高精度 | 通用材料，有机分子 |
| 2 | **Orb** | `Orb_inference` | 快速推理，多样化预训练数据集 | 多材料预测 |
| 1 | **eSCN** | `eSEN_inference` | 等变球面通道，OCP 数据集 | 催化，表面反应 |
| 3 | **MatterSim** | `MatterSim_inference` | M3GNet 架构，不确定性估计 | MOF 吸附，通用材料 |
| 4 | **SevenNet** | `SevenNet_inference` | 7 层等变 GNN，力准确度高 | 分子模拟，动力学 |
| 5 | **EquiformerV2** | `EquiformerV2_inference` | E(3) 等变 Transformer | 大规模系统，OCP |
| 6 | **GRACE** | `GRACE_inference` | 图基函数，DGL 后端 | MOF 气体吸附，快速计算 |

*排名基于 MOFSimBench 性能*

## 🚀 快速开始

### 安装

每个模型包可以独立安装：

```bash
# 示例：安装 MACE（CPU 版本）
cd MACE_inference/mace-inference
pip install -e ".[cpu]"

# 或者安装 GPU 版本
pip install -e ".[gpu]"
```

### 基本使用

所有模型共享统一的 API：

```python
from mace_inference import MACEInference
from ase.io import read

# 初始化模型
model = MACEInference(model_path="path/to/model.pth", device="cuda")

# 加载结构
atoms = read("structure.cif")

# 单点能量和力计算
result = model.calculate(atoms)
print(f"能量: {result['energy']} eV")
print(f"力的形状: {result['forces'].shape}")

# 结构优化
optimized = model.optimize(atoms, fmax=0.01)
optimized.write("optimized.cif")

# 分子动力学
trajectory = model.run_md(
    atoms,
    temperature=300,
    steps=1000,
    timestep=1.0
)
```

### 命令行界面

每个包都提供完整的命令行工具：

```bash
# 单点计算
mace-inference single-point structure.cif --model model.pth

# 结构优化
mace-inference optimize structure.cif --fmax 0.01 --output opt.cif

# 分子动力学
mace-inference md structure.cif --temp 300 --steps 10000

# 声子计算
mace-inference phonon structure.cif --supercell 2 2 2

# 体模量计算
mace-inference bulk-modulus structure.cif

# 模型信息
mace-inference info --model model.pth
```

## 📦 项目结构

```
MLFF-inference/
├── README.md                      # 英文版本
├── README-cn.md                   # 本文件（中文）
├── docs/                          # 文档
│   └── MOFSimBench_论文分析_*.md
├── MACE_inference/                # MACE 模型包
│   ├── requirements-cpu.txt
│   ├── requirements-gpu.txt
│   ├── INSTALL.md
│   └── mace-inference/
│       ├── src/mace_inference/
│       ├── examples/
│       ├── tests/
│       └── docs/
├── Orb_inference/                 # Orb 模型包
├── eSEN_inference/                # eSCN 模型包
├── MatterSim_inference/           # MatterSim 模型包
├── SevenNet_inference/            # SevenNet 模型包
├── EquiformerV2_inference/        # EquiformerV2 模型包
└── GRACE_inference/               # GRACE 模型包
```

## 🔧 可用任务

所有模型包支持以下计算任务：

### 1. 单点计算
计算给定结构的能量、力和应力。

### 2. 结构优化
优化原子位置和/或晶格参数以最小化能量。

### 3. 分子动力学 (MD)
- NVE 系综
- NVT 系综（Langevin 恒温器）
- NPT 系综（Berendsen 恒压器）

### 4. 声子计算
使用有限位移法计算声子色散、态密度和热力学性质。

### 5. 力学性质
通过应变-应力关系计算弹性常数和体模量。

### 6. 吸附能（特定模型）
计算 MOF 结构上的气体吸附能（MatterSim、GRACE）。

## 📚 文档

每个模型包包含完整的文档：

- **README.md**：概述和快速开始（英文）
- **QUICKSTART.md**：分步教程（英文）
- **INSTALL_GUIDE.md**：详细安装说明（英文）
- **INSTALL.md**：安装指南（中文）
- **{Model}_API_reference.md**：API 参考文档（中文）
- **{Model}_tasks.md**：任务说明文档（中文）

## 💻 系统要求

### 最低要求
- Python 3.8 或更高版本
- 8 GB 内存
- 10 GB 磁盘空间

### GPU 推荐配置
- CUDA 11.8 或 12.1
- 16 GB 内存
- 具有 8+ GB 显存的 NVIDIA GPU

### 支持的平台
- Linux（Ubuntu 20.04+，CentOS 7+）
- macOS（10.15+）
- Windows 10/11

## 🛠️ 开发

### 运行测试

```bash
cd {model}-inference
pytest tests/
```

### 代码结构

每个模型包遵循一致的结构：

```python
# 核心推理类
class {Model}Inference:
    def __init__(self, model_path, device="cpu")
    def calculate(self, atoms)
    def optimize(self, atoms, fmax=0.05)
    def run_md(self, atoms, temperature, steps)
    def calculate_phonon(self, atoms, supercell)
    def calculate_bulk_modulus(self, atoms)
    
# 实用工具模块
utils/
├── device.py      # 设备管理
└── io.py          # 文件 I/O 操作

# 任务模块
tasks/
├── static.py      # 单点计算和优化
├── dynamics.py    # 分子动力学
├── phonon.py      # 声子计算
└── mechanics.py   # 力学性质
```

## 🤝 贡献

欢迎贡献！请：

1. Fork 本仓库
2. 创建功能分支（`git checkout -b feature/AmazingFeature`）
3. 提交更改（`git commit -m 'Add some AmazingFeature'`）
4. 推送到分支（`git push origin feature/AmazingFeature`）
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见各包中的 LICENSE 文件。

## 🙏 致谢

- **MOFSimBench**：用于评估 MOF 系统上 MLFF 模型的基准框架
- **ASE**：用于结构操作的原子模拟环境
- 各个模型的开发者及其团队：
  - MACE 团队（剑桥大学，洛桑联邦理工学院）
  - Orb 团队（Orbital Materials）
  - eSCN/OCP 团队（Meta AI Research）
  - MatterSim 团队（微软研究院）
  - SevenNet 团队
  - EquiformerV2/OCP 团队（Meta AI Research）
  - GRACE 团队

## 📞 联系方式

如有问题、问题或建议：

- 在 GitHub 上提交 issue
- 邮箱：shadow.li981@gmail.com
- 查看各个包的文档
- 参考原始模型仓库

## 🔗 参考文献

1. MOFSimBench：MOF 系统机器学习力场基准测试
2. MACE：高阶等变消息传递神经网络
3. Orb：材料科学预训练模型
4. eSCN：等变球面通道网络
5. MatterSim：材料深度学习势
6. SevenNet：多层等变图神经网络
7. EquiformerV2：E(3) 等变 Transformer
8. GRACE：材料图基函数

## 📊 引用

如果您在研究中使用本工具包，请引用相关模型论文以及：

```bibtex
@software{mlff_inference,
  title={MLFF-inference: 机器学习力场推理工具包},
  author={Shibo Li},
  year={2026},
  url={https://github.com/lichman0405/mlff-inference}
}
```

---

**注意**：这是一个仅用于推理的工具包。如需模型训练，请参考原始模型仓库。
