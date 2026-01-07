# GRACE Inference 安装指南

> **GRACE**: 基于图注意力机制的高效力场模型  
> **开发团队**: 图机器学习研究团队  
> **特色**: DGL 加速、高效图神经网络、多场景推理任务

---

## 📋 目录

1. [系统要求](#1-系统要求)
2. [安装方法](#2-安装方法)
3. [验证安装](#3-验证安装)
4. [GPU 配置](#4-gpu-配置)
5. [常见问题](#5-常见问题)
6. [依赖说明](#6-依赖说明)

---

## 1. 系统要求

### 最低配置
- **操作系统**: Linux, macOS, Windows (WSL2)
- **Python**: 3.9 - 3.11 (推荐 3.10)
- **内存**: 8GB RAM (推荐 16GB+)
- **磁盘**: 5GB 可用空间

### GPU 版本额外要求
- **GPU**: NVIDIA GPU (计算能力 >= 6.0)
- **CUDA**: 11.8 或 12.1+
- **GPU 显存**: >= 8GB (推荐 >= 16GB)
- **驱动**: NVIDIA Driver >= 450.80.02

---

## 2. 安装方法

### 方法 1: Pip 安装 (推荐)

#### CPU 版本
```bash
# 创建虚拟环境
conda create -n grace-cpu python=3.10
conda activate grace-cpu

# 安装 PyTorch (CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装 DGL (CPU)
pip install dgl -f https://data.dgl.ai/wheels/repo.html

# 安装 GRACE 和依赖
pip install grace-gnn ase phonopy pymatgen

# 安装本推理包
cd GRACE_inference
pip install -r requirements-cpu.txt
```

#### GPU 版本
```bash
# 创建虚拟环境
conda create -n grace-gpu python=3.10
conda activate grace-gpu

# 安装 PyTorch (GPU) - 根据 CUDA 版本选择
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 DGL (GPU) - 根据 CUDA 版本选择
# CUDA 11.8
pip install dgl-cu118 -f https://data.dgl.ai/wheels/repo.html

# CUDA 12.1
pip install dgl-cu121 -f https://data.dgl.ai/wheels/repo.html

# 安装 GRACE 和依赖
pip install grace-gnn ase phonopy pymatgen spglib

# 安装本推理包
cd GRACE_inference
pip install -r requirements-gpu.txt
```

### 方法 2: Conda 完整环境

```bash
# 创建环境并安装所有依赖
conda create -n grace python=3.10
conda activate grace

# 安装 PyTorch (GPU)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# 或安装 PyTorch (CPU)
conda install pytorch cpuonly -c pytorch

# 安装 DGL (需要从 pip 安装)
pip install dgl-cu118 -f https://data.dgl.ai/wheels/repo.html  # GPU 版本
# 或
pip install dgl -f https://data.dgl.ai/wheels/repo.html  # CPU 版本

# 安装其他依赖
pip install grace-gnn ase phonopy pymatgen scipy h5py matplotlib pandas tqdm pyyaml prettytable
```

### 方法 3: 开发者安装 (从源码)

```bash
# 克隆仓库
git clone https://github.com/your-org/grace-inference
cd grace-inference/grace-inference

# 创建开发环境
conda create -n grace-dev python=3.10
conda activate grace-dev

# 安装 PyTorch 和 DGL
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install dgl-cu118 -f https://data.dgl.ai/wheels/repo.html

# 以可编辑模式安装
pip install -e .

# 或使用 pip 安装依赖后本地安装
pip install -r requirements-gpu.txt
python setup.py develop
```

---

## 3. 验证安装

### 快速验证
```bash
# 激活环境
conda activate grace-gpu  # 或 grace-cpu

# 检查 Python 版本
python --version  # 应显示 3.10.x

# 检查核心包
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import dgl; print('DGL:', dgl.__version__)"
python -c "import grace_gnn; print('GRACE:', grace_gnn.__version__)"
python -c "import ase; print('ASE:', ase.__version__)"
```

### 运行测试脚本
```bash
cd grace-inference

# 运行安装测试
python tests/test_install.py

# 预期输出示例
# ✓ Python version: 3.10.11
# ✓ PyTorch installed: 2.0.1
# ✓ DGL installed: 1.1.2
# ✓ GRACE-GNN installed: 0.1.0
# ✓ ASE installed: 3.22.1
# ✓ CUDA available: True (GPU version)
# ✓ DGL CUDA enabled: True
# All tests passed!
```

### 运行示例计算
```bash
cd examples

# 单点能计算示例
python 01_single_point.py

# 预期输出
# Loading GRACE model...
# Computing energy for structure...
# Energy: -156.78 eV
# Forces shape: (24, 3)
# Max force: 0.23 eV/Å
```

---

## 4. GPU 配置

### 检查 GPU 可用性

```bash
# 检查 CUDA 是否可用
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA version:', torch.version.cuda)"
python -c "import torch; print('GPU count:', torch.cuda.device_count())"

# 检查 DGL GPU 支持
python -c "import dgl; print('DGL CUDA enabled:', dgl.cuda.is_available())"

# 查看 GPU 信息
nvidia-smi
```

### 多 GPU 配置

```bash
# 指定使用的 GPU (使用 GPU 0)
export CUDA_VISIBLE_DEVICES=0

# 使用多个 GPU (GPU 0 和 1)
export CUDA_VISIBLE_DEVICES=0,1

# 在 Windows 中
set CUDA_VISIBLE_DEVICES=0
```

### GPU 内存优化

```python
# 在 Python 脚本中
import torch

# 启用 cudnn benchmark (加速计算)
torch.backends.cudnn.benchmark = True

# 设置 DGL 使用的 GPU
import dgl
dgl.cuda.set_device(0)  # 使用 GPU 0

# 清空 GPU 缓存
torch.cuda.empty_cache()
```

### 性能建议

| 硬件配置 | 推荐用途 | 批量大小 |
|---------|---------|---------|
| CPU only | 小分子测试、开发调试 | 1-4 |
| GPU 8GB | 中等 MOF 结构 | 4-8 |
| GPU 16GB | 大型 MOF、长时 MD | 8-16 |
| GPU 24GB+ | 高通量计算、超大结构 | 16-32 |

---

## 5. 常见问题

### Q1: DGL 安装失败
**问题**: `ERROR: Could not find a version that satisfies the requirement dgl`

**解决方案**:
```bash
# 方法 1: 从官方源安装
pip install dgl -f https://data.dgl.ai/wheels/repo.html

# 方法 2: 指定 CUDA 版本
pip install dgl-cu118 -f https://data.dgl.ai/wheels/repo.html  # CUDA 11.8
pip install dgl-cu121 -f https://data.dgl.ai/wheels/repo.html  # CUDA 12.1

# 方法 3: CPU 版本
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

### Q2: CUDA 版本不匹配
**问题**: `RuntimeError: CUDA error: no kernel image is available for execution`

**解决方案**:
```bash
# 检查系统 CUDA 版本
nvcc --version
nvidia-smi  # 查看驱动支持的最高 CUDA 版本

# 重新安装匹配的 PyTorch 和 DGL
# 例如 CUDA 11.8
pip uninstall torch dgl
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install dgl-cu118 -f https://data.dgl.ai/wheels/repo.html
```

### Q3: 导入 GRACE 失败
**问题**: `ModuleNotFoundError: No module named 'grace_gnn'`

**解决方案**:
```bash
# 检查是否正确激活环境
conda activate grace-gpu

# 重新安装 grace-gnn
pip install --upgrade grace-gnn

# 如果从源码安装
cd grace-inference
pip install -e .
```

### Q4: DGL 图构建错误
**问题**: `DGLError: Expect number of features to match number of nodes`

**解决方案**:
```python
# 确保节点特征和节点数匹配
import dgl
import torch

# 正确构建图
g = dgl.graph((src, dst))
g.ndata['feat'] = node_features  # 确保 shape = (num_nodes, feat_dim)

# 检查维度
print(f"Nodes: {g.num_nodes()}, Features: {g.ndata['feat'].shape}")
```

### Q5: 内存不足错误
**问题**: `RuntimeError: CUDA out of memory`

**解决方案**:
```python
# 1. 减少批量大小
batch_size = 4  # 改为更小的值

# 2. 使用梯度累积
torch.cuda.empty_cache()

# 3. 降低模型精度 (如果支持)
model.half()  # 使用 FP16

# 4. 使用 CPU 运行
device = 'cpu'
```

### Q6: Windows 路径问题
**问题**: 路径分隔符错误

**解决方案**:
```python
import os
from pathlib import Path

# 使用 Path 处理路径
model_path = Path("models/grace_model.pt")
structure_path = Path("structures/MOF.cif")

# 或使用 os.path.join
model_path = os.path.join("models", "grace_model.pt")
```

### Q7: DGL 版本兼容性
**问题**: `ImportError: DGL requires torch >= 1.12.0`

**解决方案**:
```bash
# 确保 PyTorch 版本满足要求
pip install torch>=2.0.0

# 或降级 DGL
pip install dgl==1.0.0  # 根据 PyTorch 版本选择兼容的 DGL
```

---

## 6. 依赖说明

### 核心依赖

| 包名 | 版本要求 | 用途 | 重要性 |
|------|---------|------|--------|
| `python` | 3.9-3.11 | 运行环境 | 必需 |
| `torch` | >=1.12.0 | 深度学习框架 | 必需 |
| `dgl` | >=1.0.0 | 图神经网络库 | 必需 |
| `grace-gnn` | >=0.1.0 | GRACE 模型 | 必需 |
| `ase` | >=3.22.0 | 原子模拟环境 | 必需 |

### 计算依赖

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| `numpy` | >=1.21.0,<2.0.0 | 数值计算 |
| `scipy` | >=1.7.0 | 科学计算 |
| `phonopy` | >=2.20.0 | 声子计算 |
| `pymatgen` | >=2023.0.0 | 材料分析 |

### 工具依赖

| 包名 | 用途 |
|------|------|
| `h5py` | HDF5 文件支持 |
| `matplotlib` | 绘图可视化 |
| `pandas` | 数据分析 |
| `tqdm` | 进度条显示 |
| `pyyaml` | YAML 配置 |
| `prettytable` | 表格格式化 |

### DGL 详细说明

**DGL (Deep Graph Library)** 是 GRACE 的核心依赖，提供高效的图神经网络计算：

- **功能**: 图构建、消息传递、异构图支持
- **优势**: GPU 加速、内存优化、批处理支持
- **版本选择**:
  - CPU: `dgl` (通用版本)
  - CUDA 11.8: `dgl-cu118`
  - CUDA 12.1: `dgl-cu121`

**安装指南**:
```bash
# CPU 版本
pip install dgl -f https://data.dgl.ai/wheels/repo.html

# GPU 版本 (CUDA 11.8)
pip install dgl-cu118 -f https://data.dgl.ai/wheels/repo.html

# GPU 版本 (CUDA 12.1)
pip install dgl-cu121 -f https://data.dgl.ai/wheels/repo.html
```

**DGL vs PyTorch Geometric**:
- DGL 提供更灵活的消息传递机制
- 更好的异构图支持
- 与 PyTorch 深度集成
- GRACE 专门针对 DGL 优化

### 可选依赖

```bash
# 进阶分析工具
pip install spglib           # 空间群分析
pip install networkx         # 图分析
pip install plotly           # 交互式可视化

# 性能分析
pip install memory_profiler  # 内存分析
pip install line_profiler    # 代码性能分析
```

### 依赖版本组合推荐

#### 稳定组合 (推荐)
```txt
python==3.10.11
torch==2.0.1
dgl-cu118==1.1.2
grace-gnn==0.1.0
ase==3.22.1
phonopy==2.20.0
```

#### 最新组合 (前沿)
```txt
python==3.11.5
torch==2.1.0
dgl-cu121==1.1.3
grace-gnn==0.2.0
ase==3.23.0
phonopy==2.21.0
```

### 依赖冲突解决

```bash
# 如果遇到依赖冲突
pip install --upgrade pip setuptools wheel

# 使用 conda 解决复杂依赖
conda install -c conda-forge numpy scipy

# 创建干净环境
conda create -n grace-clean python=3.10
conda activate grace-clean
pip install -r requirements-gpu.txt
```

---

## 📚 相关文档

- [快速入门指南](grace-inference/QUICKSTART.md)
- [API 参考手册](GRACE_inference_API_reference.md)
- [推理任务说明](GRACE_inference_tasks.md)
- [DGL 官方文档](https://docs.dgl.ai/)
- [PyTorch 文档](https://pytorch.org/docs/)

---

## 🆘 获取帮助

- **GitHub Issues**: 提交 bug 报告或功能请求
- **讨论区**: 技术交流和问题解答
- **文档**: 查看完整使用指南
- **示例**: 参考 `examples/` 目录中的代码

---

## 📝 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](grace-inference/LICENSE) 文件

---

**最后更新**: 2026年1月  
**维护团队**: GRACE Inference 开发组
