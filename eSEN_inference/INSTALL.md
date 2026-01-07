# eSEN Inference - 安装指南

> **eSEN (Smooth & Expressive Equivariant Networks)**: MOFSimBench 排名 **#1** 的通用机器学习力场  
> **开发团队**: Meta FAIR - Fu et al. 2025  
> **核心依赖**: [FAIR-Chem/fairchem](https://github.com/FAIR-Chem/fairchem)

---

## 目录

1. [系统要求](#系统要求)
2. [快速安装](#快速安装)
3. [从源码安装](#从源码安装)
4. [模型下载](#模型下载)
5. [验证安装](#验证安装)
6. [常见问题](#常见问题)
7. [卸载](#卸载)

---

## 系统要求

### 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| **CPU** | 4 核 (x86_64 / ARM64) | 8+ 核 |
| **内存** | 8 GB RAM | 16+ GB RAM |
| **GPU** | 可选 (NVIDIA with CUDA) | NVIDIA GPU, 8+ GB VRAM |
| **存储** | 5 GB 可用空间 | 10+ GB SSD |

### 软件要求

| 软件 | 版本 | 说明 |
|------|------|------|
| **Python** | 3.9, 3.10, 3.11 | **推荐 3.10** |
| **CUDA** | 11.8 / 12.1+ | GPU 加速 (可选) |
| **GCC/Clang** | 7+ / 10+ | 编译扩展模块 |

### 操作系统

- ✅ **Linux** (Ubuntu 20.04+, CentOS 8+, etc.)
- ✅ **macOS** (12.0+, Intel / Apple Silicon)
- ✅ **Windows** (10/11, with WSL2 推荐)

---

## 快速安装

### 方法 1: pip 安装 (推荐)

```bash
# 创建 Conda 环境 (推荐)
conda create -n esen python=3.10 -y
conda activate esen

# 安装 PyTorch (CUDA 版本)
# CUDA 11.8
conda install pytorch==2.1.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# CUDA 12.1
conda install pytorch==2.1.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# CPU only
conda install pytorch==2.1.0 torchvision torchaudio cpuonly -c pytorch

# 安装 eSEN Inference
pip install esen-inference

# 安装完成！验证
esen-infer --version
```

### 方法 2: 一键安装脚本

```bash
# 下载安装脚本
wget https://github.com/yourusername/esen-inference/raw/main/scripts/install.sh
chmod +x install.sh

# 运行安装 (自动检测 CUDA)
./install.sh

# 或手动指定选项
./install.sh --cuda 11.8 --python 3.10
```

---

## 从源码安装

### Step 1: 克隆仓库

```bash
git clone https://github.com/yourusername/esen-inference.git
cd esen-inference
```

### Step 2: 创建环境

```bash
# 使用 Conda (推荐)
conda env create -f environment.yml
conda activate esen

# 或使用 venv
python3.10 -m venv esen_env
source esen_env/bin/activate  # Linux/macOS
# esen_env\Scripts\activate  # Windows
```

### Step 3: 安装依赖

```bash
# 核心依赖
pip install -r requirements.txt

# 开发依赖 (可选)
pip install -r requirements-dev.txt
```

### Step 4: 安装 eSEN Inference

```bash
# 开发模式安装 (推荐)
pip install -e .

# 或标准安装
pip install .
```

### Step 5: 验证安装

```bash
python -c "from esen_inference import ESENInference; print('✓ Installation successful!')"
esen-infer --version
```

---

## 模型下载

eSEN Inference 会在首次使用时 **自动下载** 预训练模型检查点。

### 自动下载

```python
from esen_inference import ESENInference

# 首次使用会自动下载 eSEN-30M-OAM (~500 MB)
esen = ESENInference(model_name='esen-30m-oam', device='cuda')
# Downloading checkpoint: esen-30m-oam.pt ... Done!
```

### 手动下载 (可选)

```bash
# 下载 eSEN-30M-OAM 模型
mkdir -p ~/.cache/esen_inference/checkpoints
cd ~/.cache/esen_inference/checkpoints

# 从 GitHub Releases 下载
wget https://github.com/FAIR-Chem/fairchem/releases/download/v1.0/esen-30m-oam.pt

# 从 Hugging Face 下载 (备用)
wget https://huggingface.co/fair-chem/esen-30m-oam/resolve/main/checkpoint.pt -O esen-30m-oam.pt
```

### 检查点位置

| OS | 默认路径 |
|----|----------|
| **Linux** | `~/.cache/esen_inference/checkpoints/` |
| **macOS** | `~/Library/Caches/esen_inference/checkpoints/` |
| **Windows** | `%LOCALAPPDATA%\esen_inference\checkpoints\` |

### 可用模型

| 模型名称 | 文件名 | 大小 | 训练数据 |
|----------|--------|------|----------|
| `esen-30m-oam` | `esen-30m-oam.pt` | ~500 MB | OMat24 + MPtraj + sAlex |
| `esen-30m-mp` | `esen-30m-mp.pt` | ~500 MB | MPtraj only |

---

## 验证安装

### 测试 1: 导入库

```python
python3 << EOF
from esen_inference import ESENInference
from esen_inference.utils import device
from esen_inference.tasks import StaticTask, OptimizationTask

print("✓ All imports successful!")
EOF
```

### 测试 2: GPU 可用性

```python
python3 << EOF
import torch
from esen_inference.utils.device import get_device_info

info = get_device_info()
print(f"Device type: {info['device_type']}")

if info['device_type'] == 'cuda':
    print(f"✓ GPU available: {info['device_name']}")
    print(f"  CUDA version: {info['cuda_version']}")
    print(f"  GPU memory: {info['gpu_memory_total']} MB")
else:
    print("! GPU not available, using CPU")
EOF
```

### 测试 3: 单点计算

```bash
# 下载测试结构
wget https://github.com/yourusername/esen-inference/raw/main/examples/data/MOF-5.cif

# 运行单点计算
esen-infer single-point MOF-5.cif --output result.json

# 查看结果
cat result.json
```

### 测试 4: 完整测试套件

```bash
# 运行所有测试
pytest tests/ -v

# 运行快速测试 (仅基础功能)
pytest tests/test_install.py -v
```

---

## 常见问题

### Q1: CUDA 版本不匹配

**问题**:
```
RuntimeError: CUDA version mismatch: PyTorch compiled with CUDA 11.8 but system has CUDA 12.1
```

**解决**:
```bash
# 检查系统 CUDA 版本
nvcc --version

# 重新安装匹配的 PyTorch
# 例如系统 CUDA 12.1:
pip uninstall torch
conda install pytorch==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Q2: fairchem 安装失败

**问题**:
```
ERROR: Could not build wheels for fairchem
```

**解决**:
```bash
# 安装编译依赖
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# CentOS/RHEL
sudo yum install gcc gcc-c++ python3-devel

# macOS
xcode-select --install

# 重新安装
pip install --no-cache-dir fairchem
```

### Q3: Phonopy 导入错误

**问题**:
```python
ImportError: No module named 'phonopy._phonopy'
```

**解决**:
```bash
# 重新安装 Phonopy
pip uninstall phonopy -y
pip install phonopy --no-binary phonopy

# 或使用 Conda (推荐)
conda install -c conda-forge phonopy
```

### Q4: 内存不足 (GPU)

**问题**:
```
RuntimeError: CUDA out of memory
```

**解决**:
```python
# 方法 1: 降低精度
esen = ESENInference(model_name='esen-30m-oam', device='cuda', precision='float32')

# 方法 2: 使用 CPU
esen = ESENInference(model_name='esen-30m-oam', device='cpu')

# 方法 3: 清理 GPU 缓存
import torch
torch.cuda.empty_cache()
```

### Q5: macOS Apple Silicon 安装

**问题**: M1/M2 Mac 安装错误

**解决**:
```bash
# 使用 Conda (推荐)
conda create -n esen python=3.10 -y
conda activate esen

# 安装 PyTorch (Apple Silicon 优化)
conda install pytorch::pytorch torchvision torchaudio -c pytorch

# 安装 eSEN Inference
pip install esen-inference

# 使用 MPS 加速 (Metal Performance Shaders)
# Python 代码:
esen = ESENInference(model_name='esen-30m-oam', device='mps')
```

### Q6: Windows 安装建议

**推荐**: 使用 WSL2 (Windows Subsystem for Linux)

```powershell
# PowerShell (管理员)
wsl --install -d Ubuntu-22.04
wsl

# 在 WSL2 中按 Linux 安装流程操作
```

**或原生 Windows**:
```powershell
# 安装 Miniconda
# https://docs.conda.io/en/latest/miniconda.html

# Conda Prompt
conda create -n esen python=3.10 -y
conda activate esen
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install esen-inference
```

### Q7: 模型下载慢/失败

**解决**:
```bash
# 方法 1: 使用镜像
pip install esen-inference -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方法 2: 手动下载模型 (见 "模型下载" 章节)

# 方法 3: 使用代理
export http_proxy=http://your_proxy:port
export https_proxy=http://your_proxy:port
pip install esen-inference
```

### Q8: ASE 版本冲突

**问题**:
```
ERROR: ase 3.23.0 conflicts with ase>=3.22.0,<3.23.0
```

**解决**:
```bash
# 卸载所有相关包
pip uninstall ase esen-inference fairchem -y

# 重新安装 (指定 ASE 版本)
pip install 'ase>=3.22.0,<3.24.0'
pip install esen-inference
```

---

## 依赖列表

### 核心依赖 (requirements.txt)

```txt
# PyTorch ecosystem
torch>=2.0.0,<2.2.0
torchvision>=0.15.0
torchaudio>=2.0.0

# FAIR-Chem (eSEN 核心)
fairchem>=1.0.0

# ASE (Atomic Simulation Environment)
ase>=3.22.0,<3.24.0

# Phonopy (声子计算)
phonopy>=2.20.0
h5py>=3.8.0

# Scientific computing
numpy>=1.24.0,<2.0.0
scipy>=1.10.0
matplotlib>=3.7.0

# I/O and utilities
pymatgen>=2023.9.0
pyyaml>=6.0

# Optional: accelerators
# nvidia-ml-py3>=7.352.0  # GPU 监控
```

### 开发依赖 (requirements-dev.txt)

```txt
# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-xdist>=3.3.0

# Code quality
black>=23.7.0
isort>=5.12.0
flake8>=6.1.0
mypy>=1.5.0

# Documentation
sphinx>=7.1.0
sphinx-rtd-theme>=1.3.0

# Jupyter
jupyterlab>=4.0.0
ipywidgets>=8.1.0
```

---

## 环境文件

### environment.yml (Conda)

```yaml
name: esen
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch::pytorch=2.1.0
  - pytorch::torchvision=0.16.0
  - pytorch::torchaudio=2.1.0
  - pytorch::pytorch-cuda=11.8
  - conda-forge::ase=3.22.1
  - conda-forge::phonopy=2.20.0
  - conda-forge::h5py=3.9.0
  - numpy=1.24.3
  - scipy=1.11.2
  - matplotlib=3.7.2
  - pip
  - pip:
      - fairchem>=1.0.0
      - pymatgen>=2023.9.0
      - esen-inference
```

创建环境:
```bash
conda env create -f environment.yml
conda activate esen
```

---

## Docker 安装 (高级)

### Dockerfile

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 安装 Python 和依赖
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget \
    && rm -rf /var/lib/apt/lists/*

# 安装 PyTorch
RUN pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 安装 eSEN Inference
RUN pip3 install esen-inference

# 设置工作目录
WORKDIR /workspace

# 默认命令
CMD ["esen-infer", "--help"]
```

### 构建和运行

```bash
# 构建镜像
docker build -t esen-inference:latest .

# 运行容器 (GPU)
docker run --gpus all -it esen-inference:latest bash

# 挂载数据目录
docker run --gpus all -v $(pwd):/workspace -it esen-inference:latest \
    esen-infer single-point /workspace/MOF-5.cif
```

---

## 卸载

### 完全卸载

```bash
# 卸载 eSEN Inference
pip uninstall esen-inference -y

# 卸载依赖 (可选)
pip uninstall fairchem ase phonopy -y

# 删除模型缓存
rm -rf ~/.cache/esen_inference

# 删除 Conda 环境 (如果使用)
conda deactivate
conda env remove -n esen
```

### 保留配置

```bash
# 仅卸载软件包，保留缓存和配置
pip uninstall esen-inference -y
```

---

## 更新

### 更新到最新版本

```bash
# PyPI 更新
pip install --upgrade esen-inference

# 从源码更新 (开发版)
cd esen-inference
git pull origin main
pip install -e . --force-reinstall
```

### 查看版本

```bash
esen-infer --version
python -c "import esen_inference; print(esen_inference.__version__)"
```

---

## 安装验证清单

运行以下命令验证安装完整性:

```bash
# 1. Python 版本
python --version  # 应为 3.9, 3.10, 或 3.11

# 2. PyTorch 安装
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 3. CUDA 可用性 (如果有 GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 4. fairchem 导入
python -c "from fairchem.core import OCPCalculator; print('✓ fairchem OK')"

# 5. eSEN Inference 导入
python -c "from esen_inference import ESENInference; print('✓ eSEN Inference OK')"

# 6. CLI 可用性
esen-infer --version

# 7. 快速测试
pytest tests/test_install.py -v  # 如果安装了测试套件
```

**全部通过即安装成功！** ✅

---

## 技术支持

### 报告问题

- **GitHub Issues**: https://github.com/yourusername/esen-inference/issues
- **讨论区**: https://github.com/yourusername/esen-inference/discussions

### 常用资源

- **eSEN 论文**: [arXiv:2502.12147](https://arxiv.org/abs/2502.12147)
- **FAIR-Chem 文档**: https://fair-chem.github.io/
- **ASE 文档**: https://wiki.fysik.dtu.dk/ase/
- **Phonopy 文档**: https://phonopy.github.io/phonopy/

---

**文档版本**: v1.0  
**更新日期**: 2026-01-07  
**支持的 Python 版本**: 3.9, 3.10, 3.11  
**推荐配置**: Python 3.10 + CUDA 11.8 + PyTorch 2.1.0
