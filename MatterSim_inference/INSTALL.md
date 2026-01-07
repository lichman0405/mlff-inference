# MatterSim Inference - 安装指南

> **MatterSim**: MOFSimBench 排名 **#3** 的通用机器学习力场  
> **开发团队**: Microsoft Research - Yang et al. 2024  
> **核心依赖**: [microsoft/mattersim](https://github.com/microsoft/mattersim)

---

## 目录

1. [系统要求](#系统要求)
2. [快速安装](#快速安装)
3. [从源码安装](#从源码安装)
4. [模型下载](#模型下载)
5. [验证安装](#验证安装)
6. [常见问题](#常见问题)

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

- ✅ **Linux** (Ubuntu 20.04+, CentOS 8+)
- ✅ **macOS** (12.0+, Intel / Apple Silicon)
- ✅ **Windows** (10/11, with WSL2 推荐)

---

## 快速安装

### 方法 1: pip 安装 (推荐)

```bash
# 创建 Conda 环境 (推荐)
conda create -n mattersim python=3.10 -y
conda activate mattersim

# 安装 PyTorch (CUDA 版本)
# CUDA 11.8
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# 安装 MatterSim
pip install mattersim

# 安装 MatterSim Inference
pip install mattersim-inference

# 安装完成！验证
mattersim-infer --version
```

### 方法 2: 从源码安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/mattersim-inference.git
cd mattersim-inference

# 创建环境
conda create -n mattersim python=3.10 -y
conda activate mattersim

# 安装 PyTorch
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 安装 MatterSim
pip install mattersim

# 安装本包 (开发模式)
pip install -e .
```

---

## 模型下载

MatterSim 模型会在首次使用时自动下载。也可以手动下载：

```python
from mattersim.forcefield import MatterSimCalculator

# 自动下载并加载模型
calc = MatterSimCalculator(device="cuda")
```

### 可用模型

| 模型名称 | 参数量 | 说明 |
|----------|--------|------|
| **MatterSim-v1-1M** | 1M | 轻量版，速度快 |
| **MatterSim-v1-5M** | 5M | 标准版，推荐 |

---

## 验证安装

### 快速验证

```python
from mattersim_inference import MatterSimInference

# 初始化
calc = MatterSimInference(device="auto")
print(f"Device: {calc.device}")
print("✓ Installation successful!")
```

### 完整验证

```bash
# 运行测试
cd mattersim-inference
pytest tests/ -v
```

---

## 常见问题

### 1. mattersim 安装失败

```bash
# 确保 PyTorch 已安装
pip install torch==2.1.0

# 然后安装 mattersim
pip install mattersim
```

### 2. CUDA 不可用

```python
import torch
print(torch.cuda.is_available())  # 应为 True
print(torch.version.cuda)
```

### 3. 内存不足

- 使用较小的超胞
- 使用 CPU (内存更大)
- 减少 MD 步数

### 4. CLI 命令未找到

```bash
# 确保安装了 CLI
pip install -e .

# 或直接运行
python -m mattersim_inference.cli --version
```

---

## 卸载

```bash
pip uninstall mattersim-inference mattersim
conda deactivate
conda env remove -n mattersim
```

---

## 下一步

- 查看 [QUICKSTART.md](mattersim-inference/QUICKSTART.md) 快速上手
- 阅读 [MatterSim_inference_tasks.md](MatterSim_inference_tasks.md) 了解任务详情
- 运行 [examples/](mattersim-inference/examples/) 中的示例
