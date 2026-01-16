# Deep Learning and LLM Lab

## 环境搭建

首先在 M 芯片 Mac 上安装了 Miniconda，随后创建 conda 环境

```bash
conda create -n dl python=3.10 -y

conda activate dl

# 安装 PyTorch
conda install pytorch -c pytorch -y

# 安装其他常用库
conda install jupyter matplotlib pandas scikit-learn -y

# 安装 Transformers、Datasets、Tokenizers、Accelerate、Hugging Face Hub
pip install transformers datasets tokenizers accelerate huggingface-hub
```

测试 PyTorch 是否支持 MPS
```python
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"是否支持 MPS: {torch.backends.mps.is_available()}")
print(f"MPS 是否已启用: {torch.backends.mps.is_built()}")

# 测试一个简单的计算
if torch.backends.mps.is_available():
    x = torch.ones(3, 3, device="mps")
    print("成功在 GPU (MPS) 上创建 Tensor!")
else:
    print("未能使用 GPU。")
```

## 常用命令

启动编辑器
```bash
jupyter notebook
```

删除 Hugging Face 缓存
```bash
huggingface-cli delete-cache
```

删除本地模型
```bash
rm -rf ~/.cache/huggingface/hub
```

删除本地数据集
```bash
rm -rf ~/.cache/huggingface/datasets
```