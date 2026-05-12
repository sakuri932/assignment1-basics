# GPT 训练可行性研究报告

> 目标：在 TinyStories 上训练 TransformerLM  
> 平台 A：MacBook Air M5 16GB（本地测试）  
> 平台 B：Linux + RTX 4090 24GB（正式训练）

---

## 一、数据就绪情况

| 文件 | Token 数 | 文件大小 | 状态 |
|------|----------|----------|------|
| `data/result/tinystories_train.npy` | 540,796,778 | 1.0 GB | ✅ 就绪 |
| `data/result/tinystories_valid.npy` | 5,461,210 | 10 MB | ✅ 就绪 |

- dtype: `uint16`，max token ID = 9999（vocab_size = 10,000 完全覆盖）
- 无需任何预处理，直接用 `np.load()` 加载即可

---

## 二、代码就绪情况

| 模块 | 文件 | 状态 |
|------|------|------|
| 模型架构 | `cs336_basics/model.py` | ✅ TransformerLM 完整实现 |
| 优化器 | `cs336_basics/optimizer.py` | ✅ AdamW + 余弦学习率调度 + 梯度裁剪 |
| 训练循环 | `cs336_basics/training.py` | ✅ `train()` 函数含评估、检查点、日志 |
| 分词器 | `cs336_basics/tokenizer.py` | ✅ 已训练完毕（result/ 目录） |

测试通过率：43/45（95.6%），2 个因系统资源跳过，功能完整。

**LM head 未绑定 Embedding 权重**（代码注释明确说明），参数量相应偏大。

---

## 三、模型参数量计算

### 推荐小模型（TinyStories）

| 超参数 | 值 | 说明 |
|--------|----|------|
| `vocab_size` | 10,000 | BPE 词表 |
| `context_length` | 256 | 上下文窗口 |
| `d_model` | 512 | 隐藏维度 |
| `num_layers` | 4 | Transformer 块数 |
| `num_heads` | 16 | 注意力头数（d_head = 32） |
| `d_ff` | 1,344 | SwiGLU 内层（≈ 8/3 × 512，取 64 倍数） |
| `rope_theta` | 10,000 | RoPE 基频 |

**参数量明细：**

```
Embedding:         10,000 × 512         =  5.12M
× 4 层：
  MHSA (Q,K,V,O):  4 × 512 × 512       =  1.05M/层
  SwiGLU (W1,W2,W3): 3 × 512 × 1,344   =  2.06M/层
  RMSNorm × 2:      2 × 512             ≈  0M/层
  小计/层                                 = 3.11M
4 层合计                                 = 12.44M
Final RMSNorm:                           ≈  0M
LM Head:          512 × 10,000          =  5.12M
─────────────────────────────────────────────────
总参数量                                 ≈ 22.7M
```

---

## 四、平台 A：MacBook Air M5 16GB

### 4.1 内存分析

| 组成 | 计算 | 大小 |
|------|------|------|
| 模型参数（float32） | 22.7M × 4B | 90.8 MB |
| 梯度 | 22.7M × 4B | 90.8 MB |
| Adam m、v 两个状态 | 22.7M × 4B × 2 | 181.6 MB |
| **模型+优化器小计** | | **363 MB** |
| 注意力激活值（batch=32，ctx=256，4层） | 32×16×256²×4B×4层 | ~537 MB |
| 其他隐藏状态 | | ~200 MB |
| **激活值小计** | | ~737 MB |
| **总计** | | **≈ 1.1 GB** |

16GB 统一内存中 GPU 可用部分通常为 10-12GB，**1.1 GB 远低于上限，完全可行**。

### 4.2 速度估算

- M5 GPU（约 16 核，估算 ~8 TFLOPS FP32）
- MPS 实测利用率约 30-50%，有效算力约 **2.5-4 TFLOPS**
- 每步 FLOPs = 6 × 22.7M × 32 × 256 ≈ **1.12 TFLOPS**
- 估算速度：**2-4 步/秒**

| 训练步数 | 处理 token 数 | 估算耗时 |
|----------|---------------|----------|
| 5,000 步 | 41M tokens | 20-42 分钟 |
| 10,000 步 | 82M tokens | 40-83 分钟 |
| 20,000 步 | 164M tokens | 1.4-2.8 小时 |
| 65,918 步（1 epoch） | 540M tokens | 4.6-9.2 小时 |

**建议初始测试**：10,000 步（约 1-1.5 小时），足以观察 loss 下降曲线。

### 4.3 注意事项

- **MPS 设备**：PyTorch 2.x 已支持，用 `torch.backends.mps.is_available()` 检测
- **bf16**：MPS 对 bf16 支持不完整，保持 **float32**
- **flash attention**：MPS 不支持，但本代码用的是自定义 `scaled_dot_product_attention`，无影响

---

## 五、平台 B：Linux + RTX 4090 24GB

### 5.1 内存分析

相同小模型（22.7M params）内存消耗约 1.1 GB，对 24 GB 显存毫无压力。  
**可以显著增大模型和批次**：

| 配置 | 参数量 | 显存估算 |
|------|--------|----------|
| 小模型 batch=128, ctx=512 | 22.7M | ~4 GB |
| 中模型 d_model=768, 12层, batch=64, ctx=512 | ~85M | ~10 GB |
| 中模型 batch=128 | ~85M | ~18 GB |

RTX 4090 24GB 显存可以跑到 **85M 参数 + batch=128** 的规模。

### 5.2 速度估算（小模型，与 M5 相同配置对比）

- RTX 4090：82.6 TFLOPS FP32（bf16 可达 165 TFLOPS）
- CUDA 利用率约 40-60%，有效算力约 **33-50 TFLOPS**
- batch=64, ctx=256 时每步 FLOPs ≈ 2.24 TFLOPS
- 估算速度：**15-22 步/秒**（vs M5 的 2-4 步/秒，约快 6-8 倍）

| 训练步数 | 处理 token 数 | 估算耗时 |
|----------|---------------|----------|
| 10,000 步 | 82M tokens（batch=64） | **7-11 分钟** |
| 65,918 步（1 epoch，batch=64） | 540M tokens | **50-75 分钟** |
| 3 epoch（batch=64） | 1.62B tokens | **2.5-3.5 小时** |

### 5.3 可用加速特性

| 特性 | M5 | RTX 4090 |
|------|----|----------|
| bf16 训练 | ❌ | ✅（速度约翻倍） |
| torch.compile | 部分支持 | ✅（可再提速 20-50%） |
| Flash Attention | ❌ | ✅（仅 SDPA 接口） |
| 多 GPU | ❌ | 单卡即可 |

---

## 六、推荐超参数配置

### 平台 A（M5，初步测试）

```python
# 模型
vocab_size      = 10_000
context_length  = 256
d_model         = 512
num_layers      = 4
num_heads       = 16
d_ff            = 1344
rope_theta      = 10_000

# 训练
device          = "mps"
batch_size      = 32
max_iters       = 10_000
warmup_iters    = 200
max_lr          = 1e-3
min_lr          = 1e-4
weight_decay    = 0.01
max_grad_norm   = 1.0
eval_interval   = 500
checkpoint_interval = 2_000
```

### 平台 B（RTX 4090，正式训练）

```python
# 模型（可以用更大的）
vocab_size      = 10_000
context_length  = 512
d_model         = 768
num_layers      = 12
num_heads       = 12
d_ff            = 2_048

# 训练
device          = "cuda"
batch_size      = 128
max_iters       = 20_000          # ≈ 3 epoch
warmup_iters    = 400
max_lr          = 6e-4            # 更大的模型用更小的 lr
min_lr          = 6e-5
weight_decay    = 0.1
max_grad_norm   = 1.0
eval_interval   = 500
checkpoint_interval = 2_000
```

---

## 七、跨平台代码兼容性

现有代码几乎不需要修改，只需在启动脚本中自动检测设备：

```python
import torch

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

**唯一需要注意**：若将来在 RTX 4090 上启用 bf16，要在模型初始化时传 `dtype=torch.bfloat16`，并确保 `cross_entropy_loss` 内部 `log_sum_exp` 在 float32 中计算（当前实现已处理数值稳定性，问题不大）。

---

## 八、预期训练结果

TinyStories 是儿童故事语料，结构简单，小模型可以达到较低困惑度：

| 模型大小 | 参考 PPL（验证集） |
|----------|-------------------|
| ~1M 参数 | ~20-30 |
| ~10M 参数 | ~8-15 |
| ~22M 参数（本配置） | **~6-10** |
| ~85M 参数（RTX配置） | **~4-7** |

---

## 九、结论

| 项目 | 结论 |
|------|------|
| **代码就绪** | ✅ 所有模块已实现，测试通过率 95.6% |
| **数据就绪** | ✅ 540M 训练 tokens 已编码为 npy，直接可用 |
| **M5 可行性** | ✅ 内存 1.1 GB（远低于 16 GB），速度 2-4 步/秒，10K 步约 1 小时 |
| **RTX 4090 可行性** | ✅ 可训练更大模型（~85M），全 epoch 约 1 小时 |
| **跨平台兼容** | ✅ 一份代码，自动检测 mps/cuda/cpu |
| **建议** | M5 上跑 5K~10K 步验证流程正确；随后在 RTX 4090 上跑完整训练 |
