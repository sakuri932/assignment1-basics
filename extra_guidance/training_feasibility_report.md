# GPT 训练可行性研究报告

> 目标 A：在 TinyStories 上训练 TransformerLM（小模型验证）  
> 目标 B：在 OpenWebText 上训练完整大参数版本（正式训练）  
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

## 九、结论（TinyStories 阶段）

| 项目 | 结论 |
|------|------|
| **代码就绪** | ✅ 所有模块已实现，测试通过率 95.6% |
| **数据就绪** | ✅ 540M 训练 tokens 已编码为 npy，直接可用 |
| **M5 可行性** | ✅ 内存 1.1 GB（远低于 16 GB），实测约 1.2 s/step，1 epoch ≈ 22 小时 |
| **RTX 4090 可行性** | ✅ 可训练更大模型（~134M），全 epoch 约 7-15 小时 |
| **跨平台兼容** | ✅ 一份代码，自动检测 mps/cuda/cpu |
| **建议** | M5 上跑完整 1 epoch TinyStories 验证流程；随后在 RTX 4090 上跑 OWT 完整训练 |

---

---

# Part II：OpenWebText 大参数版本训练（RTX 4090）

> 本部分基于 TinyStories 阶段完成后，在 Linux + RTX 4090 24GB 服务器上  
> 训练 vocab_size=32,000 的完整规模模型。

---

## 十、OWT 数据集分析

### 10.1 数据文件

| 文件 | Token 数 | 文件大小 | 状态 |
|------|----------|----------|------|
| `data/result/owt_train.npy` | 2,727,120,452 | 5.1 GB | ✅ 就绪 |
| `data/result/owt_valid.npy` | 66,401,098 | 127 MB | ✅ 就绪 |
| `data/result/owt_vocab.json` | vocab_size = 32,000 | 791 KB | ✅ 就绪 |
| `data/result/owt_merges.txt` | 31,999 条合并规则 | 458 KB | ✅ 就绪 |

- dtype: `uint16`，max token ID = 31,999（vocab_size = 32,000 完全覆盖）
- OWT 总量约为 TinyStories 的 **5 倍**，文本复杂度远高于儿童故事

### 10.2 与 TinyStories 对比

| 维度 | TinyStories | OpenWebText |
|------|-------------|-------------|
| 训练 tokens | 540M | **2,727M（2.7B）** |
| 词表大小 | 10,000 | **32,000** |
| 文本类型 | 儿童故事（简单、重复） | 互联网文章（多样、复杂） |
| 参考 PPL 上限 | ~6-10（22M 模型） | ~30-40（130M 模型） |

OWT 语料复杂度更高，需要更大的模型才能充分拟合，因此配合 RTX 4090 扩大模型规模是合理的。

---

## 十一、大参数模型配置与参数量

保持与 TinyStories 推荐配置相同的架构风格（LLaMA 风格），仅扩大 vocab 和层数：

| 超参数 | TinyStories 小模型 | **OWT 大模型** | 说明 |
|--------|-------------------|---------------|------|
| `vocab_size` | 10,000 | **32,000** | OWT BPE 词表 |
| `context_length` | 256 | **512** | 覆盖更长依赖 |
| `d_model` | 512 | **768** | 隐藏维度 |
| `num_layers` | 4 | **12** | Transformer 块数 |
| `num_heads` | 16 | **12** | 注意力头数（d_head = 64） |
| `d_ff` | 1,344 | **2,048** | SwiGLU 内层（= 8/3 × 768，恰好整除） |
| `rope_theta` | 10,000 | **10,000** | RoPE 基频（不变） |

**参数量明细：**

```
Embedding:          32,000 × 768          =  24.58M
× 12 层：
  MHSA (Q,K,V,O):   4 × 768 × 768        =  2.36M/层
  SwiGLU (W1,W2,W3): 3 × 768 × 2,048     =  4.72M/层
  RMSNorm × 2:       2 × 768              ≈  0M/层
  小计/层                                  =  7.08M
12 层合计                                  = 84.93M
Final RMSNorm:                             ≈  0M
LM Head:            768 × 32,000          =  24.58M
──────────────────────────────────────────────────────
总参数量                                   ≈ 134.1M
```

> 与 TinyStories 22.7M 相比，参数量增大约 **5.9 倍**，主要来自更深的网络层数（×3）  
> 和更大的词表（Embedding + LM Head 从 10.2M 增至 49.2M）。

---

## 十二、显存分析（RTX 4090，batch=64，ctx=512，bf16）

采用自动混合精度（AMP）：前向 bf16，优化器状态 fp32。

| 组成 | 计算 | 大小 |
|------|------|------|
| 模型参数（bf16） | 134.1M × 2B | 0.27 GB |
| 梯度（fp32） | 134.1M × 4B | 0.54 GB |
| Adam m + v（fp32） | 134.1M × 8B | 1.07 GB |
| **模型+优化器小计** | | **1.88 GB** |
| 注意力激活值（无 Flash Attn） | 64×12×512²×2B×12层 | 4.83 GB |
| FFN 激活值（W1,W3 输出保留） | 64×512×2048×2B×2×12层 | 3.22 GB |
| 残差流激活值 | 64×512×768×2B×12层 | 0.60 GB |
| **激活值小计** | | **8.65 GB** |
| **总计** | | **≈ 10.5 GB** |

RTX 4090 有 24 GB 显存，**10.5 GB 约占 44%，安全余量充足**。

> **注意**：本代码使用自定义 SDPA（非 PyTorch 内置 SDPA），注意力矩阵 O(N²) 全部驻留显存。  
> 若后续替换为 Flash Attention，注意力激活可降至 ~0.3 GB，总显存降至 **~5 GB**，  
> 届时可大幅增大 batch_size 或 context_length。

### 不同 batch size 的显存对比

| batch_size | ctx | 激活值估算 | 总显存 | 可行性 |
|------------|-----|-----------|--------|--------|
| 32 | 512 | 4.3 GB | **6.2 GB** | ✅ 非常安全 |
| **64** | **512** | **8.7 GB** | **10.5 GB** | **✅ 推荐** |
| 96 | 512 | 13.0 GB | 14.9 GB | ✅ 可以尝试 |
| 128 | 512 | 17.3 GB | 19.2 GB | ⚠️ 偏紧，有 OOM 风险 |

---

## 十三、速度与训练时长估算

### 13.1 每步 FLOPs

```
FLOPs/step = 6 × N_params × batch_size × context_length
           = 6 × 134.1M × 64 × 512
           = 26.36 TFLOPs/step
```

### 13.2 RTX 4090 速度估算

| 精度模式 | 理论算力 | 有效利用率 | 估算步速 | 1 epoch 耗时 |
|----------|---------|-----------|---------|-------------|
| fp32 | 82.6 TFLOPS | ~50% | ~1.6 steps/sec | **~14.8 小时** |
| **bf16（AMP）** | **165.2 TFLOPS** | **~50%** | **~3.1 steps/sec** | **~7.4 小时** |
| bf16 + torch.compile | 165.2 TFLOPS | ~65% | ~4.1 steps/sec | **~5.7 小时** |

**1 epoch 步数**（batch=64，ctx=512）：  
2,727,120,452 ÷ (64 × 512) = **83,225 步**

> 推荐方案：bf16 + torch.compile，1 epoch 约 **5-8 小时**。  
> 从 Chinchilla 最优算力视角，134M 参数模型约需 2.7B tokens（恰好是 OWT 1 epoch），  
> 因此训练 1 epoch 本身即接近最优计算分配。

### 13.3 关键 Milestone 时间表

| 步数 | 处理 tokens | 估算耗时（bf16） | 建议用途 |
|------|-------------|----------------|---------|
| 2,000 | 65M | ~11 分钟 | 确认 loss 正常下降 |
| 10,000 | 328M | ~54 分钟 | 观察训练曲线，调整 LR |
| 30,000 | 983M | ~2.7 小时 | 中期检查点 |
| 83,225 | 2,727M（1 epoch） | **~7.5 小时** | 完整训练 |

---

## 十四、推荐超参数配置（OWT 版）

```python
# ─── 模型结构 ───────────────────────────────────────────────────────────────
vocab_size      = 32_000
context_length  = 512
d_model         = 768
num_layers      = 12
num_heads       = 12
d_ff            = 2_048
rope_theta      = 10_000.0

# ─── 训练 ───────────────────────────────────────────────────────────────────
device          = "cuda"
batch_size      = 64
max_iters       = 83_225       # 1 epoch on OWT (2.727B tokens / 64 / 512)
warmup_iters    = 1_500        # ~2% of total steps
max_lr          = 3e-4         # 更大模型用更小 LR（vs TinyStories 的 1e-3）
min_lr          = 3e-5
weight_decay    = 0.1          # 更大模型适当增大正则
betas           = (0.9, 0.95)  # GPT-2/3 风格（β₂ 从 0.999 降至 0.95，减小噪声）
eps             = 1e-8
max_grad_norm   = 1.0

# ─── 评估 & 检查点 ───────────────────────────────────────────────────────────
eval_interval        = 500
checkpoint_interval  = 5_000
```

**LR 选择依据**：
- Chinchilla 建议 ~100M 参数级别模型 max_lr 约 2e-4 ~ 6e-4
- GPT-2 small（117M）使用 2.5e-4；GPT-3 6.7B 使用 1.2e-4
- 保守取 3e-4，若训练初期 loss 下降过慢可尝试提高至 6e-4

---

## 十五、需要的代码改动

### 15.1 新增训练启动脚本（必须）

新建 `run/train_owt.py`，参照 `run/train_tinystories.py`，修改以下内容：
- `DATA_DIR` 指向 OWT 数据文件
- `CONFIG` 使用上述 OWT 超参数
- 数据加载：`np.load("owt_train.npy")` / `np.load("owt_valid.npy")`

### 15.2 启用自动混合精度 bf16（强烈推荐）

在 `cs336_basics/training.py` 的训练循环中，将前向+损失计算包裹在 autocast 中：

```python
# 在 training.py 中的主循环内
with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
    logits = model(x)
    loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), y.view(-1))
```

> 只需 3 行改动，fp32 的优化器状态自动保留，无需其他修改。  
> 不加 autocast 也能训练（全 fp32），但速度慢约 2 倍，显存多 1.5 倍。

### 15.3 启用 torch.compile（可选，推荐）

```python
# 在模型初始化后、训练开始前（train_owt.py 中）
if device == "cuda":
    print("Compiling model with torch.compile...")
    model = torch.compile(model)
```

首次编译约需 2-5 分钟，之后步速提升 20-40%，长时间训练完全值得。

### 15.4 评估批次数量调整（可选）

当前 `training.py` 固定评估 10 个 batch，对 OWT 的 66M val tokens 来说样本量偏小。  
可将评估 batch 数提高到 50-100，使 val_loss 估计更稳定：

```python
# training.py line 434
for _ in range(50):   # 原来是 10
```

---

## 十六、预期训练结果（OWT）

OWT 是真实互联网文本，语言分布复杂，困惑度参考值：

| 模型大小 | 参考 PPL（OWT 验证集） | 备注 |
|----------|----------------------|------|
| ~22M 参数 | ~55-70 | 欠拟合 |
| ~134M 参数（本配置，1 epoch） | **~28-38** | GPT-2 small 水平 |
| ~134M 参数（多 epoch） | ~25-32 | 需要 2-3 epoch |
| ~350M 参数 | ~20-28 | GPT-2 medium 水平 |

> 1 epoch 达到 PPL ~30-38 是合理预期。若目标 PPL < 30，可考虑训练 2-3 epoch  
>（总时长 15-25 小时，需要服务器持续运行）。

---

## 十七、综合结论

| 项目 | TinyStories 阶段 | OWT 阶段 |
|------|-----------------|---------|
| **数据就绪** | ✅ 540M tokens | ✅ 2,727M tokens |
| **模型参数** | 22.7M | **134.1M** |
| **运行平台** | MacBook Air M5 | **Linux + RTX 4090** |
| **精度** | fp32（MPS 不支持 bf16） | **bf16 AMP** |
| **显存占用** | 1.1 GB（统一内存） | **~10.5 GB / 24 GB** |
| **速度** | 实测 ~1.2 s/step | 估算 ~3-4 steps/sec（bf16） |
| **1 epoch 耗时** | ~22 小时（实测） | **~7-8 小时（bf16）** |
| **代码改动** | 无 | autocast（3行）+ 新启动脚本 |
| **可行性** | ✅ 已验证完成 | ✅ 显存充足，时间合理 |

**下一步行动**：
1. 在服务器上确认 PyTorch CUDA 环境（`torch.cuda.is_available()`）
2. 将 OWT npy 文件传输到服务器
3. 新建 `run/train_owt.py`，加入 autocast 和 torch.compile
4. 先跑 2,000 步冒烟测试，确认 loss 正常下降后开启完整训练
