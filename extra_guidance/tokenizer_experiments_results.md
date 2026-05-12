# Tokenizer Experiments 实验结果

> Section 2.7 — 运行时间：2026-05-05 15:00 ~ 2026-05-06 01:07（服务器：i9-14900K / 128GB RAM）

---

## 环境

| 参数 | 值 |
|------|----|
| 服务器 CPU | Intel i9-14900K（32线程） |
| 内存 | 128 GB |
| GPU | NVIDIA RTX 4090（BPE 训练不使用 GPU） |
| Python | 3.12.3 |
| 运行路径 | `~/workspace/assignment1/tokenizer/` |

---

## 步骤 0：训练分词器

| 分词器 | 语料 | vocab_size | 训练耗时 |
|--------|------|-----------|----------|
| TinyStories | TinyStoriesV2-GPT4-train.txt（2.1 GB） | 10,000 | 已预先训练，直接加载 |
| OpenWebText | owt_train.txt（12 GB） | 32,000 | **15,201.6s（≈ 4h13m）** |

> 注：OWT 分词器使用朴素 BPE 实现（`max()` 全扫描，无 heap 优化），32,000 次合并 × 全量 pair_counts 扫描。BPE 合并阶段为单线程，峰值内存约 8.5 GB。

---

## (a) 压缩比（各分词器处理自己的数据集）

样本：各取前 10 篇文档。

| 分词器 | 数据集 | 原始字节 | Token 数 | 压缩比（字节/token） |
|--------|--------|----------|----------|----------------------|
| TinyStories（10K vocab） | TinyStories | 7,412 | 1,786 | **4.1501** |
| OpenWebText（32K vocab） | OpenWebText | 31,487 | 6,712 | **4.6912** |

OWT 分词器压缩比更高，因为：
1. vocab_size 更大（32K vs 10K），能学到更长的常见子词
2. 训练语料更多样，覆盖更广的英语文本模式

---

## (b) 跨分词器：TinyStories 分词器处理 OWT 样本

| 分词器 | 数据集 | 原始字节 | Token 数 | 压缩比（字节/token） |
|--------|--------|----------|----------|----------------------|
| TinyStories | OWT | 31,487 | 9,873 | **3.1892** |
| OWT（自己的） | OWT | 31,487 | 6,712 | 4.6912 |

TinyStories 分词器在 OWT 上的压缩率比 OWT 自身分词器低 **32%**，说明领域不匹配会显著降低 BPE 压缩效率。

### 定性示例（第 1 篇文档，前 200 字节）

```
TinyStories 分词器 → 66 tokens
OWT 分词器         → 49 tokens

前 10 个 token（两者相同）：
['What', ' wouldn', "'t", ' you', ' do', ' to', ' save', ' someone', ' you', ' love']
```

前 10 个 token 恰好一致，因为这段文本都是常见英语单词，两个分词器都已学习。差异出现在较长、较专业的词汇上。

---

## (c) 吞吐量估算

测速文件：`TinyStoriesV2-GPT4-valid.txt`（22.5 MB），只读前 10.5 MB。

| 指标 | 值 |
|------|----|
| 输入大小 | 10.5 MB |
| Token 数 | 2,547,736 |
| 耗时 | 12.33s |
| **吞吐量** | **0.85 MB/s（850,985 字节/s）** |
| 对 Pile（825 GB）的估计时间 | 969,464s ≈ **269.3 小时** |

> 纯 Python 实现速度有限。生产级分词器（如 tiktoken）使用 Rust/C++ 实现，速度可达 100 MB/s 以上。

---

## (d) 编码完整数据集为 uint16 NumPy 数组

| 输出文件 | Token 数 | 文件大小 | 编码耗时 |
|----------|----------|----------|----------|
| `tinystories_train.npy` | 540,796,778 | 1,081.6 MB | 2,853.5s（≈ 47.6 min） |
| `tinystories_valid.npy` | 5,461,210 | 10.9 MB | 29.3s |
| `owt_train.npy` | 2,727,120,452 | 5,454.2 MB（≈ 5.3 GB） | 17,793.5s（≈ 4h56m） |
| `owt_valid.npy` | 66,401,098 | 132.8 MB | 442.2s（≈ 7.4 min） |

### 为什么选 uint16？

- `uint16` 范围：0 ~ 65,535
  - TinyStories vocab_size = 10,000 → 最大 ID 9,999 < 65,535 ✓
  - OWT vocab_size = 32,000 → 最大 ID 31,999 < 65,535 ✓
- 相比 `uint32`（4 字节/token）节省一半存储空间
- 不能用 `uint8`（最大 255），无法存储 BPE 合并后的 token ID

---

## 总耗时

| 阶段 | 耗时 |
|------|------|
| OWT BPE 训练（32,000 次合并） | ≈ 4h13m |
| TinyStories 数据集编码 | ≈ 48min |
| OWT 数据集编码 | ≈ 5h03m |
| **总计** | **≈ 10h16m** |

所有输出文件位于服务器 `~/workspace/assignment1/tokenizer/data/`。
