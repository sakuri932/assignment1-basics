# CS336 Assignment 1 测试体系详解

本文档详细介绍 `tests/` 目录下测试代码的工作机制，帮助你理解每个测试的逻辑、测试与实现代码之间的关系，以及如何高效地使用测试来验证自己的实现。

---

## 目录

1. [目录结构总览](#1-目录结构总览)
2. [核心基础设施](#2-核心基础设施)
   - [conftest.py — pytest 配置与 Fixture](#21-conftestpy--pytest-配置与-fixture)
   - [common.py — 公共工具](#22-commonpy--公共工具)
   - [adapters.py — 适配器层](#23-adapterspy--适配器层)
3. [适配器模式详解](#3-适配器模式详解)
4. [Snapshot 快照测试机制](#4-snapshot-快照测试机制)
5. [各测试文件详解](#5-各测试文件详解)
   - [test_tokenizer.py — 分词器测试](#51-test_tokenizerpy--分词器测试)
   - [test_train_bpe.py — BPE 训练测试](#52-test_train_bpepy--bpe-训练测试)
   - [test_model.py — 模型组件测试](#53-test_modelpy--模型组件测试)
   - [test_nn_utils.py — 训练工具测试](#54-test_nn_utilspy--训练工具测试)
   - [test_optimizer.py — 优化器测试](#55-test_optimizerpy--优化器测试)
   - [test_serialization.py — 检查点测试](#56-test_serializationpy--检查点测试)
   - [test_data.py — 数据处理测试](#57-test_datapy--数据处理测试)
6. [fixtures/ 目录](#6-fixtures-目录)
7. [_snapshots/ 目录](#7-_snapshots-目录)
8. [如何运行测试](#8-如何运行测试)
9. [测试流程全景图](#9-测试流程全景图)

---

## 1. 目录结构总览

```
tests/
├── __init__.py                    # 使 tests 成为 Python 包（支持相对导入）
├── conftest.py                    # pytest 全局配置、fixture、快照工具
├── common.py                      # 共用路径常量和工具函数
├── adapters.py                    # 适配器层：连接测试与你的实现
├── fixtures/                      # 测试用的固定输入文件
│   ├── gpt2_vocab.json            # GPT-2 词表（用于分词器测试）
│   ├── gpt2_merges.txt            # GPT-2 合并规则
│   ├── corpus.en                  # 英文语料（BPE 训练测试）
│   ├── address.txt                # 地址文本（分词器 roundtrip 测试）
│   ├── german.txt                 # 德文文本（多语言测试）
│   ├── tinystories_sample.txt     # TinyStories 样本（5KB 级别）
│   ├── tinystories_sample_5M.txt  # TinyStories 样本（5MB 级别，内存测试）
│   ├── train-bpe-reference-vocab.json   # BPE 训练期望词表
│   ├── train-bpe-reference-merges.txt   # BPE 训练期望合并规则
│   └── ts_tests/
│       ├── model.pt               # 预训练模型权重（PyTorch 格式）
│       └── model_config.json      # 模型配置参数
├── _snapshots/                    # 快照文件（自动生成，不要手动修改）
│   ├── test_linear.npz
│   ├── test_embedding.npz
│   └── ...（其他 .npz 和 .pkl 文件）
├── test_tokenizer.py              # 分词器 encode/decode 测试
├── test_train_bpe.py              # BPE 训练测试
├── test_model.py                  # 模型各组件的数值正确性测试
├── test_nn_utils.py               # softmax / cross_entropy / gradient_clipping 测试
├── test_optimizer.py              # AdamW 优化器和学习率调度测试
├── test_serialization.py          # 检查点保存/加载测试
└── test_data.py                   # get_batch 数据采样测试
```

---

## 2. 核心基础设施

### 2.1 conftest.py — pytest 配置与 Fixture

`conftest.py` 是 pytest 的特殊配置文件，在整个 `tests/` 包内自动加载，无需手动导入。它提供两类核心功能：

#### 快照测试工具类

**`NumpySnapshot`（用于张量/数组的数值比较）**

```python
class NumpySnapshot:
    def assert_match(self, actual, rtol=1e-4, atol=1e-2, test_name=..., force_update=...):
        ...
```

- 保存格式：`.npz`（NumPy 压缩格式，可存多个数组）
- 比较方式：`np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-2)`
- 允许微小浮点误差，适合神经网络输出的验证

**`Snapshot`（用于任意 Python 对象的精确比较）**

```python
class Snapshot:
    def assert_match(self, actual, test_name=..., force_update=...):
        ...
```

- 保存格式：`.pkl`（Python pickle 序列化）
- 比较方式：`assert actual == expected`（精确相等）
- 用于词表 `dict`、合并列表 `list` 等结构化数据

#### pytest Fixture

Fixture 是 pytest 的依赖注入机制。你在测试函数参数中写上 fixture 名称，pytest 就会自动计算并传入对应的值。**Fixture 之间可以互相依赖**，形成链式计算：

```
d_head=16, n_heads=4
       ↓
    d_model = n_heads * d_head = 64
       ↓
    q = torch.randn(batch_size=4, n_queries=12, d_model=64)  # seed=1
```

**完整 Fixture 依赖图：**

| Fixture | 值 | 依赖 |
|---|---|---|
| `n_layers` | 3 | — |
| `vocab_size` | 10,000 | — |
| `batch_size` | 4 | — |
| `n_queries` | 12 | — |
| `n_keys` | 16 | — |
| `n_heads` | 4 | — |
| `d_head` | 16 | — |
| `d_model` | 64 | `n_heads * d_head` |
| `d_ff` | 128 | — |
| `theta` | 10000.0 | — |
| `q` | `randn(4, 12, 64)` | `batch_size, n_queries, d_model` |
| `k` | `randn(4, 16, 64)` | `batch_size, n_keys, d_model` |
| `v` | `randn(4, 16, 64)` | `batch_size, n_keys, d_model` |
| `in_embeddings` | `randn(4, 12, 64)` | `batch_size, n_queries, d_model` |
| `mask` | `randn(4, 12, 16) > 0.5` | `batch_size, n_queries, n_keys` |
| `in_indices` | `randint(0, 10000, (4, 12))` | `batch_size, n_queries` |
| `pos_ids` | `arange(0, 12)` | `n_queries` |
| `numpy_snapshot` | `NumpySnapshot` 实例 | `request`（pytest 内置）|
| `snapshot` | `Snapshot` 实例 | `request` |
| `ts_state_dict` | `(state_dict, config)` | `request` |

**`ts_state_dict` fixture** 加载 `fixtures/ts_tests/model.pt`（参考实现的预训练权重）和 `model_config.json`。这是 Transformer 模型测试的权重来源，权重键名中的 `_orig_mod.` 前缀会被自动去除：

```python
state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
```

---

### 2.2 common.py — 公共工具

```python
FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent) / "fixtures"
```

所有测试文件通过 `from .common import FIXTURES_PATH` 导入这个路径常量，避免硬编码路径。

**`gpt2_bytes_to_unicode()`** 实现了 GPT-2 的字节→Unicode 可打印字符映射：

- 字节 0–255 中，可直接打印的 188 个字符保持原样（如 `!` → `'!'`）
- 剩余 68 个不可打印字符映射到 `chr(256+n)` 范围（如空格 `32` → `'Ġ'`）
- 这使得词表和合并规则可以用纯 JSON/文本格式序列化

该函数的逆映射（`{v: k for k, v in gpt2_bytes_to_unicode().items()}`）用于将 GPT-2 的 JSON 词表文件中的字符串表示还原为原始字节。

---

### 2.3 adapters.py — 适配器层

`adapters.py` 是测试体系与你的实现代码之间的**唯一接口**。它的作用：

1. **隔离**：测试只依赖 `adapters.py` 中的函数，不直接导入你的实现
2. **标准化**：将你的实现包装成测试期望的统一调用接口
3. **灵活性**：如果你的实现接口有细微差异，只需修改 `adapters.py`，无需改测试

所有 adapter 函数都遵循相同模式：接收权重和输入 → 实例化你的模块 → 加载权重 → 在 `torch.no_grad()` 下执行前向传播 → 返回结果。

---

## 3. 适配器模式详解

测试的整体调用链如下：

```
测试函数
  ↓ 调用
adapters.py 中的 run_xxx() 函数
  ↓ 导入
cs336_basics/ 中你的实现（Tokenizer, Linear, TransformerLM 等）
```

**示例：`test_linear` 的完整调用链**

```python
# test_model.py
def test_linear(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    w1_weight = ts_state_dict[0]["layers.0.ffn.w1.weight"]
    output = run_linear(d_in=d_model, d_out=d_ff, weights=w1_weight, in_features=in_embeddings)
    numpy_snapshot.assert_match(output)
    # ↑ 用函数名 "test_linear" 作为快照文件名，与 tests/_snapshots/test_linear.npz 比较

# adapters.py
def run_linear(d_in, d_out, weights, in_features):
    layer = Linear(in_features=d_in, out_features=d_out)  # 你实现的 Linear
    layer.weight.data = weights                              # 加载参考权重
    with torch.no_grad():
        return layer(in_features)                           # 执行前向传播
```

**`ts_state_dict` 权重的来源**：`fixtures/ts_tests/model.pt` 是用参考实现训练好的小型 Transformer，权重键名如：

```
token_embeddings.weight          → 词嵌入
layers.0.attn.q_proj.weight     → 第0层 Q 投影
layers.0.attn.k_proj.weight     → 第0层 K 投影
layers.0.attn.v_proj.weight     → 第0层 V 投影
layers.0.attn.output_proj.weight → 第0层输出投影
layers.0.ffn.w1.weight          → 第0层 FFN W1
layers.0.ffn.w2.weight          → 第0层 FFN W2
layers.0.ffn.w3.weight          → 第0层 FFN W3
layers.0.ln1.weight             → 第0层 LayerNorm 1
layers.0.ln2.weight             → 第0层 LayerNorm 2
layers.1.xxx / layers.2.xxx     → 第1、2层同上
output_layer_norm.weight        → 输出归一化
lm_head.weight                  → 语言模型头
```

测试中 `ts_state_dict[0]` 是 state dict，`ts_state_dict[1]` 是 config dict（模型维度参数）。

---

## 4. Snapshot 快照测试机制

快照测试（Snapshot Testing）的思路是：**第一次运行时保存"黄金输出"，后续运行时与之比较**。

### 工作流程

**第一次运行（生成快照）**：
- 传入 `force_update=True` 或删除旧 `.npz` 文件
- 测试会将当前输出保存到 `tests/_snapshots/<test_name>.npz`
- 这一步需要你的实现已经正确

**后续运行（验证快照）**：
- 加载已保存的 `.npz` 文件
- 用 `np.testing.assert_allclose` 比较当前输出与保存的黄金值
- 如果数值差异超过容差（`rtol=1e-4, atol=1e-2`），测试失败

### 快照文件命名

快照文件名 = pytest 测试函数名（`request.node.name`）。例如：
- `test_linear` → `_snapshots/test_linear.npz`
- `test_swiglu` → `_snapshots/test_swiglu.npz`
- `test_adamw` → `_snapshots/test_adamw.npz`

### 与固定期望值的区别

对于某些有精确期望值的测试（如 `test_get_lr_cosine_schedule`），测试直接硬编码期望值，不用快照：

```python
expected_lrs = [0, 0.14285..., 0.28571..., ...]
numpy.testing.assert_allclose(numpy.array(actual_lrs), numpy.array(expected_lrs))
```

这类测试在任何环境下期望值都相同，不依赖具体硬件或随机数。

---

## 5. 各测试文件详解

### 5.1 test_tokenizer.py — 分词器测试

**测试目标**：验证 `Tokenizer.encode()` 和 `decode()` 的正确性

**核心辅助函数**：

```python
def get_tokenizer_from_vocab_merges_path(vocab_path, merges_path, special_tokens=None):
    # 1. 加载 GPT-2 词表 JSON 文件（Unicode 字符格式）
    # 2. 用 gpt2_byte_decoder 将 Unicode 字符还原为原始字节
    # 3. 构造 vocab: {int → bytes} 格式
    # 4. 加载合并规则，同样还原为 bytes 对
    # 5. 调用 adapters.get_tokenizer(vocab, merges, special_tokens)
```

**测试分类**：

| 测试名 | 类型 | 验证内容 |
|---|---|---|
| `test_roundtrip_empty` | Roundtrip | 空字符串 encode→decode 还原 |
| `test_roundtrip_single_character` | Roundtrip | 单 ASCII 字符 |
| `test_roundtrip_single_unicode_character` | Roundtrip | 单 Unicode 字符（如 🙃） |
| `test_roundtrip_ascii_string` | Roundtrip | 普通英文句子 |
| `test_roundtrip_unicode_string` | Roundtrip | 含重音符号的字符串 |
| `test_roundtrip_unicode_string_with_special_tokens` | Roundtrip + 特殊token | 含 `<\|endoftext\|>` 的字符串 |
| `test_empty_matches_tiktoken` | 与 tiktoken 对比 | 结果与 OpenAI tiktoken 一致 |
| `test_single_character_matches_tiktoken` | 与 tiktoken 对比 | 同上 |
| `test_single_unicode_character_matches_tiktoken` | 与 tiktoken 对比 | 同上 |
| `test_ascii_string_matches_tiktoken` | 与 tiktoken 对比 | 同上 |
| `test_unicode_string_matches_tiktoken` | 与 tiktoken 对比 | 同上 |
| `test_unicode_string_with_special_tokens_matches_tiktoken` | 与 tiktoken 对比 + 特殊token | 同上 |
| `test_overlapping_special_tokens` | 特殊 token 边界 | 重叠 token（`<\|eof\|><\|eof\|>` 视为双 token）长的优先匹配 |
| `test_address_roundtrip` | 文件级 Roundtrip | address.txt 文件 |
| `test_address_matches_tiktoken` | 文件级对比 | 同上 |
| `test_german_roundtrip` | 非英语 Roundtrip | german.txt（含德语字符） |
| `test_german_matches_tiktoken` | 非英语对比 | 同上 |
| `test_tinystories_sample_roundtrip` | 大文件 Roundtrip | tinystories_sample.txt |
| `test_tinystories_matches_tiktoken` | 大文件对比 | 同上 |
| `test_encode_special_token_trailing_newlines` | 边界 | 特殊 token 后跟换行符 |
| `test_encode_special_token_double_newline_non_whitespace` | 边界 | 特殊 token 与非空白字符相邻 |
| `test_encode_iterable_tinystories_sample_roundtrip` | 流式接口 | `encode_iterable()` 结果与 `encode()` 一致 |
| `test_encode_iterable_tinystories_matches_tiktoken` | 流式接口对比 | 同上 |
| `test_encode_iterable_memory_usage` | 内存约束（仅 Linux）| `encode_iterable` 在 5MB 文件下内存 ≤ 1MB |
| `test_encode_memory_usage` | 内存约束（仅 Linux，xfail）| `encode` 在 5MB 文件下预期**失败**（内存超限） |

**关键点**：
- "与 tiktoken 对比"测试要求你的分词结果与 OpenAI 官方 GPT-2 分词器**完全一致**（token ID 精确相等）
- `test_overlapping_special_tokens` 测试了最长匹配原则：`<|eof|><|eof|>` 和 `<|eof|>` 同时注册时，应匹配更长的双 token
- `test_encode_memory_usage` 是 `@pytest.mark.xfail`，表示这个测试**预期失败**，因为普通 `encode` 接口没有内存约束

**内存限制实现**：

```python
def memory_limit(max_mem):
    def decorator(f):
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())
            prev_limits = resource.getrlimit(resource.RLIMIT_AS)
            # 设置虚拟内存上限 = 当前RSS + 允许的增量
            resource.setrlimit(resource.RLIMIT_AS, (process.memory_info().rss + max_mem, -1))
            try:
                return f(*args, **kwargs)
            finally:
                resource.setrlimit(resource.RLIMIT_AS, prev_limits)  # 恢复限制
        return wrapper
    return decorator

@memory_limit(int(1e6))  # 1MB 上限
def _encode_iterable(tokenizer, iterable):
    yield from tokenizer.encode_iterable(iterable)
```

---

### 5.2 test_train_bpe.py — BPE 训练测试

**测试目标**：验证 `train_bpe()` 的速度、正确性和特殊 token 处理

#### `test_train_bpe_speed`

```python
start_time = time.time()
vocab, merges = run_train_bpe(input_path=FIXTURES_PATH/"corpus.en", vocab_size=500, ...)
assert end_time - start_time < 1.5  # 必须在 1.5 秒内完成
```

- 参考实现耗时约 0.38 秒；朴素实现（全量重扫）约 3 秒
- 这是对**增量更新策略**的隐式要求：不能每次 merge 后重新扫描整个语料

#### `test_train_bpe`

与 `fixtures/train-bpe-reference-merges.txt` 精确比较合并规则顺序，与 `fixtures/train-bpe-reference-vocab.json` 比较词表（键值集合，不要求顺序）。

#### `test_train_bpe_special_tokens`

```python
# 训练 vocab_size=1000，special_tokens=["<|endoftext|>"]
# 验证：除了 <|endoftext|> 本身，vocab 中任何 token 都不包含 b"<|"
for word_bytes in vocabs_without_specials:
    assert b"<|" not in word_bytes
```

然后使用 `snapshot.assert_match({"vocab_keys": ..., "vocab_values": ..., "merges": ...})` 做整体快照验证。这个测试使用的是 `.pkl` 格式快照（精确相等），而不是数值近似比较。

---

### 5.3 test_model.py — 模型组件测试

**测试目标**：用参考权重 (`ts_state_dict`) 验证每个模型组件的数值输出

所有测试都调用 `numpy_snapshot.assert_match(output)`，将输出与预先保存的黄金值比较（允许 `atol=1e-5` 到 `atol=1e-4` 的误差）。

| 测试 | 测试的组件 | 权重来源 | 特殊说明 |
|---|---|---|---|
| `test_linear` | `Linear` 前向传播 | `layers.0.ffn.w1.weight` | 输入形状 `(4, 12, 64)`，输出形状 `(4, 12, 128)` |
| `test_embedding` | `Embedding` 查找 | `token_embeddings.weight` | 输入为 token ID，输出为嵌入向量 |
| `test_swiglu` | `SwiGLU` FFN | `layers.0.ffn.{w1,w2,w3}.weight` | W1 gate + W3 门控后乘 W2 投影 |
| `test_scaled_dot_product_attention` | 缩放点积注意力 | — | Q/K/V 用随机 fixture，带 mask |
| `test_4d_scaled_dot_product_attention` | 缩放点积注意力 | — | 4D 张量（batch × head × seq × d），形状用 einops 转换 |
| `test_multihead_self_attention` | 不带 RoPE 的 MHA | `layers.0.attn.{q,k,v,output}_proj.weight` | |
| `test_multihead_self_attention_with_rope` | 带 RoPE 的 MHA | 同上 + `pos_ids` | `pos_ids` 形状从 `(12,)` reshape 为 `(1, 12)` |
| `test_transformer_block` | 完整 Transformer 块 | `layers.0.*` 所有权重 | 过滤键名后用 `load_state_dict` 加载 |
| `test_transformer_lm` | 完整 Transformer LM | 全部权重 | 误差容差较大 `atol=1e-4, rtol=1e-2` |
| `test_transformer_lm_truncated_input` | LM 对截断输入 | 全部权重 | `in_indices[..., :seq_len//2]`，验证可变长度输入 |
| `test_rmsnorm` | `RMSNorm` 归一化 | `layers.1.ln1.weight` | 注意用第1层而非第0层权重 |
| `test_rope` | RoPE 旋转编码 | — | 用随机 `in_embeddings`，`d_model=64`, `theta=10000` |
| `test_silu_matches_pytorch` | SiLU 激活函数 | — | 与 `F.silu()` 精确比较 |

**`test_transformer_block` 的权重过滤**：

```python
block_weights = {k.replace("layers.0.", ""): v for k, v in ts_state_dict[0].items() if "layers.0." in k}
# 结果键名：attn.q_proj.weight, ln1.weight, ffn.w1.weight, ...
```

---

### 5.4 test_nn_utils.py — 训练工具测试

**测试目标**：验证 `softmax`、`cross_entropy_loss`、`gradient_clipping` 的数值正确性

#### `test_softmax_matches_pytorch`

```python
expected = F.softmax(x, dim=-1)
run_softmax(x, dim=-1)         # 与 PyTorch 结果比较，atol=1e-5
run_softmax(x + 100, dim=-1)   # 加 100 后结果应相同（测试数值稳定性，用 log-sum-exp 技巧）
```

**关键**：softmax 数值稳定性——输入加常数不影响结果，但朴素实现会溢出。

#### `test_cross_entropy`

```python
expected = F.cross_entropy(inputs.view(-1, vocab_size), targets.view(-1))
run_cross_entropy(inputs.view(-1, vocab_size), targets.view(-1))
# 同时测试 1000 倍大的输入（测试数值稳定性）
```

#### `test_gradient_clipping`

```python
# 用 PyTorch 的 clip_grad_norm_ 作为参考
clip_grad_norm_(t1, max_norm)          # PyTorch 参考
run_gradient_clipping(t1_c, max_norm)  # 你的实现

# 注意：最后一个参数 requires_grad=False，测试是否正确跳过无梯度参数
t1[-1].requires_grad_(False)
```

---

### 5.5 test_optimizer.py — 优化器测试

**测试目标**：验证 `AdamW` 实现和余弦学习率调度

#### `test_adamw`

```python
pytorch_weights = _optimize(torch.optim.AdamW)   # PyTorch 官方实现
actual_weights = _optimize(get_adamw_cls())        # 你的实现

# 如果与 PyTorch 匹配，测试直接通过
if torch.allclose(actual_weights, pytorch_weights, atol=1e-4):
    return

# 否则，与快照中保存的参考实现结果比较
numpy_snapshot.assert_match(actual_weights, atol=1e-4)
```

**为什么这样设计**：AdamW 有两种等价的 weight decay 实现方式（在参数更新前或后应用），在浮点精度下结果略有差异。两种都可以接受。

`_optimize` 函数的测试场景：

```python
# 训练一个 3→2 的线性模型，目标 y = [x0+x1, -x2]
model = nn.Linear(3, 2, bias=False)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
for _ in range(1000):  # 1000 步优化
    loss = ((model(x) - y) ** 2).sum()
    loss.backward()
    optimizer.step()
```

#### `test_get_lr_cosine_schedule`

使用硬编码的 25 个期望值验证余弦学习率调度的每一步：

- 步骤 0-6（warmup）：线性从 0 增长到 `max_lr=1.0`
- 步骤 7-20（余弦衰减）：从 1.0 余弦衰减到 `min_lr=0.1`
- 步骤 21+（超过周期）：保持 `min_lr=0.1`

---

### 5.6 test_serialization.py — 检查点测试

**测试目标**：验证模型和优化器状态可以正确保存和加载

```python
# 定义测试网络
class _TestNet(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)
```

测试流程：
1. 训练 10 步
2. 调用 `run_save_checkpoint(model, optimizer, iteration=10, out=tmp_path/"checkpoint.pt")`
3. 创建**新的**模型和优化器实例
4. 调用 `run_load_checkpoint(src=..., model=new_model, optimizer=new_optimizer)`
5. 验证：
   - 返回的迭代步数 == 10
   - 新模型的参数与原模型**完全一致**（`assert_allclose`）
   - 新优化器的状态与原优化器**精确相等**（`are_optimizers_equal`）

`tmp_path` 是 pytest 内置 fixture，每次测试都提供一个唯一的临时目录，测试后自动清理。

---

### 5.7 test_data.py — 数据处理测试

**测试目标**：验证 `get_batch()` 的形状、偏移和随机性

测试场景：`dataset = np.arange(0, 100)`，`context_length=7`，`batch_size=32`

```python
for _ in range(1000):  # 运行 1000 次采样
    x, y = run_get_batch(dataset, batch_size=32, context_length=7, device="cpu")
    
    assert x.shape == (32, 7)   # 形状检查
    assert y.shape == (32, 7)
    np.testing.assert_allclose((x + 1).detach().numpy(), y.detach().numpy())  # y 总是 x 偏移 1
    
    starting_indices.update(x[:, 0].tolist())  # 收集起始索引
```

随机性验证：
- 有效起始索引为 `[0, 100-7-1]` = `[0, 92]`，共 93 个
- 期望每个起始索引出现次数 ≈ `(1000×32)/93 ≈ 344` 次
- 使用 5σ 范围（覆盖 99.99994% 概率）验证均匀分布

设备处理测试：
```python
with pytest.raises((RuntimeError, AssertionError)):
    run_get_batch(dataset, batch_size, context_length, device="cuda:99")  # 无效设备
```

---

## 6. fixtures/ 目录

| 文件 | 用途 | 使用者 |
|---|---|---|
| `gpt2_vocab.json` | GPT-2 词表（Unicode 字符格式，50,257 个 token） | `test_tokenizer.py` |
| `gpt2_merges.txt` | GPT-2 合并规则（50,000 条） | `test_tokenizer.py` |
| `corpus.en` | 英文语料，用于 BPE 训练速度和正确性测试 | `test_train_bpe.py` |
| `train-bpe-reference-vocab.json` | 对 corpus.en 训练 vocab_size=500 的期望词表 | `test_train_bpe.py` |
| `train-bpe-reference-merges.txt` | 对应的期望合并规则 | `test_train_bpe.py` |
| `address.txt` | 含地址格式的英文文本（测试数字/标点边界） | `test_tokenizer.py` |
| `german.txt` | 德文文本（含非 ASCII 字符） | `test_tokenizer.py` |
| `tinystories_sample.txt` | TinyStories 数据集样本（~5KB） | `test_tokenizer.py` |
| `tinystories_sample_5M.txt` | TinyStories 数据集样本（~5MB，内存测试） | `test_tokenizer.py`, `test_train_bpe.py` |
| `special_token_trailing_newlines.txt` | 特殊 token 后跟换行的边界测试文本 | `test_tokenizer.py` |
| `special_token_double_newlines_non_whitespace.txt` | 特殊 token 与非空白相邻的边界文本 | `test_tokenizer.py` |
| `ts_tests/model.pt` | 小型 Transformer 参考实现的权重 | `conftest.py (ts_state_dict)` |
| `ts_tests/model_config.json` | 对应模型的配置（`d_model`, `n_heads` 等） | `conftest.py (ts_state_dict)` |

---

## 7. _snapshots/ 目录

该目录由 `NumpySnapshot` 和 `Snapshot` 类自动管理：

- `.npz` 文件对应 `numpy_snapshot.assert_match()` 调用（数组，允许浮点误差）
- `.pkl` 文件对应 `snapshot.assert_match()` 调用（Python 对象，精确相等）

**不要手动编辑这些文件**。如果你的实现改变了，需要更新快照时，在测试代码中临时设置 `force_update=True`，或删除对应文件让测试重新生成。

生成规则：文件名 = 测试函数名（由 `request.node.name` 提供）。

---

## 8. 如何运行测试

本项目使用 `uv` 管理 Python 环境（Python 3.13），需要用 `uv run` 执行命令。

```bash
# 进入项目目录
cd /Users/admin/Documents/GitHub/CS336/assignment1-basics

# 运行所有测试
uv run pytest tests/

# 运行某个测试文件
uv run pytest tests/test_model.py

# 运行某个特定测试
uv run pytest tests/test_model.py::test_linear

# 运行名称包含关键词的测试
uv run pytest tests/ -k "linear"

# 显示详细输出
uv run pytest tests/ -v

# 显示 print 输出（调试用）
uv run pytest tests/ -s

# 只运行上次失败的测试
uv run pytest tests/ --lf

# 停止在第一个失败
uv run pytest tests/ -x

# 跳过某些测试
uv run pytest tests/ --ignore=tests/test_train_bpe.py

# 运行 BPE 训练测试（可能较慢）
uv run pytest tests/test_train_bpe.py -v

# 查看每个测试的耗时
uv run pytest tests/ --durations=10
```

**常见问题**：

- 如果测试显示 `FileNotFoundError: xxx.npz`，说明对应快照不存在。需要先用参考实现或正确实现生成快照，或联系 TA 获取 `_snapshots/` 目录
- 如果测试显示 `AssertionError: Array 'array' does not match snapshot`，说明你的实现输出与参考实现不同
- 内存限制测试（`test_encode_iterable_memory_usage`）只在 Linux 上运行，macOS 上会自动跳过

---

## 9. 测试流程全景图

```
pytest 启动
    │
    ├─ 加载 conftest.py（自动）
    │   ├─ 注册 NumpySnapshot, Snapshot 工具类
    │   └─ 注册所有 fixture（d_model, batch_size, q, k, v, ...)
    │
    ├─ 扫描 test_*.py 文件
    │   └─ 每个 test_xxx 函数 = 一个测试用例
    │
    └─ 对每个测试：
        │
        ├─ 1. 解析参数列表，注入 fixture
        │      test_linear(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff)
        │      → numpy_snapshot = NumpySnapshot(default_test_name="test_linear")
        │      → ts_state_dict = torch.load("fixtures/ts_tests/model.pt")
        │      → in_embeddings = torch.randn(4, 12, 64, seed=4)
        │      → d_model = 64, d_ff = 128
        │
        ├─ 2. 执行测试体
        │      → 调用 adapters.run_xxx()
        │      → adapters.run_xxx() 调用你的实现 cs336_basics.xxx
        │      → 你的实现返回张量
        │
        └─ 3. 断言
               → numpy_snapshot.assert_match(output)
               → 加载 _snapshots/test_linear.npz
               → np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-2)
               → 通过 ✓ 或 失败 ✗
```

**总结**：测试体系的核心设计原则是**将实现与测试完全解耦**。你只需实现 `cs336_basics/` 下的各个模块，测试框架通过 `adapters.py` 调用你的实现，再与预先保存的参考输出（快照或精确期望值）比较。当某个测试失败时，快照比较的误差信息会精确告诉你哪个数组、哪个位置的数值与参考不符。
