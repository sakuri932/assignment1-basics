"""
Transformer 语言模型的神经网络基础组件。

从零实现以下组件（不使用 torch.nn.functional 的高阶层实现）：
- Linear: 无偏置线性变换
- Embedding: Token 嵌入查找
- RMSNorm: 均方根层归一化
- silu: SiLU 激活函数（Swish）
- SwiGLU: 带门控的位置前馈网络
- RotaryPositionalEmbedding: 旋转位置编码 (RoPE)
- scaled_dot_product_attention: 缩放点积注意力
- MultiHeadSelfAttention: 因果多头自注意力 + RoPE

所有模块都继承自 torch.nn.Module，遵循 PyTorch 模块化规范。

References:
- Vaswani et al., 2017: Attention Is All You Need
- Zhang & Sennrich, 2019: Root Mean Square Layer Normalization
- Su et al., 2021: RoFormer: Enhanced Transformer with Rotary Position Embedding
- Touvron et al., 2023: LLaMA: Open and Efficient Foundation Language Models
"""

import math
import torch
import torch.nn as nn
from einops import rearrange, einsum


class Linear(nn.Module):
    """
    无偏置线性变换: y = xW^T

    在现代大语言模型（GPT-3、LLaMA 等）中，线性层通常不包含偏置项，
    原因是偏置项带来的参数量和计算量提升不显著，而去掉偏置可以简化分布。

    权重矩阵 W 形状为 (d_out, d_in)，作为 nn.Parameter 存储。
    前向传播计算 y = x @ W.T，其中 x 形状为 (..., d_in)。

    权重初始化: 截断正态分布 N(0, σ²)，σ² = 2/(d_in + d_out)，截断于 [-3σ, 3σ]
    这是 Glorot/Xavier 初始化的变体，适用于前归一化 Transformer。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        # 存储 W: (d_out, d_in)，使用截断正态初始化
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算线性变换 y = xW^T。

        Args:
            x: 输入张量，形状为 (..., d_in)

        Returns:
            输出张量，形状为 (..., d_out)
        """
        # x: (..., d_in), weight: (d_out, d_in)
        # x @ weight.T: (..., d_out)
        return x @ self.weight.T


class Embedding(nn.Module):
    """
    Token 嵌入查找层: 将整数 token ID 映射为稠密向量。

    嵌入矩阵形状为 (vocab_size, d_model)。
    前向传播通过整数索引直接查找对应行向量。

    这等价于对 one-hot 向量做矩阵乘法，但直接索引效率更高。

    权重初始化: 截断正态分布 N(0, 1)，截断于 [-3, 3]
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        # 嵌入矩阵: (vocab_size, d_model)
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        查找 token ID 对应的嵌入向量。

        Args:
            token_ids: 整数张量，形状为 (...)

        Returns:
            嵌入向量，形状为 (..., d_model)
        """
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
    均方根层归一化 (Root Mean Square Layer Normalization)。

    与标准 LayerNorm 不同，RMSNorm 只按 RMS 缩放，不进行均值中心化：
        RMSNorm(a_i) = a_i / RMS(a) * g_i
        RMS(a) = sqrt(1/d * Σ a_i² + ε)

    其中 g_i 是可学习的增益参数（初始化为 1），ε 是数值稳定项（通常 1e-5）。

    RMSNorm 比 LayerNorm 计算更简单（省去均值计算），且在 LLaMA 等模型中
    被验证效果相当甚至更好。

    数值稳定性：在 float32 精度下进行归一化，避免 float16/bfloat16 的溢出。
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.eps = eps
        # 可学习的增益参数，初始化为全 1
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入张量应用 RMSNorm。

        先 upccast 到 float32 避免数值溢出，计算后 downcast 回原始精度。

        Args:
            x: 输入张量，形状为 (..., d_model)

        Returns:
            归一化后的张量，形状与输入相同
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)  # Upcast 避免平方运算时的溢出

        # 沿最后一维（d_model）计算 RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 归一化并应用可学习增益
        result = x / rms * self.weight

        return result.to(in_dtype)  # Downcast 回原始精度


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU (Sigmoid Linear Unit) 激活函数，又称 Swish。

    公式: SiLU(x) = x * σ(x) = x / (1 + e^{-x})

    与 ReLU 相比，SiLU 在零点处平滑（无折点），且允许负值区间的小梯度。
    在语言模型的前馈网络中已被证明优于 ReLU 和 GELU。

    Args:
        x: 任意形状的输入张量

    Returns:
        应用 SiLU 后的张量，形状与输入相同
    """
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """
    带 SwiGLU 门控的位置前馈网络。

    公式: FFN(x) = W₂ · (SiLU(W₁x) ⊙ W₃x)

    其中:
    - W₁, W₃: 上投影矩阵 (d_model → d_ff)，维度约为 8/3 * d_model
    - W₂: 下投影矩阵 (d_ff → d_model)
    - ⊙: 逐元素乘法（门控机制）

    门控线性单元 (GLU) 的作用：
    SiLU(W₁x) 作为"门"控制信息流通，W₃x 作为"值"携带内容信息。
    两者相乘后，模型可以动态地决定哪些特征需要传递，哪些需要抑制，
    从而缓解深度网络中的梯度消失问题。

    d_ff 设定为 ≈ 8/3 * d_model，并舍入为 64 的倍数以利用硬件矩阵单元。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if d_ff is None:
            # 默认: 8/3 * d_model，向上舍入为 64 的倍数
            d_ff_raw = int(8 / 3 * d_model)
            d_ff = ((d_ff_raw + 63) // 64) * 64

        self.d_ff = d_ff
        # 门控投影（用于计算 SiLU 门）
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        # 下投影（将 d_ff 压缩回 d_model）
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        # 值投影（门控的被乘数）
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用 SwiGLU 前馈变换。

        Args:
            x: 输入张量，形状为 (..., d_model)

        Returns:
            输出张量，形状为 (..., d_model)
        """
        # 门控分支：SiLU(W₁x)
        gate = silu(self.w1(x))
        # 值分支：W₃x
        up = self.w3(x)
        # 逐元素门控
        gated = gate * up
        # 下投影回 d_model
        return self.w2(gated)


class RotaryPositionalEmbedding(nn.Module):
    """
    旋转位置编码 (Rotary Position Embedding, RoPE)。

    RoPE 通过对 Query 和 Key 向量施加位置相关的旋转来注入位置信息。
    对于位置 i 和维度对 k，旋转角度为：
        θ_{i,k} = i / Θ^{2(k-1)/d}  (其中 Θ 通常为 10000)

    旋转公式（对每对相邻维度 (2k-1, 2k)）：
        q'_{2k-1} = q_{2k-1} * cos(θ_{i,k}) - q_{2k} * sin(θ_{i,k})
        q'_{2k}   = q_{2k-1} * sin(θ_{i,k}) + q_{2k} * cos(θ_{i,k})

    RoPE 的优势：
    1. 无可学习参数，计算高效
    2. 相对位置信息通过 Q·K 内积自然保留
    3. 可以外推到训练时未见过的更长序列

    预计算 cos/sin 缓冲区（不作为参数），避免每次前向传播重复计算。
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 预计算逆频率: θ_k = 1/Θ^{2(k-1)/d} for k=0,1,...,d/2-1
        # dim_indices: [0, 2, 4, ..., d_k-2]，形状 (d_k/2,)
        dim_indices = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (theta ** (dim_indices / d_k))

        # 预计算所有位置的 cos/sin 值
        # positions: (max_seq_len,)
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)

        # angles[i, k] = i * inv_freq[k]，形状 (max_seq_len, d_k/2)
        angles = torch.outer(positions, inv_freq)

        # 注册为 buffer（非参数，但随模型移动设备/保存）
        # persistent=False: 不保存到 state_dict（可按需重建）
        self.register_buffer("cos_buffer", torch.cos(angles), persistent=False)
        self.register_buffer("sin_buffer", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        对 Query 或 Key 张量应用 RoPE 旋转。

        Args:
            x: Query 或 Key 张量，形状为 (..., seq_len, d_k)
            token_positions: 位置索引张量，形状为 (..., seq_len)
                             值为 [0, max_seq_len) 范围内的整数

        Returns:
            旋转后的张量，形状与输入相同
        """
        # 按位置索引取 cos/sin 值
        # token_positions: (..., seq_len) → cos: (..., seq_len, d_k/2)
        cos = self.cos_buffer[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin_buffer[token_positions]  # (..., seq_len, d_k/2)

        # 将 x 按奇偶维度分割，分别对应旋转矩阵的两行
        x_even = x[..., ::2]   # (..., seq_len, d_k/2) - 偶数下标维度
        x_odd  = x[..., 1::2]  # (..., seq_len, d_k/2) - 奇数下标维度

        # 应用 2D 旋转（参见 eq. 8 in Su et al., 2021）
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos

        # 将旋转后的偶数和奇数维度重新交错拼合
        # stack → (..., seq_len, d_k/2, 2)，reshape → (..., seq_len, d_k)
        x_rotated = torch.stack([x_rot_even, x_rot_odd], dim=-1)
        return x_rotated.flatten(-2)  # 展平最后两个维度


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    缩放点积注意力: Attention(Q, K, V) = softmax(QK^T / √d_k) V

    缩放因子 1/√d_k 防止点积随维度增大而过大，从而避免 softmax 落入饱和区域
    （梯度极小）。这是 Vaswani et al. 2017 的关键设计之一。

    因果掩码: 通过在掩码为 False 的位置填充 -∞，使 softmax 后对应权重趋近于 0，
    实现"不关注未来 token"的效果。

    数值稳定性: 在计算 exp 前减去每行最大值（log-sum-exp trick），
    防止指数运算溢出。

    Args:
        Q: Query 张量，形状为 (..., queries, d_k)
        K: Key 张量，形状为 (..., keys, d_k)
        V: Value 张量，形状为 (..., keys, d_v)
        mask: 可选布尔掩码，形状为 (..., queries, keys)
              True=允许关注，False=屏蔽（设为 -∞）

    Returns:
        注意力输出，形状为 (..., queries, d_v)
    """
    d_k = Q.shape[-1]

    # 计算注意力分数: Q @ K^T / √d_k，形状 (..., queries, keys)
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(d_k)

    # 应用掩码：False 位置填充 -∞（softmax 后权重为 0）
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # 数值稳定的 softmax：减去每行最大值
    scores_max = scores.max(dim=-1, keepdim=True).values
    # 处理全 -∞ 的行（被完全掩蔽的查询位置）
    scores_max = torch.where(
        torch.isinf(scores_max),
        torch.zeros_like(scores_max),
        scores_max,
    )
    scores_shifted = scores - scores_max
    exp_scores = torch.exp(scores_shifted)

    # 被掩蔽的位置在求和时贡献为 0
    if mask is not None:
        exp_scores = exp_scores.masked_fill(~mask, 0.0)

    # 归一化得到注意力权重
    attn_weights = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + 1e-9)

    # 加权求和 Value: (..., queries, d_v)
    output = einsum(attn_weights, V, "... q k, ... k d -> ... q d")

    return output


class MultiHeadSelfAttention(nn.Module):
    """
    带 RoPE 的因果多头自注意力。

    多头注意力允许模型在不同子空间中并行关注序列的不同部分：
        MultiHead(x) = Concat(head₁, ..., headₕ) W_O
        head_i = Attention(xW_Q^i, xW_K^i, xW_V^i)

    实现细节：
    1. 使用单个大矩阵同时计算所有头的 Q/K/V（等效于分头分别计算，但更高效）
    2. 因果掩码：下三角矩阵，位置 i 只能关注 j ≤ i 的位置
    3. RoPE 应用于 Q 和 K（不应用于 V），为每个头提供相同的位置旋转

    d_k = d_v = d_model / num_heads（按 Vaswani et al. 的设置）

    权重键命名约定（与 adapters.py 中的 state_dict 键匹配）：
    - q_proj.weight, k_proj.weight, v_proj.weight: 形状 (d_model, d_model)
    - output_proj.weight: 形状 (d_model, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每头的维度

        # QKV 投影（所有头合并为一个大矩阵，效率更高）
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # 输出投影（将拼接后的多头输出投影回 d_model）
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # 旋转位置编码（所有头共享同一 RoPE 模块）
        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=self.d_k,
            max_seq_len=max_seq_len,
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        计算因果多头自注意力。

        Args:
            x: 输入张量，形状为 (..., seq_len, d_model)
            token_positions: 位置索引，形状为 (..., seq_len)。
                             如果为 None，使用 [0, 1, ..., seq_len-1]

        Returns:
            注意力输出，形状为 (..., seq_len, d_model)
        """
        *batch_dims, seq_len, _ = x.shape

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)

        # 计算 Q, K, V 投影: (..., seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 拆分为多头: (..., num_heads, seq_len, d_k)
        Q = rearrange(Q, "... s (h d) -> ... h s d", h=self.num_heads)
        K = rearrange(K, "... s (h d) -> ... h s d", h=self.num_heads)
        V = rearrange(V, "... s (h d) -> ... h s d", h=self.num_heads)

        # 扩展 token_positions 到与 Q/K 的批次维度相同
        # token_positions: (..., seq_len) → (..., 1, seq_len)，再广播到 (..., H, seq_len)
        tp = token_positions
        # 插入头维度（倒数第二个位置，即 seq_len 前面）
        tp = tp.unsqueeze(-2)  # (..., 1, seq_len)
        # 扩展到与 Q/K 前几维一致（但不强制 expand，让广播处理）
        tp = tp.expand(*Q.shape[:-1])  # (..., H, seq_len)

        # 对 Q 和 K 应用 RoPE（V 不旋转）
        Q = self.rope(Q, tp)
        K = self.rope(K, tp)

        # 构建因果掩码: 下三角矩阵，True 表示允许关注
        # 形状 (seq_len, seq_len)，广播到所有批次和头
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )

        # 计算多头注意力: (..., H, seq_len, d_k)
        attn_out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # 拼接所有头: (..., seq_len, d_model)
        attn_out = rearrange(attn_out, "... h s d -> ... s (h d)")

        # 输出投影
        return self.output_proj(attn_out)
