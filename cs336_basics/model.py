"""
完整的 Transformer 语言模型架构。

本文件实现：
- TransformerBlock: 前归一化（pre-norm）Transformer 块
- TransformerLM: 完整的 Transformer 语言模型

架构遵循 LLaMA 风格的现代 GPT 变体：
- 前归一化（Pre-norm）而非后归一化（Post-norm），稳定训练
- RMSNorm 代替 LayerNorm，计算更高效
- SwiGLU 代替 ReLU FFN，性能更优
- RoPE 代替绝对位置编码，更好地外推到长序列

References:
- Vaswani et al., 2017: Attention Is All You Need
- Touvron et al., 2023: LLaMA: Open and Efficient Foundation Language Models
- Zhang & Sennrich, 2019: Root Mean Square Layer Normalization
"""

import torch
import torch.nn as nn

from cs336_basics.nn_utils import (
    Embedding,
    Linear,
    MultiHeadSelfAttention,
    RMSNorm,
    SwiGLU,
)


class TransformerBlock(nn.Module):
    """
    前归一化（Pre-norm）Transformer 块。

    每个块包含两个子层：
    1. 多头自注意力（Multi-Head Self-Attention, MHSA）
    2. 位置前馈网络（Position-wise Feed-Forward Network, FFN）

    每个子层前先做 RMSNorm（前归一化），然后加上残差连接：
        y = x + MHSA(RMSNorm(x))
        z = y + FFN(RMSNorm(y))

    前归一化 vs 后归一化：
    - 后归一化（原始 Transformer）：在残差相加后再归一化
    - 前归一化（此实现）：在子层输入前先归一化

    前归一化的优势：
    - 存在一条从输入到输出的"干净残差流"，梯度传播更顺畅
    - 不需要对学习率调度和初始化特别小心
    - 是 GPT-3、LLaMA、PaLM 等当代模型的标准选择

    注意：前归一化需要在最后一个 Transformer 块的输出后额外加一个 RMSNorm
    （在 TransformerLM 中实现），因为最后一块的输出没有经过归一化。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        """
        Args:
            d_model: 模型的隐藏维度（嵌入维度）
            num_heads: 多头注意力的头数，d_model 必须能被整除
            d_ff: FFN 内层维度（通常约为 8/3 * d_model）
            max_seq_len: 最大序列长度（用于预计算 RoPE 缓冲区）
            theta: RoPE 的 Θ 参数（通常为 10000）
            device, dtype: PyTorch 设备和数据类型
        """
        super().__init__()

        # 第一个 RMSNorm：在自注意力子层之前
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)

        # 多头自注意力（含 RoPE）
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )

        # 第二个 RMSNorm：在 FFN 子层之前
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

        # SwiGLU 前馈网络
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        前向传播：依次应用 MHSA 和 FFN，各带残差连接。

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            token_positions: 位置索引，形状为 (batch_size, seq_len) 或 (seq_len,)

        Returns:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        # 自注意力子层：前归一化 + 注意力 + 残差
        x = x + self.attn(self.ln1(x), token_positions)

        # FFN 子层：前归一化 + SwiGLU + 残差
        x = x + self.ffn(self.ln2(x))

        return x


class TransformerLM(nn.Module):
    """
    完整的 Transformer 语言模型（解码器架构）。

    完整架构（参见论文 Figure 1）：
    1. Token 嵌入层：将整数 token ID 映射为 d_model 维向量
    2. N 个 Transformer 块（堆叠）
    3. 最终 RMSNorm：对最后一个 Transformer 块的输出进行归一化
    4. LM 头（语言模型头）：将 d_model 投影到 vocab_size 维的 logits

    语言模型的任务：给定前 t 个 token，预测第 t+1 个 token 的概率分布。
    模型输出 logits（未归一化分数），对应每个词表条目的对数概率（取 softmax 后）。

    关键参数与典型值（参考 TinyStories 实验）：
    - vocab_size: 10000（TinyStories）或 32000（OpenWebText）
    - context_length: 256（短序列）或 1024（较长序列）
    - d_model: 512（小模型）
    - num_layers: 4（约 17M 非嵌入参数）
    - num_heads: 16
    - d_ff: 1344（≈ 8/3 * 512，取 64 倍数）
    - rope_theta: 10000
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        """
        Args:
            vocab_size: 词表大小（决定嵌入矩阵和 LM 头的维度）
            context_length: 最大上下文长度（用于 RoPE 预计算）
            d_model: 模型隐藏维度
            num_layers: Transformer 块数量
            num_heads: 每个注意力层的头数
            d_ff: FFN 内层维度
            rope_theta: RoPE 的 Θ 参数
            device, dtype: PyTorch 设备和数据类型
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

        # Token 嵌入层：将 token ID 映射为 d_model 维向量
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        # 堆叠 num_layers 个 Transformer 块
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # 最终归一化层（前归一化架构在最后需要额外一个 RMSNorm）
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

        # LM 头：将 d_model 投影到 vocab_size 维 logits（无偏置）
        # 注意：不共享嵌入权重（与原始 GPT-2 不同，为简化实现）
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        语言模型前向传播。

        给定 token ID 序列，输出每个位置上词表的 logits。
        logits[i] 代表在前 i 个 token 的条件下，第 i+1 个 token 的未归一化对数概率。

        Args:
            token_ids: 整数张量，形状为 (batch_size, seq_len)，值在 [0, vocab_size) 范围内

        Returns:
            logits 张量，形状为 (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape

        # 1. Token 嵌入：(batch, seq_len) → (batch, seq_len, d_model)
        x = self.token_embeddings(token_ids)

        # 创建位置索引 [0, 1, ..., seq_len-1]
        positions = torch.arange(seq_len, device=token_ids.device)

        # 2. 逐层通过 Transformer 块
        for layer in self.layers:
            x = layer(x, positions)

        # 3. 最终 RMSNorm
        x = self.ln_final(x)

        # 4. LM 头：(batch, seq_len, d_model) → (batch, seq_len, vocab_size)
        logits = self.lm_head(x)

        return logits
