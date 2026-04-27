from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

# 导入我们实现的所有模块
from cs336_basics.tokenizer import Tokenizer, train_bpe
from cs336_basics.nn_utils import (
    Embedding,
    Linear,
    MultiHeadSelfAttention,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    scaled_dot_product_attention,
    silu,
)
from cs336_basics.model import TransformerBlock, TransformerLM
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.training import (
    cross_entropy_loss,
    get_batch,
    load_checkpoint,
    save_checkpoint,
    softmax,
)


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    给定 Linear 层的权重，计算输入张量的线性变换。
    实例化我们的 Linear 模块，加载权重，然后执行前向传播。
    """
    # 创建 Linear 层（形状匹配 weights）
    layer = Linear(in_features=d_in, out_features=d_out)
    # 加载预设权重（直接赋值，不用 load_state_dict 以避免格式问题）
    layer.weight.data = weights
    # 前向传播（不需要梯度）
    with torch.no_grad():
        return layer(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    给定 Embedding 层的权重，查找 token IDs 对应的嵌入向量。
    """
    layer = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    layer.weight.data = weights
    with torch.no_grad():
        return layer(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    给定 SwiGLU 网络的权重，计算前馈变换。
    """
    layer = SwiGLU(d_model=d_model, d_ff=d_ff)
    # 加载各子层权重
    layer.w1.weight.data = w1_weight
    layer.w2.weight.data = w2_weight
    layer.w3.weight.data = w3_weight
    with torch.no_grad():
        return layer(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    计算缩放点积注意力。
    """
    with torch.no_grad():
        return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    给定 QKV 和输出投影权重，计算不带 RoPE 的多头自注意力。
    注意：不使用 RoPE，但我们的实现默认带 RoPE。
    这里使用一个足够大的 max_seq_len 并禁用 RoPE 的影响。
    实际上我们直接用 RoPE 版本，RoPE 在相对位置为 0 时不改变方向（只是旋转）。
    为了通过测试，这里构造一个不带 RoPE 的版本。
    """
    # 创建足够大的 max_seq_len
    seq_len = in_features.shape[-2]
    max_seq_len = max(seq_len, 2048)

    # 使用我们的 MultiHeadSelfAttention，但禁用 RoPE 效果
    # 通过设置 theta 为无穷大近似（所有角度为 0）实际上不旋转
    # 更好的方式：直接手动实现不带 RoPE 的多头注意力

    # 手动实现（不带 RoPE）
    d_k = d_model // num_heads

    # Q, K, V 投影
    Q = in_features @ q_proj_weight.T  # (..., seq_len, d_model)
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T

    # 拆分为多头
    from einops import rearrange
    Q = rearrange(Q, "... s (h d) -> ... h s d", h=num_heads)
    K = rearrange(K, "... s (h d) -> ... h s d", h=num_heads)
    V = rearrange(V, "... s (h d) -> ... h s d", h=num_heads)

    # 因果掩码
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool)
    )

    # 计算注意力
    attn_out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

    # 拼接头
    attn_out = rearrange(attn_out, "... h s d -> ... s (h d)")

    # 输出投影
    output = attn_out @ o_proj_weight.T

    return output


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    给定权重和可选的位置索引，计算带 RoPE 的多头自注意力。
    """
    layer = MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        device=in_features.device,
    )
    # 加载权重
    layer.q_proj.weight.data = q_proj_weight
    layer.k_proj.weight.data = k_proj_weight
    layer.v_proj.weight.data = v_proj_weight
    layer.output_proj.weight.data = o_proj_weight

    with torch.no_grad():
        return layer(in_features, token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    对 Query 或 Key 张量应用 RoPE 旋转。
    """
    rope = RotaryPositionalEmbedding(
        theta=theta,
        d_k=d_k,
        max_seq_len=max_seq_len,
        device=in_query_or_key.device,
    )
    with torch.no_grad():
        return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    给定权重字典和输入，运行一个 Transformer 块。

    weights 键：
    - attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight, attn.output_proj.weight
    - ln1.weight, ln2.weight
    - ffn.w1.weight, ffn.w2.weight, ffn.w3.weight
    """
    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
        device=in_features.device,
    )
    block.load_state_dict(weights)
    block.eval()
    with torch.no_grad():
        return block(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """
    给定权重字典和输入 token 索引，运行完整 Transformer 语言模型。
    """
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=in_indices.device,
    )
    model.load_state_dict(weights)
    model.eval()
    with torch.no_grad():
        return model(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    给定 RMSNorm 的增益权重，对输入进行归一化。
    """
    layer = RMSNorm(d_model=d_model, eps=eps)
    layer.weight.data = weights
    with torch.no_grad():
        return layer(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """
    对输入张量应用 SiLU 激活函数。
    """
    with torch.no_grad():
        return silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从 token ID 数组中随机采样一个批次。
    """
    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    对输入张量在指定维度应用 softmax。
    """
    with torch.no_grad():
        return softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """
    计算平均交叉熵损失。
    """
    with torch.no_grad():
        return cross_entropy_loss(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    原地裁剪参数梯度的 L2 范数。
    """
    gradient_clipping(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    """
    返回实现了 AdamW 的优化器类。
    """
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    计算给定步数的余弦学习率调度值。
    """
    return get_lr_cosine_schedule(
        it=it,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    )


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    保存模型、优化器状态和迭代步数到检查点。
    """
    save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    从检查点恢复模型和优化器状态，返回保存的迭代步数。
    """
    return load_checkpoint(src, model, optimizer)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """
    给定词表、合并列表和特殊 token，创建 BPE 分词器。
    """
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    给定语料库路径，训练 BPE 分词器并返回词表和合并列表。
    """
    return train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        **kwargs,
    )
