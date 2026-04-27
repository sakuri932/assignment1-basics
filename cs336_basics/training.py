"""
训练基础设施：数据加载、检查点、损失函数和训练循环。

本文件实现：
- cross_entropy_loss: 数值稳定的交叉熵损失
- get_batch: 随机采样训练批次
- save_checkpoint / load_checkpoint: 模型和优化器状态序列化
- softmax: 数值稳定的 softmax
- train: 完整的训练循环

交叉熵损失是语言模型训练的核心：
    ℓ(θ; D) = 1/|D| * 1/m * Σ_{x∈D} Σ_{i=1}^{m} -log p_θ(x_{i+1} | x_{1:i})

困惑度 (Perplexity) 是评估语言模型的标准指标：
    PPL = exp(平均交叉熵损失)
    PPL 越低表示模型对测试数据的预测越准确。
"""

import os
import math
import time
import json
import numpy as np
import torch
import torch.nn as nn
from typing import IO, BinaryIO


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    数值稳定的 softmax。

    标准 softmax 公式：softmax(v)_i = exp(v_i) / Σ exp(v_j)

    数值问题：当 v_i 较大（如 >80）时，exp(v_i) 会溢出为 inf。
    解决方案：利用 softmax 的平移不变性
        softmax(v - c) = softmax(v)（对任意常数 c）
    取 c = max(v)，使最大元素变为 0，避免 exp 溢出。

    Args:
        x: 输入张量，任意形状
        dim: 对哪个维度进行 softmax

    Returns:
        归一化后的概率分布张量，形状与输入相同
    """
    # 减去最大值保证数值稳定
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def cross_entropy_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    数值稳定的交叉熵损失。

    交叉熵定义：
        ℓ_i = -log softmax(o_i)[x_{i+1}]
            = -o_i[x_{i+1}] + log Σ_a exp(o_i[a])

    其中 o_i 是第 i 个位置的 logits，x_{i+1} 是目标 token。

    数值优化：
    1. 减去每行最大 logit（log-sum-exp trick）防止 exp 溢出
    2. 直接操作 log，避免 softmax 后再取 log 的精度损失

    公式推导：
        ℓ_i = -o_i[x_{i+1}] + log Σ_a exp(o_i[a])
            = -(o_i[x_{i+1}] - max_o) + log Σ_a exp(o_i[a] - max_o) + max_o - max_o
            = -(o_i[x_{i+1}] - max_o) + log Σ_a exp(o_i[a] - max_o)

    Args:
        inputs: logits 张量，形状为 (batch_size, vocab_size)，inputs[i][j] 是第 i 个样本属于类 j 的未归一化分数
        targets: 目标类别索引张量，形状为 (batch_size,)，值在 [0, vocab_size) 之间

    Returns:
        标量，所有样本的平均交叉熵损失
    """
    # 减去每行最大值（数值稳定）
    inputs_max = inputs.max(dim=-1, keepdim=True).values
    inputs_shifted = inputs - inputs_max  # (batch, vocab_size)

    # log-sum-exp: log Σ exp(x_i - max)
    log_sum_exp = torch.log(torch.exp(inputs_shifted).sum(dim=-1))  # (batch,)

    # 目标类别的 logit（已减去 max）
    # gather: 按 targets 索引取对应位置的 logit
    target_logits = inputs_shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (batch,)

    # 交叉熵 = -目标 logit + log-sum-exp
    per_sample_loss = log_sum_exp - target_logits  # (batch,)

    # 返回批次平均损失
    return per_sample_loss.mean()


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从 token ID 数组中随机采样训练批次。

    数据加载策略：
    - 整个数据集是一个连续的 token ID 序列（所有文档拼接，用特殊 token 分隔）
    - 每个训练样本是长度为 context_length 的连续子序列
    - 对应的标签是将输入序列右移一位（预测下一个 token）

    例如：数据集 = [x₁, x₂, x₃, x₄, x₅], context_length = 3
    - 输入：[x₂, x₃, x₄]
    - 标签：[x₃, x₄, x₅]

    使用 np.memmap 支持超大文件：数据集可能大于 RAM，
    memmap 按需从磁盘加载，只读取实际访问的部分。

    Args:
        dataset: 1D numpy 整数数组（token IDs）
        batch_size: 批次大小 B
        context_length: 每个样本的 token 长度 m
        device: PyTorch 设备字符串（'cpu', 'cuda:0', 'mps' 等）

    Returns:
        (x, y): 一对 LongTensor，形状均为 (batch_size, context_length)
                x 是输入序列，y 是对应的下一 token 标签
    """
    n = len(dataset)
    # 随机采样 batch_size 个起始位置
    # 确保 start + context_length < n（取 labels 需要多一位）
    starts = torch.randint(0, n - context_length, (batch_size,))

    # 构建输入和标签张量
    x_list = []
    y_list = []
    for s in starts:
        s = s.item()
        x_list.append(torch.tensor(dataset[s : s + context_length].astype(np.int64), dtype=torch.long))
        y_list.append(torch.tensor(dataset[s + 1 : s + context_length + 1].astype(np.int64), dtype=torch.long))

    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)

    return x, y


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    保存训练检查点（模型权重 + 优化器状态 + 迭代步数）。

    完整检查点包含恢复训练所需的全部状态：
    - model.state_dict(): 所有可学习参数（权重、bias 等）
    - optimizer.state_dict(): 优化器状态（AdamW 中为一阶矩和二阶矩估计）
    - iteration: 训练步数（用于恢复学习率调度）

    为什么需要保存优化器状态？
    AdamW 的一阶矩 m 和二阶矩 v 是有状态的：
    - 它们记录了梯度的历史信息，使后续更新更平滑
    - 如果只恢复模型权重而不恢复优化器状态，
      相当于重新开始优化，可能导致训练不稳定

    Args:
        model: 要保存的模型
        optimizer: 要保存的优化器
        iteration: 当前训练步数
        out: 输出文件路径或类文件对象
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    从检查点文件恢复模型和优化器状态。

    Args:
        src: 检查点文件路径或类文件对象
        model: 要恢复状态的模型（需与保存时结构一致）
        optimizer: 要恢复状态的优化器

    Returns:
        保存时的训练步数（用于继续学习率调度）
    """
    checkpoint = torch.load(src, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


def decode_text(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float | None = None,
    device: str = "cpu",
) -> str:
    """
    从语言模型生成文本（自回归解码）。

    解码过程（每步）：
    1. 将当前 token 序列输入模型，得到 logits
    2. 取最后一个位置的 logits（对应下一 token 的预测）
    3. 应用温度缩放：softmax(logits / τ)
    4. 可选 top-p（nucleus）采样：只保留概率质量前 p 的 token
    5. 从概率分布采样一个 token
    6. 将新 token 追加到序列，重复

    温度参数 τ（temperature scaling）：
    - τ → 0: 接近 greedy decoding（总选最大概率 token），输出保守
    - τ = 1: 标准采样
    - τ > 1: 更随机，输出多样但可能不连贯

    Top-p（nucleus）采样：
    - 只保留累积概率恰好达到 p 的最小 token 集合
    - 动态调整候选集大小，兼顾多样性和质量

    Args:
        model: 训练好的 TransformerLM
        tokenizer: Tokenizer 实例
        prompt: 起始文本（提示词）
        max_new_tokens: 最多生成的新 token 数
        temperature: 温度参数 τ
        top_p: nucleus sampling 的概率阈值（None 表示不使用）
        device: 设备字符串

    Returns:
        生成的完整文本（包含 prompt）
    """
    model.eval()

    # 编码 prompt
    input_ids = tokenizer.encode(prompt)
    token_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    # 找到 <|endoftext|> 的 ID（用于停止生成）
    eos_bytes = b"<|endoftext|>"
    eos_id = tokenizer.bytes_to_id.get(eos_bytes, None)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 截断到 context_length（避免超长序列）
            if hasattr(model, "context_length"):
                token_ids = token_ids[:, -model.context_length :]

            # 前向传播：取最后一位置的 logits
            logits = model(token_ids)  # (1, seq_len, vocab_size)
            next_token_logits = logits[0, -1, :]  # (vocab_size,)

            # 温度缩放
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # 转为概率
            probs = softmax(next_token_logits, dim=0)

            # Top-p（nucleus）采样
            if top_p is not None and top_p < 1.0:
                # 按概率降序排列
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=0)

                # 找到累积概率超过 top_p 的截断点
                # 保留截断点之前的 token（及截断点本身）
                cutoff_mask = cumsum_probs - sorted_probs > top_p
                sorted_probs[cutoff_mask] = 0.0
                # 归一化
                sorted_probs = sorted_probs / sorted_probs.sum()

                # 从缩减后的分布采样
                sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices[sampled_idx]
            else:
                # 普通采样
                next_token = torch.multinomial(probs, num_samples=1)

            # 追加新 token
            token_ids = torch.cat([token_ids, next_token.unsqueeze(0)], dim=1)

            # 遇到 EOS 停止
            if eos_id is not None and next_token.item() == eos_id:
                break

    # 解码所有生成的 token
    generated_ids = token_ids[0].tolist()
    return tokenizer.decode(generated_ids)


def train(
    model: nn.Module,
    train_dataset: np.ndarray,
    val_dataset: np.ndarray | None,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    context_length: int,
    max_iters: int,
    eval_interval: int,
    checkpoint_dir: str | None,
    checkpoint_interval: int,
    device: str,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    max_grad_norm: float = 1.0,
    start_iter: int = 0,
    log_file: str | None = None,
) -> None:
    """
    完整的训练循环。

    训练结构（每步）：
    1. 采样批次数据
    2. 设置学习率（余弦调度）
    3. 前向传播计算 loss
    4. 反向传播计算梯度
    5. 梯度裁剪（防止梯度爆炸）
    6. 优化器更新参数
    7. 清零梯度
    8. 定期评估验证集 loss
    9. 定期保存检查点

    Args:
        model: TransformerLM 模型
        train_dataset: 训练 token ID 数组（1D numpy）
        val_dataset: 验证 token ID 数组（可选）
        optimizer: AdamW 优化器
        batch_size: 批次大小
        context_length: 上下文长度
        max_iters: 总训练步数
        eval_interval: 每隔多少步评估一次验证集
        checkpoint_dir: 检查点保存目录（None 则不保存）
        checkpoint_interval: 每隔多少步保存一次检查点
        device: 训练设备
        max_lr: 最大学习率
        min_lr: 最小学习率
        warmup_iters: 预热步数
        max_grad_norm: 梯度裁剪阈值
        start_iter: 从哪步开始（恢复训练时使用）
        log_file: 日志文件路径（None 则不记录）
    """
    from cs336_basics.optimizer import get_lr_cosine_schedule, gradient_clipping

    model.train()
    model.to(device)

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    log_data = []  # 记录训练曲线
    start_time = time.time()

    for it in range(start_iter + 1, max_iters + 1):
        # --- 设置学习率 ---
        lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # --- 采样训练批次 ---
        x, y = get_batch(train_dataset, batch_size, context_length, device)

        # --- 前向传播 ---
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)  # (batch, seq_len, vocab_size)

        # 将 (batch, seq_len, vocab_size) 展平为 (batch*seq_len, vocab_size)
        loss = cross_entropy_loss(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )

        # --- 反向传播 ---
        loss.backward()

        # --- 梯度裁剪 ---
        gradient_clipping(model.parameters(), max_grad_norm)

        # --- 参数更新 ---
        optimizer.step()

        # --- 日志记录 ---
        elapsed = time.time() - start_time
        log_entry = {
            "iter": it,
            "train_loss": loss.item(),
            "lr": lr,
            "elapsed": elapsed,
        }

        # --- 定期验证 ---
        if val_dataset is not None and it % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                # 用多批次估计验证 loss
                val_losses = []
                for _ in range(10):
                    vx, vy = get_batch(val_dataset, batch_size, context_length, device)
                    vlogits = model(vx)
                    vloss = cross_entropy_loss(
                        vlogits.view(-1, vlogits.size(-1)),
                        vy.view(-1),
                    )
                    val_losses.append(vloss.item())
                val_loss = sum(val_losses) / len(val_losses)
                val_ppl = math.exp(val_loss)
            model.train()

            log_entry["val_loss"] = val_loss
            log_entry["val_ppl"] = val_ppl
            print(
                f"Step {it:6d} | lr={lr:.2e} | train_loss={loss.item():.4f} "
                f"| val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f} | t={elapsed:.1f}s"
            )
        elif it % max(1, eval_interval // 10) == 0:
            print(
                f"Step {it:6d} | lr={lr:.2e} | train_loss={loss.item():.4f} | t={elapsed:.1f}s"
            )

        log_data.append(log_entry)

        # --- 定期保存检查点 ---
        if checkpoint_dir and it % checkpoint_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{it:07d}.pt")
            save_checkpoint(model, optimizer, it, ckpt_path)
            print(f"  → Checkpoint saved: {ckpt_path}")

    # 保存训练日志
    if log_file:
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"Training log saved to {log_file}")

    # 保存最终检查点
    if checkpoint_dir:
        final_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
        save_checkpoint(model, optimizer, max_iters, final_path)
        print(f"Final checkpoint saved: {final_path}")
