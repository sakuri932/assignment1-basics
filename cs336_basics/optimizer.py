"""
AdamW 优化器、余弦学习率调度和梯度裁剪。

本文件实现：
- AdamW: 带解耦权重衰减的 Adam 优化器
- get_lr_cosine_schedule: 带线性预热的余弦退火学习率调度
- gradient_clipping: 全局梯度 L2 范数裁剪

References:
- Kingma & Ba, 2015: Adam: A Method for Stochastic Optimization
- Loshchilov & Hutter, 2019: Decoupled Weight Decay Regularization (AdamW)
- Touvron et al., 2023: LLaMA - 余弦学习率调度
"""

import math
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    AdamW 优化器（带解耦权重衰减的 Adam）。

    AdamW 是现代语言模型训练的标准优化器（GPT-3、LLaMA 等均使用）。

    Adam 的核心思想：
    - 维护梯度的一阶矩（动量）和二阶矩（自适应学习率）
    - 对每个参数使用不同的有效学习率
    - 梯度变化大的参数：有效 lr 小（保守更新）
    - 梯度变化小的参数：有效 lr 大（积极更新）

    AdamW vs Adam（权重衰减处理方式不同）：
    - Adam：权重衰减通过 L2 正则化添加到梯度中（与 Adam 的自适应缩放耦合）
    - AdamW：权重衰减直接作用于参数（解耦），不受梯度自适应缩放的影响
    - 解耦的好处：权重衰减的效果更可预测，不会被自适应缩放"弱化"

    算法（来自 Loshchilov & Hutter 2019, Algorithm 2）：
    对每步 t = 1, 2, ...:
      g_t  = ∇θ L(θ_t; B_t)          # 计算梯度
      α_t  = α * √(1-β₂ᵗ) / (1-β₁ᵗ) # 偏差修正学习率
      θ_t  = θ_{t-1} - α * λ * θ_{t-1}  # 权重衰减（解耦）
      m_t  = β₁ * m_{t-1} + (1-β₁) * g_t  # 更新一阶矩
      v_t  = β₂ * v_{t-1} + (1-β₂) * g_t²  # 更新二阶矩
      θ_t  = θ_t - α_t * m_t / (√v_t + ε)   # 参数更新

    偏差修正（bias correction）：
    初始时 m=0, v=0，会导致初始估计偏向 0。
    偏差修正因子 1/(1-βᵗ) 补偿这种初始偏置，使早期更新更稳定。

    典型超参数：
    - lr = 1e-3 to 3e-4（语言模型）
    - betas = (0.9, 0.999) 或 (0.9, 0.95)（LLaMA 使用 0.95）
    - eps = 1e-8（数值稳定，防止除以 0）
    - weight_decay = 0.01 到 0.1
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """
        Args:
            params: 待优化参数或参数组
            lr: 基础学习率 α
            betas: (β₁, β₂)，一阶和二阶矩估计的指数衰减率
            eps: 数值稳定常数 ε，防止除以零
            weight_decay: 权重衰减系数 λ
        """
        if lr < 0.0:
            raise ValueError(f"无效的学习率: {lr}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"无效的 β₁: {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"无效的 β₂: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"无效的 ε: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"无效的 weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        执行一步参数更新。

        在 loss.backward() 之后调用，使用当前梯度更新参数。

        Args:
            closure: 可选的闭包函数，用于重新计算 loss（满足 PyTorch Optimizer API）

        Returns:
            loss 值（如果 closure 不为 None）
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue  # 跳过没有梯度的参数（如冻结参数）

                g = p.grad.data  # 当前梯度
                state = self.state[p]  # 该参数的状态字典

                # 首次调用时初始化状态
                if len(state) == 0:
                    state["step"] = 0
                    # 一阶矩（动量），初始化为 0
                    state["m"] = torch.zeros_like(p.data)
                    # 二阶矩（自适应学习率），初始化为 0
                    state["v"] = torch.zeros_like(p.data)

                state["step"] += 1
                t = state["step"]
                m = state["m"]
                v = state["v"]

                # 偏差修正后的学习率: α_t = α * √(1-β₂ᵗ) / (1-β₁ᵗ)
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                # 解耦权重衰减：直接对参数施加 L2 正则化
                # 注意：这在梯度更新之前进行，与 Adam + L2 不同
                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # 更新一阶矩估计（指数加权移动平均梯度）
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)

                # 更新二阶矩估计（指数加权移动平均梯度平方）
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # 参数更新: θ = θ - α_t * m / (√v + ε)
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    带线性预热的余弦退火学习率调度。

    这是训练 LLaMA 等模型时使用的标准学习率调度策略。

    三个阶段：
    1. 线性预热（0 ≤ t < T_w）：
       α_t = t/T_w * α_max
       从 0 线性增加到 α_max，避免初期大梯度不稳定。

    2. 余弦退火（T_w ≤ t ≤ T_c）：
       α_t = α_min + ½ * (1 + cos(π * (t-T_w)/(T_c-T_w))) * (α_max - α_min)
       平滑地从 α_max 下降到 α_min，学习率曲线呈余弦形。

    3. 后退火（t > T_c）：
       α_t = α_min
       保持最小学习率（用于微调或继续训练）。

    余弦退火的优势：
    - 相比线性衰减，余弦退火在早期学习率下降较慢（可以充分学习），
      在后期下降较快（精细收敛），通常取得更低的最终 loss。

    Args:
        it: 当前迭代步数（从 0 开始还是 1 开始取决于调用者，这里用 it=1 对应第一步）
        max_learning_rate: 最大学习率 α_max（预热结束时）
        min_learning_rate: 最小学习率 α_min（退火结束后保持）
        warmup_iters: 预热步数 T_w
        cosine_cycle_iters: 余弦周期总步数 T_c（包括预热阶段）

    Returns:
        当前迭代步的学习率
    """
    if it < warmup_iters:
        # 线性预热阶段
        return max_learning_rate * it / warmup_iters
    elif it <= cosine_cycle_iters:
        # 余弦退火阶段
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1.0 + math.cos(math.pi * progress)) * (
            max_learning_rate - min_learning_rate
        )
    else:
        # 后退火阶段：保持最小学习率
        return min_learning_rate


def gradient_clipping(
    parameters,
    max_l2_norm: float,
) -> None:
    """
    全局梯度 L2 范数裁剪。

    梯度裁剪的作用：
    当遇到特别大的梯度时（如 loss spike），直接更新参数可能导致训练不稳定。
    梯度裁剪将所有参数的梯度视为一个整体向量，如果其 L2 范数超过阈值 M，
    则按比例缩放所有梯度，使总范数恰好等于 M（实际上会稍小于 M，因为加了 ε）。

    公式：
    - global_norm = √(Σ ||g_i||²)
    - 如果 global_norm > M，则 g_i ← g_i * M / (global_norm + ε)

    注意：ε = 1e-6（PyTorch 默认值）用于防止除以 0。

    Args:
        parameters: 参数的可迭代对象（torch.nn.Parameter）
        max_l2_norm: 最大允许的 L2 范数 M（正数）
    """
    eps = 1e-6  # 数值稳定常数（PyTorch 默认值）

    # 只对有梯度的参数进行裁剪
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return

    # 计算全局 L2 范数（所有参数梯度的联合范数）
    total_norm_sq = sum(p.grad.data.norm(2).item() ** 2 for p in params_with_grad)
    total_norm = math.sqrt(total_norm_sq)

    # 只有超过阈值时才裁剪
    if total_norm > max_l2_norm:
        # 缩放因子：让总范数缩小到 max_l2_norm（略小于，因为 +eps）
        scale = max_l2_norm / (total_norm + eps)
        for p in params_with_grad:
            p.grad.data.mul_(scale)
