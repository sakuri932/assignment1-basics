"""
生成 CS336 代码讲解文档所需的所有示意图。
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe

# 统一配色方案
C_BLUE   = "#4A90D9"
C_GREEN  = "#5BAD72"
C_ORANGE = "#E8943A"
C_RED    = "#D95F5F"
C_PURPLE = "#9B72CF"
C_GRAY   = "#8E8E8E"
C_LIGHT  = "#F0F4F8"
C_DARK   = "#2C3E50"
C_TEAL   = "#3AADAD"
C_YELLOW = "#F0C040"

plt.rcParams.update({
    "font.family": ["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.dpi": 150,
})

# ─────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────

def box(ax, x, y, w, h, label, color=C_BLUE, text_color="white",
        fontsize=10, radius=0.04, alpha=1.0, bold=False):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad=0.01,rounding_size={radius}",
                          facecolor=color, edgecolor="white",
                          linewidth=1.2, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, color=text_color, weight=weight, zorder=4)

def arrow(ax, x1, y1, x2, y2, color=C_DARK, lw=1.5, shrink=4):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12,
                                shrinkA=shrink, shrinkB=shrink),
                zorder=5)

def label_arrow(ax, x1, y1, x2, y2, label, color=C_DARK, lw=1.5,
                label_offset=(0, 0.02), fontsize=9):
    arrow(ax, x1, y1, x2, y2, color=color, lw=lw)
    mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
    ax.text(mx, my, label, ha="center", va="bottom", fontsize=fontsize, color=color)

# ─────────────────────────────────────────────────────────
# 图 1：TransformerLM 整体架构
# ─────────────────────────────────────────────────────────

def fig_transformer_lm():
    fig, ax = plt.subplots(figsize=(5.5, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFBFD")

    # 各层 y 坐标（从下到上）
    layers = [
        (0.08, "Token IDs\n(batch, seq_len)", C_GRAY,   "white", 0.30, 0.07),
        (0.20, "Token Embedding\n(vocab_size → d_model)", C_TEAL, "white", 0.52, 0.07),
        (0.35, "Transformer Block × 1",  C_BLUE,   "white", 0.52, 0.07),
        (0.45, "Transformer Block × 2",  C_BLUE,   "white", 0.52, 0.07),
        (0.53, "     ·  ·  ·",           "#AAAAAA", C_DARK,  0.52, 0.06),
        (0.63, "Transformer Block × N",  C_BLUE,   "white", 0.52, 0.07),
        (0.75, "Final RMSNorm",           C_GREEN,  "white", 0.52, 0.07),
        (0.87, "LM Head  Linear\n(d_model → vocab_size)", C_ORANGE, "white", 0.52, 0.07),
        (0.97, "Logits\n(batch, seq_len, vocab_size)", C_GRAY, "white", 0.52, 0.06),
    ]

    for (y, label, color, tc, w, h) in layers:
        box(ax, 0.5, y, w, h, label, color=color, text_color=tc, fontsize=9.5)

    # 箭头
    arrow_ys = [(0.08, 0.20), (0.20, 0.35), (0.35, 0.45),
                (0.45, 0.50), (0.56, 0.63), (0.63, 0.75),
                (0.75, 0.87), (0.87, 0.97)]
    for y1, y2 in arrow_ys:
        arrow(ax, 0.5, y1 + 0.035, 0.5, y2 - 0.035)

    # 右侧标注
    ax.text(0.82, 0.44, "×\nN", ha="center", va="center",
            fontsize=18, color=C_BLUE, weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#EAF3FF", edgecolor=C_BLUE, linewidth=1.5))

    # 维度标注
    dim_notes = [
        (0.08, "(B, L)"),
        (0.20, "(B, L, d_model)"),
        (0.63, "(B, L, d_model)"),
        (0.75, "(B, L, d_model)"),
        (0.87, "(B, L, V)"),
    ]
    for y, note in dim_notes:
        ax.text(0.96, y, note, ha="right", va="center",
                fontsize=7.5, color=C_GRAY, style="italic")

    ax.set_title("TransformerLM 整体架构", fontsize=13, weight="bold",
                 color=C_DARK, pad=8)
    plt.tight_layout()
    plt.savefig("figures/01_transformer_lm.png", bbox_inches="tight",
                facecolor="#FAFBFD")
    plt.close()
    print("✓ 01_transformer_lm.png")


# ─────────────────────────────────────────────────────────
# 图 2：TransformerBlock Pre-norm 结构
# ─────────────────────────────────────────────────────────

def fig_transformer_block():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFBFD")

    cx = 0.5

    # ── 输入 x
    box(ax, cx, 0.05, 0.36, 0.06, "输入  x\n(batch, seq_len, d_model)", C_GRAY, "white", 9)

    # ── 第一个分支 (attention)
    # 残差连接起点：从 x 出发
    ax.annotate("", xy=(0.15, 0.50), xytext=(0.15, 0.05),
                arrowprops=dict(arrowstyle="-", color=C_DARK, lw=1.8), zorder=5)

    arrow(ax, cx, 0.08, cx, 0.17)
    box(ax, cx, 0.21, 0.32, 0.07, "RMSNorm  (ln1)", C_GREEN, "white", 9.5)
    arrow(ax, cx, 0.25, cx, 0.34)
    box(ax, cx, 0.38, 0.45, 0.07, "MultiHeadSelfAttention\n(含 RoPE + 因果掩码)", C_BLUE, "white", 9.5)
    arrow(ax, cx, 0.42, cx, 0.48)

    # 加法节点 ①
    circle1 = plt.Circle((cx, 0.52), 0.028, color=C_ORANGE, zorder=3)
    ax.add_patch(circle1)
    ax.text(cx, 0.52, "+", ha="center", va="center", fontsize=14, color="white", weight="bold", zorder=4)

    # 残差连接 ① 横线到加法节点
    ax.annotate("", xy=(cx - 0.028, 0.52), xytext=(0.15, 0.52),
                arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.8), zorder=5)

    # ── 第二个分支 (FFN)
    arrow(ax, cx, 0.55, cx, 0.62)
    box(ax, cx, 0.66, 0.32, 0.07, "RMSNorm  (ln2)", C_GREEN, "white", 9.5)
    arrow(ax, cx, 0.70, cx, 0.77)
    box(ax, cx, 0.81, 0.36, 0.07, "SwiGLU FFN", C_PURPLE, "white", 9.5)
    arrow(ax, cx, 0.85, cx, 0.91)

    # 加法节点 ②
    circle2 = plt.Circle((cx, 0.94), 0.028, color=C_ORANGE, zorder=3)
    ax.add_patch(circle2)
    ax.text(cx, 0.94, "+", ha="center", va="center", fontsize=14, color="white", weight="bold", zorder=4)

    # 残差连接 ②
    ax.annotate("", xy=(0.82, 0.52), xytext=(0.82, 0.05),
                arrowprops=dict(arrowstyle="-", color=C_DARK, lw=1.8), zorder=5)
    ax.annotate("", xy=(0.82, 0.94), xytext=(0.82, 0.52),
                arrowprops=dict(arrowstyle="-", color=C_DARK, lw=1.8), zorder=5)
    ax.annotate("", xy=(cx + 0.028, 0.94), xytext=(0.82, 0.94),
                arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.8), zorder=5)

    # 输出
    arrow(ax, cx, 0.97, cx, 1.01)
    box(ax, cx, 0.97, 0.36, 0.05, "输出  x_out", C_GRAY, "white", 9)

    # 标注公式
    ax.text(0.01, 0.55, "x' = x + Attn(Norm(x))", fontsize=9, color=C_BLUE,
            style="italic", ha="left")
    ax.text(0.01, 0.93, "out = x' + FFN(Norm(x'))", fontsize=9, color=C_PURPLE,
            style="italic", ha="left")

    ax.set_title("TransformerBlock — Pre-norm 残差结构", fontsize=13, weight="bold",
                 color=C_DARK, pad=8)
    plt.tight_layout()
    plt.savefig("figures/02_transformer_block.png", bbox_inches="tight",
                facecolor="#FAFBFD")
    plt.close()
    print("✓ 02_transformer_block.png")


# ─────────────────────────────────────────────────────────
# 图 3：Multi-Head Self-Attention
# ─────────────────────────────────────────────────────────

def fig_mhsa():
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFBFD")

    # 输入
    box(ax, 0.5, 0.05, 0.40, 0.06, "输入 x   (batch, seq_len, d_model)", C_GRAY, "white", 9)

    # 三个投影
    projs = [(0.18, "Q 投影\n(d_model→d_model)", C_BLUE),
             (0.50, "K 投影\n(d_model→d_model)", C_BLUE),
             (0.82, "V 投影\n(d_model→d_model)", C_BLUE)]
    for px, lbl, col in projs:
        arrow(ax, 0.5, 0.08, px, 0.19)
        box(ax, px, 0.23, 0.26, 0.07, lbl, col, "white", 9)

    # 拆分为多头
    for px, _, _ in projs:
        arrow(ax, px, 0.27, px, 0.35)
        box(ax, px, 0.38, 0.26, 0.06, "reshape → (B,H,L,d_k)", "#6699CC", "white", 8.5)

    # RoPE 只用于 Q 和 K
    for px in [0.18, 0.50]:
        arrow(ax, px, 0.41, px, 0.48)
        box(ax, px, 0.51, 0.24, 0.06, "RoPE 旋转", C_TEAL, "white", 8.5)

    # Attention 计算
    arrow(ax, 0.18, 0.54, 0.36, 0.63)
    arrow(ax, 0.50, 0.54, 0.42, 0.63)
    arrow(ax, 0.82, 0.41, 0.62, 0.63)
    box(ax, 0.5, 0.67, 0.50, 0.07,
        "Scaled Dot-Product Attention\nsoftmax(QKᵀ/√d_k + causal_mask) · V",
        C_RED, "white", 8.5)

    # 合并头
    arrow(ax, 0.5, 0.71, 0.5, 0.78)
    box(ax, 0.5, 0.81, 0.40, 0.06, "concat heads → (B, L, d_model)", "#6699CC", "white", 8.5)

    # 输出投影
    arrow(ax, 0.5, 0.84, 0.5, 0.90)
    box(ax, 0.5, 0.93, 0.36, 0.06, "输出投影  W_O", C_BLUE, "white", 9)
    arrow(ax, 0.5, 0.96, 0.5, 1.00)

    # 右侧说明
    ax.text(0.97, 0.51, "Q, K 施加 RoPE\nV 不旋转", ha="right", va="center",
            fontsize=8.5, color=C_TEAL,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E0F5F5", edgecolor=C_TEAL))
    ax.text(0.97, 0.67, "因果掩码\n(下三角 = True)", ha="right", va="center",
            fontsize=8.5, color=C_RED,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF0F0", edgecolor=C_RED))

    ax.set_title("Multi-Head Self-Attention（含 RoPE）", fontsize=13, weight="bold",
                 color=C_DARK, pad=8)
    plt.tight_layout()
    plt.savefig("figures/03_mhsa.png", bbox_inches="tight", facecolor="#FAFBFD")
    plt.close()
    print("✓ 03_mhsa.png")


# ─────────────────────────────────────────────────────────
# 图 4：SwiGLU 门控前馈
# ─────────────────────────────────────────────────────────

def fig_swiglu():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFBFD")

    # 输入
    box(ax, 0.5, 0.08, 0.36, 0.07, "输入 x   (…, d_model)", C_GRAY, "white", 10)

    # 分叉 → W1 路径（gate）和 W3 路径（up）
    arrow(ax, 0.5, 0.12, 0.28, 0.28)
    arrow(ax, 0.5, 0.12, 0.72, 0.28)

    # W1 + SiLU
    box(ax, 0.22, 0.33, 0.24, 0.08, "W₁  (d_model→d_ff)", C_BLUE, "white", 9)
    arrow(ax, 0.22, 0.37, 0.22, 0.48)
    box(ax, 0.22, 0.53, 0.22, 0.08, "SiLU(·)\n门控值 gate", C_TEAL, "white", 9)

    # W3
    box(ax, 0.78, 0.33, 0.24, 0.08, "W₃  (d_model→d_ff)", C_BLUE, "white", 9)
    arrow(ax, 0.78, 0.37, 0.78, 0.48)
    box(ax, 0.78, 0.53, 0.22, 0.08, "up 向量\n（被门控内容）", C_PURPLE, "white", 9)

    # 乘法节点
    arrow(ax, 0.22, 0.57, 0.44, 0.69)
    arrow(ax, 0.78, 0.57, 0.56, 0.69)
    circle = plt.Circle((0.5, 0.72), 0.032, color=C_ORANGE, zorder=3)
    ax.add_patch(circle)
    ax.text(0.5, 0.72, "⊙", ha="center", va="center",
            fontsize=15, color="white", weight="bold", zorder=4)
    ax.text(0.5, 0.72 - 0.06, "逐元素乘法", ha="center", va="top",
            fontsize=8.5, color=C_ORANGE)

    # W2 下投影
    arrow(ax, 0.5, 0.75, 0.5, 0.84)
    box(ax, 0.5, 0.88, 0.26, 0.08, "W₂  (d_ff→d_model)", C_BLUE, "white", 9)
    arrow(ax, 0.5, 0.92, 0.5, 0.98)
    box(ax, 0.5, 0.98, 0.32, 0.04, "输出 (…, d_model)", C_GRAY, "white", 9)

    # 注释
    note = ("d_ff ≈ 8/3 × d_model\n"
            "（保持与普通 FFN 参数量相当）")
    ax.text(0.98, 0.40, note, ha="right", va="center", fontsize=8.5, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF8E8", edgecolor=C_ORANGE))

    ax.set_title("SwiGLU 门控前馈网络", fontsize=13, weight="bold", color=C_DARK, pad=8)
    plt.tight_layout()
    plt.savefig("figures/04_swiglu.png", bbox_inches="tight", facecolor="#FAFBFD")
    plt.close()
    print("✓ 04_swiglu.png")


# ─────────────────────────────────────────────────────────
# 图 5：RoPE 旋转示意
# ─────────────────────────────────────────────────────────

def fig_rope():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("#FAFBFD")

    # ── 左图：一对维度的旋转
    ax = axes[0]
    ax.set_aspect("equal")
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.set_facecolor("#F7F9FC")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="#CCCCCC", lw=0.8)
    ax.axvline(0, color="#CCCCCC", lw=0.8)
    ax.set_xlabel("维度 2k", fontsize=10)
    ax.set_ylabel("维度 2k+1", fontsize=10)

    # 单位圆
    theta_range = np.linspace(0, 2 * math.pi, 300)
    ax.plot(np.cos(theta_range), np.sin(theta_range), color="#DDDDDD", lw=1)

    # 原始向量 (position 0)
    v0 = np.array([1.2, 0.4])
    ax.annotate("", xy=v0, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=2.2, mutation_scale=14))
    ax.text(v0[0] + 0.08, v0[1] + 0.08, "pos=0\n(无旋转)", fontsize=9, color=C_BLUE)

    # pos=2 的旋转
    theta2 = 0.6
    cos2, sin2 = math.cos(theta2), math.sin(theta2)
    v2 = np.array([v0[0]*cos2 - v0[1]*sin2, v0[0]*sin2 + v0[1]*cos2])
    ax.annotate("", xy=v2, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=2.2, mutation_scale=14))
    ax.text(v2[0] + 0.08, v2[1], "pos=2\n旋转 2θ", fontsize=9, color=C_GREEN)

    # pos=5 的旋转
    theta5 = 1.5
    cos5, sin5 = math.cos(theta5), math.sin(theta5)
    v5 = np.array([v0[0]*cos5 - v0[1]*sin5, v0[0]*sin5 + v0[1]*cos5])
    ax.annotate("", xy=v5, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, lw=2.2, mutation_scale=14))
    ax.text(v5[0] - 0.25, v5[1] + 0.08, "pos=5\n旋转 5θ", fontsize=9, color=C_ORANGE)

    # 弧形角度标注
    arc_t = np.linspace(0, theta2, 30)
    ax.plot(0.5*np.cos(arc_t), 0.5*np.sin(arc_t), color=C_GREEN, lw=1.2)
    ax.text(0.6, 0.12, "θ", fontsize=9, color=C_GREEN)

    ax.set_title("一对维度上的旋转\n（不同位置旋转不同角度）", fontsize=10, color=C_DARK)

    # ── 右图：频率随维度变化
    ax2 = axes[1]
    ax2.set_facecolor("#F7F9FC")
    ax2.spines[["top", "right"]].set_visible(False)

    d_k = 64
    theta_base = 10000.0
    dim_idx = np.arange(0, d_k, 2)
    inv_freqs = 1.0 / (theta_base ** (dim_idx / d_k))

    ax2.plot(dim_idx, inv_freqs, color=C_BLUE, lw=2.5, marker="o", markersize=4)
    ax2.fill_between(dim_idx, inv_freqs, alpha=0.15, color=C_BLUE)
    ax2.set_xlabel("维度对序号  2k", fontsize=10)
    ax2.set_ylabel("旋转频率  θ⁻¹", fontsize=10)
    ax2.set_title("RoPE 频率随维度的变化\n（高维度 = 低频 = 慢变化）", fontsize=10, color=C_DARK)
    ax2.set_yscale("log")
    ax2.grid(axis="y", alpha=0.3)

    # 标注高频和低频
    ax2.text(2, inv_freqs[0]*1.5, "高频\n（捕捉局部位置）", fontsize=8.5,
             color=C_ORANGE, ha="left")
    ax2.text(dim_idx[-4], inv_freqs[-1]*0.4, "低频\n（捕捉全局位置）", fontsize=8.5,
             color=C_TEAL, ha="right")

    fig.suptitle("旋转位置编码 (RoPE)", fontsize=13, weight="bold", color=C_DARK, y=1.01)
    plt.tight_layout()
    plt.savefig("figures/05_rope.png", bbox_inches="tight", facecolor="#FAFBFD")
    plt.close()
    print("✓ 05_rope.png")


# ─────────────────────────────────────────────────────────
# 图 6：BPE 算法流程
# ─────────────────────────────────────────────────────────

def fig_bpe():
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFBFD")

    # 顶部标题行（示例数据）
    ax.text(0.5, 0.97, "BPE 训练示例：合并一步",
            ha="center", va="top", fontsize=12, weight="bold", color=C_DARK)

    # ── 步骤 1：初始词表
    ax.text(0.03, 0.90, "① 初始词表 (256 字节 token)", fontsize=10.5,
            weight="bold", color=C_TEAL)
    ax.text(0.03, 0.84,
            'vocab = {0: b"\\x00", 1: b"\\x01", ..., 97: b"a", 98: b"b", ..., 255: b"\\xff"}',
            fontsize=9, color=C_DARK, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#EAF9F4", edgecolor=C_TEAL))

    # ── 步骤 2：预分词计数
    ax.text(0.03, 0.76, "② 预分词频次统计 (pre-token counts)", fontsize=10.5,
            weight="bold", color=C_BLUE)
    ptab = [
        ('(b"h", b"e", b"l", b"l", b"o")',  "5"),
        ('(b"l", b"o", b"w", b"e", b"r")',  "3"),
        ('(b"l", b"o", b"w")',               "7"),
    ]
    for i, (word, cnt) in enumerate(ptab):
        y = 0.70 - i * 0.06
        ax.text(0.06, y, f"{word}  →  {cnt} 次", fontsize=9,
                color=C_DARK, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEF4FF", edgecolor="#AACCFF"))

    # ── 步骤 3：统计对频次
    ax.text(0.03, 0.51, "③ 统计相邻 token 对频次 (pair_counts)", fontsize=10.5,
            weight="bold", color=C_PURPLE)
    pairs = [
        ('(b"l", b"o")',   "10",  True),
        ('(b"h", b"e")',   "5",   False),
        ('(b"o", b"w")',   "10",  False),
        ('(b"e", b"l")',   "5",   False),
    ]
    for i, (pair, cnt, is_best) in enumerate(pairs):
        y = 0.45 - i * 0.055
        color = C_RED if is_best else C_DARK
        bg = "#FFE8E8" if is_best else "#F5EEFF"
        ec = C_RED if is_best else "#BBAADD"
        txt = f"{pair}  →  {cnt}"
        if is_best:
            txt += "  ← 最高频！"
        ax.text(0.06, y, txt, fontsize=9, color=color, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=bg, edgecolor=ec))

    # 箭头 → 合并
    ax.annotate("", xy=(0.56, 0.28), xytext=(0.56, 0.43),
                arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=2.5, mutation_scale=15))
    ax.text(0.58, 0.37, '合并\n(b"l", b"o")\n→ b"lo"',
            ha="left", va="center", fontsize=9, color=C_RED, weight="bold")

    # ── 步骤 4：更新后结果
    ax.text(0.03, 0.23, "④ 合并后 — 增量更新词表与对频次", fontsize=10.5,
            weight="bold", color=C_GREEN)
    after = [
        ('(b"h", b"e", b"ll", b"o")',  "—  已合并 lo → 实际是 (h,e,l,lo)"),
        ('(b"lo", b"w", b"e", b"r")',   "—  lo 变为单个 token"),
        ('(b"lo", b"w")',                "—  lo 变为单个 token"),
    ]
    for i, (word, note) in enumerate(after):
        y = 0.17 - i * 0.055
        ax.text(0.06, y, f"{word}", fontsize=9, color=C_DARK, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#EDFAED", edgecolor=C_GREEN))
        ax.text(0.55, y, note, fontsize=8.5, color=C_GRAY, va="center")

    ax.set_title("BPE 训练流程示意（单次合并步骤）", fontsize=13, weight="bold",
                 color=C_DARK, pad=8)
    plt.tight_layout()
    plt.savefig("figures/06_bpe.png", bbox_inches="tight", facecolor="#FAFBFD")
    plt.close()
    print("✓ 06_bpe.png")


# ─────────────────────────────────────────────────────────
# 图 7：AdamW 更新步骤
# ─────────────────────────────────────────────────────────

def fig_adamw():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor("#FAFBFD")

    # ── 左图：更新流程
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    steps = [
        (0.90, "计算梯度  g = ∇L(θ)",         C_GRAY),
        (0.76, "权重衰减\nθ ← θ × (1 - lr·λ)",  C_RED),
        (0.62, "更新一阶矩\nm ← β₁m + (1-β₁)g", C_BLUE),
        (0.48, "更新二阶矩\nv ← β₂v + (1-β₂)g²", C_PURPLE),
        (0.34, "偏差修正学习率\nα_t = lr×√(1-β₂ᵗ)/(1-β₁ᵗ)", C_TEAL),
        (0.18, "参数更新\nθ ← θ - α_t × m/(√v+ε)",  C_ORANGE),
    ]

    for y, lbl, col in steps:
        box(ax, 0.5, y, 0.80, 0.10, lbl, col, "white", 9.5)
        if y > 0.18:
            arrow(ax, 0.5, y - 0.05, 0.5, y - 0.09, lw=1.5)

    # 标注 AdamW 特有步骤
    ax.text(0.97, 0.76, "AdamW\n特有！", ha="right", va="center",
            fontsize=8.5, color=C_RED, weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFE8E8", edgecolor=C_RED))

    ax.set_title("AdamW 参数更新流程", fontsize=12, weight="bold", color=C_DARK)

    # ── 右图：自适应学习率直觉
    ax2 = axes[1]
    ax2.set_facecolor("#F7F9FC")
    ax2.spines[["top", "right"]].set_visible(False)

    np.random.seed(42)
    steps_n = 200
    t = np.arange(1, steps_n + 1)

    # 模拟两个参数：高梯度方差 vs 低梯度方差
    g_high = np.random.randn(steps_n) * 2.0   # 高噪声参数
    g_low  = np.sin(np.linspace(0, 3, steps_n)) * 0.5 + 0.1  # 低噪声参数

    beta2 = 0.999
    eps = 1e-8
    v_high = np.zeros(steps_n)
    v_low  = np.zeros(steps_n)
    eff_lr_high = np.zeros(steps_n)
    eff_lr_low  = np.zeros(steps_n)

    vh, vl = 0.0, 0.0
    for i in range(steps_n):
        vh = beta2 * vh + (1 - beta2) * g_high[i] ** 2
        vl = beta2 * vl + (1 - beta2) * g_low[i]  ** 2
        bc = 1 - beta2 ** (i + 1)
        eff_lr_high[i] = 1.0 / (math.sqrt(vh / bc) + eps)
        eff_lr_low[i]  = 1.0 / (math.sqrt(vl / bc) + eps)

    ax2.plot(t, eff_lr_high / eff_lr_high.max(), color=C_RED,    lw=2, label="高梯度方差参数（有效 lr 小）")
    ax2.plot(t, eff_lr_low  / eff_lr_low.max(),  color=C_GREEN,  lw=2, label="低梯度方差参数（有效 lr 大）")
    ax2.set_xlabel("训练步数", fontsize=10)
    ax2.set_ylabel("归一化有效学习率", fontsize=10)
    ax2.set_title("Adam 自适应学习率直觉\n梯度噪声大 → 有效 lr 小", fontsize=10, color=C_DARK)
    ax2.legend(fontsize=8.5)
    ax2.grid(alpha=0.3)

    fig.suptitle("AdamW 优化器", fontsize=13, weight="bold", color=C_DARK)
    plt.tight_layout()
    plt.savefig("figures/07_adamw.png", bbox_inches="tight", facecolor="#FAFBFD")
    plt.close()
    print("✓ 07_adamw.png")


# ─────────────────────────────────────────────────────────
# 图 8：余弦学习率调度
# ─────────────────────────────────────────────────────────

def fig_lr_schedule():
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#FAFBFD")
    ax.set_facecolor("#F7F9FC")
    ax.spines[["top", "right"]].set_visible(False)

    max_lr = 3e-4
    min_lr = 3e-5
    warmup = 200
    total  = 2000

    t = np.arange(0, total + 1)
    lr = np.zeros_like(t, dtype=float)
    for i in t:
        if i < warmup:
            lr[i] = max_lr * i / warmup
        elif i <= total:
            progress = (i - warmup) / (total - warmup)
            lr[i] = min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)
        else:
            lr[i] = min_lr

    ax.plot(t, lr * 1e4, color=C_BLUE, lw=2.5)
    ax.fill_between(t, lr * 1e4, alpha=0.12, color=C_BLUE)

    # 区域背景
    ax.axvspan(0,       warmup, alpha=0.08, color=C_GREEN)
    ax.axvspan(warmup,  total,  alpha=0.08, color=C_ORANGE)

    # 标注
    ax.axvline(warmup, color=C_GREEN,  ls="--", lw=1.5, alpha=0.7)
    ax.axhline(min_lr*1e4, color=C_GRAY, ls=":", lw=1.2)

    ax.text(warmup/2, max_lr*1e4*0.55, "① 线性预热\n(Linear Warmup)",
            ha="center", fontsize=9, color=C_GREEN, weight="bold")
    ax.text((warmup + total)/2, max_lr*1e4*0.65, "② 余弦退火\n(Cosine Annealing)",
            ha="center", fontsize=9, color=C_ORANGE, weight="bold")
    ax.text(total*0.97, min_lr*1e4*1.8, "③ 最小 lr 保持",
            ha="right", fontsize=8.5, color=C_GRAY)

    ax.set_xlabel("训练步数 (iter)", fontsize=10)
    ax.set_ylabel("学习率  (×10⁻⁴)", fontsize=10)
    ax.set_title("带线性预热的余弦学习率调度", fontsize=13, weight="bold", color=C_DARK)

    ax.set_xlim(0, total)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/08_lr_schedule.png", bbox_inches="tight", facecolor="#FAFBFD")
    plt.close()
    print("✓ 08_lr_schedule.png")


# ─────────────────────────────────────────────────────────
# 图 9：Scaled Dot-Product Attention 细节
# ─────────────────────────────────────────────────────────

def fig_attention_detail():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor("#FAFBFD")

    # ── 左图：Attention 分数矩阵 heatmap（带因果掩码）
    ax = axes[0]
    seq_len = 6
    np.random.seed(7)
    raw_scores = np.random.randn(seq_len, seq_len) * 1.5

    # 应用因果掩码：上三角 = -inf
    causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    masked_scores = raw_scores.copy()
    masked_scores[~causal_mask] = -np.inf

    # 稳定 softmax
    row_max = np.nanmax(masked_scores, axis=1, keepdims=True)
    exp_s = np.where(causal_mask, np.exp(masked_scores - row_max), 0)
    attn_weights = exp_s / (exp_s.sum(axis=1, keepdims=True) + 1e-9)

    im = ax.imshow(attn_weights, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tokens = ["The", "cat", "sat", "on", "a", "mat"]
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tokens, fontsize=9)
    ax.set_xlabel("Key 位置（被关注的 token）", fontsize=9)
    ax.set_ylabel("Query 位置（当前 token）", fontsize=9)
    ax.set_title("Attention 权重矩阵\n（因果掩码：只关注过去 token）", fontsize=10, color=C_DARK)

    # 在对角线上标注 "无法看到未来"
    for i in range(seq_len):
        for j in range(seq_len):
            if not causal_mask[i, j]:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                           fill=True, facecolor="#FFCCCC", alpha=0.6))
    ax.text(seq_len*0.72, seq_len*0.22, "掩蔽区域\n(masked)", fontsize=8.5,
            color=C_RED, ha="center", weight="bold")

    # ── 右图：数值稳定 softmax 对比
    ax2 = axes[1]
    ax2.set_facecolor("#F7F9FC")
    ax2.spines[["top", "right"]].set_visible(False)

    x_vals = np.array([100.0, 102.0, 98.0, 99.0, 101.0])
    labels = [f"x={v:.0f}" for v in x_vals]

    # 朴素 softmax（溢出）
    try:
        naive = np.exp(x_vals) / np.exp(x_vals).sum()
    except Exception:
        naive = np.full(len(x_vals), np.nan)

    # 稳定 softmax
    x_shifted = x_vals - x_vals.max()
    stable = np.exp(x_shifted) / np.exp(x_shifted).sum()

    xpos = np.arange(len(x_vals))
    ax2.bar(xpos - 0.2, stable, 0.38, color=C_BLUE,   alpha=0.85, label="数值稳定 softmax（正确）")
    ax2.bar(xpos + 0.2, stable, 0.38, color="#EEEEEE", alpha=0.6,  label="朴素 softmax（溢出→NaN）",
            hatch="///", edgecolor=C_RED, linewidth=1)

    # 在朴素柱上标 "NaN/inf"
    for xi in xpos:
        ax2.text(xi + 0.2, stable[xi] + 0.01, "溢出!", ha="center",
                 fontsize=8, color=C_RED, weight="bold")

    ax2.set_xticks(xpos)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("softmax 输出值", fontsize=10)
    ax2.set_title("数值稳定 softmax\n（减去 max 后再计算 exp）", fontsize=10, color=C_DARK)
    ax2.legend(fontsize=8.5)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Scaled Dot-Product Attention 细节", fontsize=13,
                 weight="bold", color=C_DARK)
    plt.tight_layout()
    plt.savefig("figures/09_attention_detail.png", bbox_inches="tight",
                facecolor="#FAFBFD")
    plt.close()
    print("✓ 09_attention_detail.png")


# ─────────────────────────────────────────────────────────
# 图 10：RMSNorm vs LayerNorm
# ─────────────────────────────────────────────────────────

def fig_rmsnorm():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("#FAFBFD")

    # ── 左图：归一化公式对比（文字图）
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "LayerNorm vs RMSNorm", ha="center", va="top",
            fontsize=12, weight="bold", color=C_DARK)

    # LayerNorm
    box(ax, 0.5, 0.78, 0.88, 0.22, "", "#F0F4FF", C_DARK, 9.5, radius=0.05)
    ax.text(0.5, 0.88, "LayerNorm", ha="center", va="center",
            fontsize=11, weight="bold", color=C_BLUE)
    ax.text(0.5, 0.79,
            "μ = mean(x)    σ = std(x)\n"
            "LN(x) = (x − μ) / (σ + ε) × g + b\n"
            "需要计算均值 + 方差 + 偏置",
            ha="center", va="center", fontsize=9, color=C_DARK, family="monospace")

    # RMSNorm
    box(ax, 0.5, 0.48, 0.88, 0.22, "", "#F0FFF4", C_DARK, 9.5, radius=0.05)
    ax.text(0.5, 0.58, "RMSNorm", ha="center", va="center",
            fontsize=11, weight="bold", color=C_GREEN)
    ax.text(0.5, 0.49,
            "RMS = sqrt(mean(x²) + ε)\n"
            "RN(x) = x / RMS × g\n"
            "省去均值计算，无偏置项",
            ha="center", va="center", fontsize=9, color=C_DARK, family="monospace")

    # 对比
    box(ax, 0.5, 0.22, 0.88, 0.16, "", "#FFFAF0", C_DARK, 9.5, radius=0.05)
    ax.text(0.5, 0.27, "✓ RMSNorm：计算量更小，效果相当\n"
                        "✓ LLaMA / GPT-3 等均采用 RMSNorm",
            ha="center", va="center", fontsize=9, color=C_ORANGE)

    # ── 右图：归一化前后的分布
    ax2 = axes[1]
    ax2.set_facecolor("#F7F9FC")
    ax2.spines[["top", "right"]].set_visible(False)

    np.random.seed(10)
    x_raw = np.random.randn(1000) * 5 + 3
    rms_val = np.sqrt(np.mean(x_raw ** 2) + 1e-5)
    x_rms = x_raw / rms_val

    mean_val = x_raw.mean()
    std_val  = x_raw.std()
    x_ln  = (x_raw - mean_val) / (std_val + 1e-5)

    bins = np.linspace(-4, 12, 60)
    ax2.hist(x_raw, bins=bins, color=C_GRAY,   alpha=0.6, label="原始输入",   density=True)
    bins2 = np.linspace(-4, 4, 60)
    ax2.hist(x_rms, bins=bins2, color=C_GREEN, alpha=0.7, label="RMSNorm 后", density=True)
    ax2.hist(x_ln,  bins=bins2, color=C_BLUE,  alpha=0.5, label="LayerNorm 后", density=True,
             linestyle="--", histtype="step", linewidth=2)

    ax2.axvline(0, color=C_DARK, ls="--", lw=1)
    ax2.set_xlabel("激活值", fontsize=10)
    ax2.set_ylabel("概率密度", fontsize=10)
    ax2.set_title("归一化前后的分布变化", fontsize=10, color=C_DARK)
    ax2.legend(fontsize=8.5)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("RMSNorm：比 LayerNorm 更简洁的归一化", fontsize=13,
                 weight="bold", color=C_DARK)
    plt.tight_layout()
    plt.savefig("figures/10_rmsnorm.png", bbox_inches="tight", facecolor="#FAFBFD")
    plt.close()
    print("✓ 10_rmsnorm.png")


# ─────────────────────────────────────────────────────────
# 图 11：训练循环流程图
# ─────────────────────────────────────────────────────────

def fig_training_loop():
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFBFD")

    steps = [
        (0.93, "开始训练\nfor it = 1 … max_iters",         C_DARK,   True),
        (0.82, "① 设置学习率\nLR ← cosine_schedule(it)",   C_TEAL,   False),
        (0.72, "② 采样批次\nx, y ← get_batch(dataset)",    C_BLUE,   False),
        (0.61, "③ 前向传播\nlogits ← model(x)",            C_BLUE,   False),
        (0.50, "④ 计算损失\nloss ← cross_entropy(logits, y)", C_RED, False),
        (0.39, "⑤ 反向传播\nloss.backward()",              C_PURPLE, False),
        (0.28, "⑥ 梯度裁剪\nclip(grad, max_norm=1.0)",     C_ORANGE, False),
        (0.17, "⑦ 参数更新\noptimizer.step()",             C_GREEN,  False),
        (0.06, "定期：验证 & 保存检查点",                   C_GRAY,   False),
    ]

    for y, lbl, col, is_start in steps:
        w = 0.70 if not is_start else 0.58
        box(ax, 0.5, y, w, 0.07, lbl, col, "white", 9.5, bold=is_start)

    # 箭头
    for i in range(len(steps) - 1):
        y1, y2 = steps[i][0], steps[i+1][0]
        arrow(ax, 0.5, y1 - 0.035, 0.5, y2 + 0.035)

    # 循环回路（右侧回箭头）
    ax.annotate("", xy=(0.84, 0.85), xytext=(0.84, 0.10),
                arrowprops=dict(arrowstyle="-", color=C_DARK, lw=1.5, linestyle="dashed"))
    ax.annotate("", xy=(0.65, 0.93), xytext=(0.84, 0.93),
                arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.5, mutation_scale=12))
    ax.text(0.91, 0.50, "下一\n个 it", ha="center", va="center",
            fontsize=8.5, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0", edgecolor=C_GRAY))

    ax.set_title("完整训练循环", fontsize=13, weight="bold", color=C_DARK, pad=8)
    plt.tight_layout()
    plt.savefig("figures/11_training_loop.png", bbox_inches="tight",
                facecolor="#FAFBFD")
    plt.close()
    print("✓ 11_training_loop.png")


# ─────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig_transformer_lm()
    fig_transformer_block()
    fig_mhsa()
    fig_swiglu()
    fig_rope()
    fig_bpe()
    fig_adamw()
    fig_lr_schedule()
    fig_attention_detail()
    fig_rmsnorm()
    fig_training_loop()
    print("\n全部图片已生成到 figures/ 目录。")
