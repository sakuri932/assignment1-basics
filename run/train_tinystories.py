"""
TinyStories GPT 训练启动脚本

用法：
    python run/train_tinystories.py              # 正常训练（含 wandb）
    python run/train_tinystories.py --no-wandb   # 禁用 wandb（离线/调试用）
    python run/train_tinystories.py --resume run/checkpoints/checkpoint_0010000.pt
"""

import sys
import argparse
from pathlib import Path

# 将项目根目录加入 Python 路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.training import train, load_checkpoint

# ─── 超参数配置 ────────────────────────────────────────────────────────────────

CONFIG = {
    # 模型结构
    "vocab_size":      10_000,
    "context_length":  256,
    "d_model":         512,
    "num_layers":      4,
    "num_heads":       16,
    "d_ff":            1_344,   # ≈ 8/3 × 512，取 64 倍数
    "rope_theta":      10_000.0,
    # 训练
    "batch_size":      32,
    "max_iters":       65_918,  # 1 epoch on TinyStories (540M tokens / 32 / 256)
    "warmup_iters":    200,
    "max_lr":          1e-3,
    "min_lr":          1e-4,
    "weight_decay":    0.01,
    "betas":           (0.9, 0.999),
    "eps":             1e-8,
    "max_grad_norm":   1.0,
    # 评估 & 检查点
    "eval_interval":        500,
    "checkpoint_interval":  5_000,
}

DATA_DIR       = ROOT / "data" / "result"
CHECKPOINT_DIR = ROOT / "run" / "checkpoints"
LOG_FILE       = ROOT / "run" / "train_log.json"


# ─── 工具函数 ──────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ─── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train TinyStories GPT")
    parser.add_argument("--no-wandb", action="store_true", help="禁用 wandb 记录")
    parser.add_argument("--resume", type=str, default=None,
                        help="从检查点路径恢复训练")
    args = parser.parse_args()

    device = get_device()
    print(f"设备：{device}")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    print("加载数据集...")
    train_data = np.load(DATA_DIR / "tinystories_train.npy")
    val_data   = np.load(DATA_DIR / "tinystories_valid.npy")
    print(f"  训练集：{len(train_data):,} tokens")
    print(f"  验证集：{len(val_data):,} tokens")

    # ── 初始化模型 ────────────────────────────────────────────────────────────
    print("初始化模型...")
    model = TransformerLM(
        vocab_size=CONFIG["vocab_size"],
        context_length=CONFIG["context_length"],
        d_model=CONFIG["d_model"],
        num_layers=CONFIG["num_layers"],
        num_heads=CONFIG["num_heads"],
        d_ff=CONFIG["d_ff"],
        rope_theta=CONFIG["rope_theta"],
    )
    n_params = count_params(model)
    print(f"  参数量：{n_params:,}（{n_params / 1e6:.1f}M）")

    # ── 初始化优化器 ──────────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["max_lr"],
        betas=CONFIG["betas"],
        eps=CONFIG["eps"],
        weight_decay=CONFIG["weight_decay"],
    )

    # ── 恢复检查点（可选）────────────────────────────────────────────────────
    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"  从步骤 {start_iter} 恢复训练")

    # ── wandb 初始化 ──────────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project="cs336-tinystories",
            name=f"ts-{CONFIG['d_model']}d-{CONFIG['num_layers']}L",
            config={**CONFIG, "device": device, "param_count": n_params},
            resume="allow",
        )
        print(f"  wandb run: {wandb_run.url}")

    # ── 开始训练 ──────────────────────────────────────────────────────────────
    print(f"\n开始训练（共 {CONFIG['max_iters']:,} 步，1 epoch）...\n")
    train(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        optimizer=optimizer,
        batch_size=CONFIG["batch_size"],
        context_length=CONFIG["context_length"],
        max_iters=CONFIG["max_iters"],
        eval_interval=CONFIG["eval_interval"],
        checkpoint_dir=str(CHECKPOINT_DIR),
        checkpoint_interval=CONFIG["checkpoint_interval"],
        device=device,
        max_lr=CONFIG["max_lr"],
        min_lr=CONFIG["min_lr"],
        warmup_iters=CONFIG["warmup_iters"],
        max_grad_norm=CONFIG["max_grad_norm"],
        start_iter=start_iter,
        log_file=str(LOG_FILE),
        wandb_run=wandb_run,
    )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
