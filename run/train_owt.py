"""
OpenWebText GPT 训练启动脚本（RTX 4090 服务器版）

用法：
    python run/train_owt.py              # 正常训练（含 wandb）
    python run/train_owt.py --no-wandb   # 禁用 wandb
    python run/train_owt.py --resume run/checkpoints_owt/checkpoint_0010000.pt
"""

import os
import sys
import argparse
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    "vocab_size":      32_000,
    "context_length":  512,
    "d_model":         768,
    "num_layers":      12,
    "num_heads":       12,
    "d_ff":            2_048,
    "rope_theta":      10_000.0,
    # 训练
    "batch_size":      64,
    "max_iters":       83_225,   # 1 epoch on OWT (2.727B tokens / 64 / 512)
    "warmup_iters":    1_500,
    "max_lr":          3e-4,
    "min_lr":          3e-5,
    "weight_decay":    0.1,
    "betas":           (0.9, 0.95),
    "eps":             1e-8,
    "max_grad_norm":   1.0,
    # 评估 & 检查点
    "eval_interval":        500,
    "checkpoint_interval":  5_000,
}

DATA_DIR       = ROOT / "data" / "result"
CHECKPOINT_DIR = Path("/mnt/a/kong/checkpoints_owt")
LOG_FILE       = ROOT / "run" / "train_owt_log.json"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="Train OWT GPT on RTX 4090")
    parser.add_argument("--no-wandb", action="store_true", help="禁用 wandb 记录")
    parser.add_argument("--resume", type=str, default=None,
                        help="从检查点路径恢复训练")
    args = parser.parse_args()

    device = get_device()
    print(f"设备：{device}")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    print("加载数据集...")
    train_data = np.load(DATA_DIR / "owt_train.npy")
    val_data   = np.load(DATA_DIR / "owt_valid.npy")
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

    # ── 梯度检查点（显存受限时保持 batch=64 的关键）──────────────────────────
    if device == "cuda":
        model.gradient_checkpointing = True
        print("  梯度检查点已启用（activation recomputation）")

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
            project="cs336-owt",
            name=f"owt-{CONFIG['d_model']}d-{CONFIG['num_layers']}L",
            config={**CONFIG, "device": device, "param_count": n_params},
            resume="allow",
        )
        print(f"  wandb run: {wandb_run.url}")

    # ── 开始训练 ──────────────────────────────────────────────────────────────
    print(f"\n开始训练（共 {CONFIG['max_iters']:,} 步，1 epoch on OWT）...\n")
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
