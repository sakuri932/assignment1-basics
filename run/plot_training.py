"""
训练曲线绘图脚本

用法：
    python run/plot_training.py                         # 使用默认日志路径
    python run/plot_training.py --log run/train_log.json
    python run/plot_training.py --out run/figures/      # 指定图片输出目录
"""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


ROOT    = Path(__file__).resolve().parent.parent
LOG_DEFAULT = ROOT / "run" / "train_log.json"
OUT_DEFAULT = ROOT / "run" / "figures"


def load_log(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def plot_training(log: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    steps       = [e["iter"]        for e in log]
    train_loss  = [e["train_loss"]  for e in log]

    val_steps   = [e["iter"]        for e in log if "val_loss" in e]
    val_loss    = [e["val_loss"]    for e in log if "val_loss" in e]
    val_ppl     = [e["val_ppl"]     for e in log if "val_ppl"  in e]
    lr_vals     = [e["lr"]          for e in log]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("TinyStories GPT Training", fontsize=14, fontweight="bold")

    # ── 图1：Train Loss & Val Loss ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(steps, train_loss, color="steelblue", alpha=0.4, linewidth=0.8,
            label="Train Loss")
    # 训练 loss 平滑曲线（50步滑动平均）
    window = 50
    if len(train_loss) >= window:
        smoothed = [
            sum(train_loss[max(0, i - window):i + 1]) / len(train_loss[max(0, i - window):i + 1])
            for i in range(len(train_loss))
        ]
        ax.plot(steps, smoothed, color="steelblue", linewidth=1.5,
                label=f"Train Loss (smooth {window})")
    if val_loss:
        ax.plot(val_steps, val_loss, color="tomato", linewidth=2,
                marker="o", markersize=3, label="Val Loss")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # ── 图2：Validation Perplexity ────────────────────────────────────────────
    ax = axes[1]
    if val_ppl:
        ax.plot(val_steps, val_ppl, color="darkorange", linewidth=2,
                marker="o", markersize=3, label="Val PPL")
        ax.set_ylabel("Perplexity")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    else:
        ax.text(0.5, 0.5, "No validation data", transform=ax.transAxes,
                ha="center", va="center", color="gray")

    # ── 图3：Learning Rate ────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(steps, lr_vals, color="mediumseagreen", linewidth=1.5)
    ax.set_ylabel("Learning Rate")
    ax.set_xlabel("Training Step")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    out_path = out_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"图表已保存：{out_path}")
    plt.show()

    # ── 打印最终指标摘要 ───────────────────────────────────────────────────────
    print("\n=== 训练摘要 ===")
    print(f"  总步数：{steps[-1]:,}")
    print(f"  最终 Train Loss：{train_loss[-1]:.4f}")
    if val_loss:
        best_val = min(val_loss)
        best_step = val_steps[val_loss.index(best_val)]
        print(f"  最优 Val Loss：{best_val:.4f}（Step {best_step:,}）")
        print(f"  最优 Val PPL： {min(val_ppl):.2f}")
        print(f"  最终 Val PPL： {val_ppl[-1]:.2f}")


def main():
    parser = argparse.ArgumentParser(description="绘制训练曲线")
    parser.add_argument("--log", type=str, default=str(LOG_DEFAULT),
                        help="train_log.json 路径")
    parser.add_argument("--out", type=str, default=str(OUT_DEFAULT),
                        help="图片输出目录")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"日志文件不存在：{log_path}")
        return

    log = load_log(log_path)
    print(f"已加载 {len(log):,} 条日志记录（共 {log[-1]['iter']:,} 步）")
    plot_training(log, Path(args.out))


if __name__ == "__main__":
    main()
