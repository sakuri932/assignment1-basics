"""
TinyStories GPT 推理脚本（交互模式）

用法：
    python run/generate.py
    python run/generate.py --checkpoint run/checkpoints/checkpoint_0065000.pt
    python run/generate.py --max_tokens 300 --temperature 0.8 --top_p 0.9

输入 Ctrl+C 或 Ctrl+D 退出。
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import softmax


def load_tokenizer(vocab_path: str, merges_path: str, special_tokens: list[str]) -> Tokenizer:
    """加载自定义 hex 格式的词表和合并文件。"""
    with open(vocab_path, encoding="utf-8") as f:
        raw = json.load(f)
    vocab = {int(k): bytes.fromhex(v) for k, v in raw.items()}

    merges = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ")
            if len(parts) == 2:
                merges.append((bytes.fromhex(parts[0]), bytes.fromhex(parts[1])))

    return Tokenizer(vocab, merges, special_tokens)

CONFIG = {
    "vocab_size":     10_000,
    "context_length": 256,
    "d_model":        512,
    "num_layers":     4,
    "num_heads":      16,
    "d_ff":           1_344,
    "rope_theta":     10_000.0,
}

DATA_DIR       = ROOT / "data" / "result"
CHECKPOINT_DIR = ROOT / "run" / "checkpoints"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="TinyStories GPT 文本生成（交互模式）")
    parser.add_argument("--checkpoint", type=str,
                        default=str(CHECKPOINT_DIR / "checkpoint_final.pt"),
                        help="检查点路径")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="最多生成的新 token 数")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="采样温度（越低越保守，越高越随机）")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="nucleus sampling 阈值（默认 0.9）")
    args = parser.parse_args()

    device = get_device()
    print(f"设备：{device}")

    # 加载分词器
    tokenizer = load_tokenizer(
        vocab_path=str(DATA_DIR / "tinystories_vocab.json"),
        merges_path=str(DATA_DIR / "tinystories_merges.txt"),
        special_tokens=["<|endoftext|>"],
    )

    # 初始化模型
    model = TransformerLM(
        vocab_size=CONFIG["vocab_size"],
        context_length=CONFIG["context_length"],
        d_model=CONFIG["d_model"],
        num_layers=CONFIG["num_layers"],
        num_heads=CONFIG["num_heads"],
        d_ff=CONFIG["d_ff"],
        rope_theta=CONFIG["rope_theta"],
    )

    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(f"已加载检查点：{args.checkpoint}（step {checkpoint['iteration']}）")
    print("输入 Ctrl+C 或 Ctrl+D 退出。\n")

    # 交互循环
    while True:
        try:
            prompt = input("请输入文本：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not prompt:
            continue

        print()
        eos_id = tokenizer.bytes_to_id.get(b"<|endoftext|>", None)
        input_ids = tokenizer.encode(prompt)
        token_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        prev_text = tokenizer.decode(input_ids)
        print(prev_text, end="", flush=True)

        model.eval()
        with torch.no_grad():
            for _ in range(args.max_tokens):
                ctx = token_ids[:, -model.context_length:]
                logits = model(ctx)
                next_logits = logits[0, -1, :] / args.temperature

                probs = softmax(next_logits, dim=0)
                if args.top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=0)
                    sorted_probs[cumsum - sorted_probs > args.top_p] = 0.0
                    sorted_probs /= sorted_probs.sum()
                    next_token = sorted_idx[torch.multinomial(sorted_probs, 1)]
                else:
                    next_token = torch.multinomial(probs, 1)

                token_ids = torch.cat([token_ids, next_token.view(1, 1)], dim=1)

                cur_text = tokenizer.decode(token_ids[0].tolist())
                print(cur_text[len(prev_text):], end="", flush=True)
                prev_text = cur_text

                if eos_id is not None and next_token.item() == eos_id:
                    break

        print("\n")


if __name__ == "__main__":
    main()
