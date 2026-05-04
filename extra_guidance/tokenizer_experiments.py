"""
CS336 Assignment 1 Section 2.7 — Tokenizer Experiments

运行方式（在 assignment1-basics 目录下）：
    uv run python extra/tokenizer_experiments.py

前提：已调用 train_bpe 训练好两个分词器，或本脚本会自动训练。
结果会打印到终端，并把编码后的 token 数组保存到 data/ 目录。
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# 确保能 import cs336_basics
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cs336_basics.tokenizer import Tokenizer, train_bpe

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPECIAL_TOKENS = ["<|endoftext|>"]
DOC_SEP = "<|endoftext|>"


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def load_or_train_tokenizer(
    train_path: Path,
    vocab_size: int,
    vocab_save: Path,
    merges_save: Path,
) -> Tokenizer:
    """
    如果已保存的词表文件存在则直接加载，否则训练并保存。
    词表存为 JSON（id → bytes 的 hex 字符串），合并规则存为每行两个 hex token。
    """
    if vocab_save.exists() and merges_save.exists():
        print(f"  加载已保存的分词器: {vocab_save.name}")
        with open(vocab_save) as f:
            raw = json.load(f)
        vocab = {int(k): bytes.fromhex(v) for k, v in raw.items()}
        merges = []
        with open(merges_save) as f:
            for line in f:
                a, b = line.strip().split(" ")
                merges.append((bytes.fromhex(a), bytes.fromhex(b)))
        return Tokenizer(vocab=vocab, merges=merges, special_tokens=SPECIAL_TOKENS)

    print(f"  训练分词器 vocab_size={vocab_size}，语料：{train_path.name} ...")
    t0 = time.time()
    vocab, merges = train_bpe(
        input_path=train_path,
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
    )
    print(f"  训练完成，耗时 {time.time() - t0:.1f}s")

    # 保存词表（bytes 对象无法直接 JSON 序列化，转成 hex 字符串）
    with open(vocab_save, "w") as f:
        json.dump({str(k): v.hex() for k, v in vocab.items()}, f)
    with open(merges_save, "w") as f:
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")
    print(f"  已保存 → {vocab_save.name}, {merges_save.name}")

    return Tokenizer(vocab=vocab, merges=merges, special_tokens=SPECIAL_TOKENS)


def sample_documents(filepath: Path, n: int = 10) -> list[str]:
    """从文件中抽取前 n 个非空文档（以 <|endoftext|> 分隔）。"""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    docs = [d.strip() for d in content.split(DOC_SEP)]
    docs = [d for d in docs if d]  # 过滤空文档
    return docs[:n]


def compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> tuple[float, int, int]:
    """
    计算压缩比 = 原始字节数 / token 数。
    返回 (ratio, total_bytes, total_tokens)
    """
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    total_tokens = sum(len(tokenizer.encode(d)) for d in docs)
    ratio = total_bytes / total_tokens if total_tokens > 0 else 0.0
    return ratio, total_bytes, total_tokens


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("步骤 0：加载或训练分词器")
    print("=" * 60)

    tok_ts = load_or_train_tokenizer(
        train_path=DATA_DIR / "TinyStoriesV2-GPT4-train.txt",
        vocab_size=10_000,
        vocab_save=DATA_DIR / "tinystories_vocab.json",
        merges_save=DATA_DIR / "tinystories_merges.txt",
    )
    tok_owt = load_or_train_tokenizer(
        train_path=DATA_DIR / "owt_train.txt",
        vocab_size=32_000,
        vocab_save=DATA_DIR / "owt_vocab.json",
        merges_save=DATA_DIR / "owt_merges.txt",
    )

    # ─────────────────────────────────────────
    # (a) 压缩比：各用自己的分词器
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("(a) 压缩比（每个分词器处理自己的数据集）")
    print("=" * 60)

    ts_docs = sample_documents(DATA_DIR / "TinyStoriesV2-GPT4-train.txt", n=10)
    owt_docs = sample_documents(DATA_DIR / "owt_train.txt", n=10)

    r_ts, b_ts, t_ts = compression_ratio(tok_ts, ts_docs)
    r_owt, b_owt, t_owt = compression_ratio(tok_owt, owt_docs)

    print(f"\nTinyStories 分词器 on TinyStories 样本（10篇文档）：")
    print(f"  原始字节：{b_ts:,}  Token 数：{t_ts:,}  压缩比：{r_ts:.4f} 字节/token")

    print(f"\nOpenWebText 分词器 on OpenWebText 样本（10篇文档）：")
    print(f"  原始字节：{b_owt:,}  Token 数：{t_owt:,}  压缩比：{r_owt:.4f} 字节/token")

    # ─────────────────────────────────────────
    # (b) 跨数据集：用 TinyStories 分词器处理 OWT
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("(b) 跨分词器：用 TinyStories 分词器处理 OpenWebText 样本")
    print("=" * 60)

    r_ts_on_owt, b_cross, t_cross = compression_ratio(tok_ts, owt_docs)
    print(f"\n  原始字节：{b_cross:,}  Token 数：{t_cross:,}  压缩比：{r_ts_on_owt:.4f} 字节/token")
    print(f"  对比 OWT 自己的分词器：{r_owt:.4f} 字节/token")
    if r_ts_on_owt < r_owt:
        print(f"  → TinyStories 分词器压缩率更低（{r_ts_on_owt:.4f} < {r_owt:.4f}），说明它不适合 OWT 领域的文本。")
    else:
        print(f"  → 压缩比与 OWT 自身分词器接近。")

    # 定性分析：展示几个 token
    print("\n  定性示例（第1篇文档，前200字节）：")
    sample = owt_docs[0][:200]
    ids_ts = tok_ts.encode(sample)
    ids_owt = tok_owt.encode(sample)
    print(f"    TinyStories 分词器 → {len(ids_ts)} tokens")
    print(f"    OWT 分词器         → {len(ids_owt)} tokens")
    print(f"    前10个 token（TinyStories）: {[tok_ts.decode([i]) for i in ids_ts[:10]]}")
    print(f"    前10个 token（OWT）:         {[tok_owt.decode([i]) for i in ids_owt[:10]]}")

    # ─────────────────────────────────────────
    # (c) 吞吐量估算
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("(c) 吞吐量估算")
    print("=" * 60)

    # 用 encode_iterable 在大文件上测速
    benchmark_path = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"
    print(f"\n  在 {benchmark_path.name}（{benchmark_path.stat().st_size / 1e6:.1f} MB）上测速...")

    total_bytes = 0
    token_count = 0
    t0 = time.time()
    with open(benchmark_path, encoding="utf-8") as f:
        # 只测前 10MB 防止等待太久
        chunk = f.read(10 * 1024 * 1024)
        total_bytes = len(chunk.encode("utf-8"))
        token_count = len(tok_ts.encode(chunk))
    elapsed = time.time() - t0

    throughput_bytes_per_sec = total_bytes / elapsed
    throughput_mb_per_sec = throughput_bytes_per_sec / 1e6

    pile_size_bytes = 825e9  # 825 GB
    est_seconds = pile_size_bytes / throughput_bytes_per_sec
    est_hours = est_seconds / 3600

    print(f"\n  测速输入：{total_bytes / 1e6:.1f} MB  → {token_count:,} tokens  耗时 {elapsed:.2f}s")
    print(f"  吞吐量：{throughput_mb_per_sec:.2f} MB/s  ({throughput_bytes_per_sec:.0f} 字节/s)")
    print(f"  对 Pile（825GB）的估计分词时间：{est_seconds:.0f}s ≈ {est_hours:.1f} 小时")

    # ─────────────────────────────────────────
    # (d) 编码完整训练/验证集为 uint16 数组
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("(d) 编码完整数据集为 uint16 NumPy 数组")
    print("=" * 60)

    tasks = [
        (tok_ts, DATA_DIR / "TinyStoriesV2-GPT4-train.txt", DATA_DIR / "tinystories_train.npy"),
        (tok_ts, DATA_DIR / "TinyStoriesV2-GPT4-valid.txt", DATA_DIR / "tinystories_valid.npy"),
        (tok_owt, DATA_DIR / "owt_train.txt", DATA_DIR / "owt_train.npy"),
        (tok_owt, DATA_DIR / "owt_valid.txt", DATA_DIR / "owt_valid.npy"),
    ]

    for tokenizer, input_path, output_path in tasks:
        if output_path.exists():
            arr = np.load(output_path)
            print(f"  已存在 {output_path.name}：{len(arr):,} tokens，{output_path.stat().st_size / 1e6:.1f} MB")
            continue

        print(f"\n  正在编码 {input_path.name} ...")
        t0 = time.time()
        ids = []
        with open(input_path, encoding="utf-8") as f:
            for token_id in tokenizer.encode_iterable(f):
                ids.append(token_id)
        arr = np.array(ids, dtype=np.uint16)
        np.save(output_path, arr)
        elapsed = time.time() - t0

        print(f"  → {output_path.name}：{len(arr):,} tokens，{output_path.stat().st_size / 1e6:.1f} MB，耗时 {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print("为什么选 uint16？")
    print("=" * 60)
    print("""
  uint16 范围：0 ~ 65535
  TinyStories 分词器 vocab_size = 10,000  → 最大 ID 9,999  < 65535 ✓
  OpenWebText 分词器 vocab_size = 32,000  → 最大 ID 31,999 < 65535 ✓

  内存/存储对比（假设 10亿 tokens）：
    uint32：4 字节/token → 4 GB
    uint16：2 字节/token → 2 GB  节省一半空间

  为什么不用 uint8？
    uint8 最大值 255，只够存原始字节（BPE 前），无法存合并后的 token ID。

  使用方式：
    arr = np.load("tinystories_train.npy")   # 直接加载
    # arr.dtype == uint16，arr[i] 是第 i 个 token 的整数 ID
""")

    print("全部完成。")


if __name__ == "__main__":
    main()
