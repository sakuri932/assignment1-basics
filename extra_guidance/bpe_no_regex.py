"""
bpe_no_regex.py — 简化版 BPE 分词器（无正则表达式）

与 cs336_basics/tokenizer.py 功能相同，但预分词和文本分割
全部用简单的字符遍历实现，不依赖 re / regex 模块。

核心简化思路
────────────
原版预分词使用了复杂的 GPT-2 正则，处理缩写、前导空格吸收等细节。
本版改用最直接的策略：按字符类型（字母/数字/空白/其他）切分文本。
相邻同类字符归为一个预分词单元，类型切换处即为边界。

这种方式比原版简单得多，同样能防止 BPE 跨词合并，
训练出的分词器自洽（训练和编码使用同一套规则）。
"""

import os
import json
import unicodedata
import multiprocessing
from collections import defaultdict
from typing import Iterator, Iterable


# ─────────────────────────────────────────────────────────────────────────────
# 预分词：按字符类型切分（字母 / 数字 / 空白 / 其他）
# ─────────────────────────────────────────────────────────────────────────────

def _char_type(c: str) -> str:
    """
    返回字符的类型标签，用于预分词时判断类型边界。

    输入：单个 Unicode 字符
    输出：'L'（字母）/ 'N'（数字）/ 'S'（空白）/ 'O'（其他）

    使用 unicodedata.category() 获取 Unicode 通用类别，
    取首字母判断大类，覆盖所有语言的字母和数字。
    """
    cat = unicodedata.category(c)
    if cat[0] == 'L':
        return 'L'
    if cat[0] == 'N':
        return 'N'
    if cat[0] == 'Z' or c in '\t\n\r\x0b\x0c':
        return 'S'
    return 'O'


def _pretokenize(text: str) -> list[str]:
    """
    将文本切分为预分词单元：连续相同类型的字符归为一组。

    输入：
        text (str): 任意文本

    输出：
        list[str]: 预分词单元列表
        示例：
            "hello world" → ["hello", " ", "world"]
            "3.14 ok"     → ["3", ".", "14", " ", "ok"]
            "it's"        → ["it", "'", "s"]

    实现：
        遍历每个字符，记录当前段的起始位置和类型。
        字符类型发生变化时，将当前段作为一个单元切出，重新开始。
    """
    if not text:
        return []

    tokens: list[str] = []
    start = 0
    current_type = _char_type(text[0])

    for i in range(1, len(text)):
        t = _char_type(text[i])
        if t != current_type:
            tokens.append(text[start:i])
            start = i
            current_type = t

    tokens.append(text[start:])   # 最后一段
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# 特殊 token 分割：线性扫描，最长匹配优先
# ─────────────────────────────────────────────────────────────────────────────

def _split_on_specials(text: str, special_tokens: list[str]) -> list[str]:
    """
    按特殊 token 分割文本，保留特殊 token 本身作为独立元素。

    输入：
        text (str): 待分割文本
        special_tokens (list[str]): 特殊 token 列表

    输出：
        list[str]: 普通文本片段与特殊 token 交替出现的列表
        示例：
            "hello<|endoftext|>world", ["<|endoftext|>"]
            → ["hello", "<|endoftext|>", "world"]

    实现：
        从左到右逐字符扫描，每个位置尝试匹配特殊 token（长的优先）。
        匹配到则提交当前片段 + 特殊 token，然后跳过；否则继续。
    """
    if not special_tokens:
        return [text]

    # 长度降序排列，确保较长 token 优先匹配，避免前缀遮蔽
    sorted_specials = sorted(special_tokens, key=len, reverse=True)

    result: list[str] = []
    i = 0
    chunk_start = 0

    while i < len(text):
        for st in sorted_specials:
            if text[i: i + len(st)] == st:
                if i > chunk_start:
                    result.append(text[chunk_start:i])
                result.append(st)
                i += len(st)
                chunk_start = i
                break
        else:
            i += 1

    if chunk_start < len(text):
        result.append(text[chunk_start:])

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 文件分块（用于并行预分词）
# ─────────────────────────────────────────────────────────────────────────────

def _find_chunk_boundaries(file, num_chunks: int, split_token: bytes) -> list[int]:
    """
    将文件分成以特殊 token 字节串对齐的块，返回字节偏移量列表。

    输入：
        file: 已打开的二进制文件对象
        num_chunks (int): 期望的块数（通常等于 CPU 核心数）
        split_token (bytes): 用于对齐边界的字节串，如 b"<|endoftext|>"

    输出：
        list[int]: 排序去重后的偏移量列表，格式 [0, ..., file_size]

    实现：
        先计算等分边界，然后对每个内部边界向后搜索最近的 split_token，
        将边界对齐到文档分隔符处，防止并行块截断文档。
        使用 bytes.find() 进行字节搜索，无需正则。
    """
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // num_chunks
    boundaries = [i * chunk_size for i in range(num_chunks + 1)]
    boundaries[-1] = file_size

    for bi in range(1, len(boundaries) - 1):
        pos = boundaries[bi]
        file.seek(pos)
        while True:
            chunk = file.read(4096)
            if not chunk:
                boundaries[bi] = file_size
                break
            idx = chunk.find(split_token)
            if idx != -1:
                boundaries[bi] = pos + idx
                break
            pos += 4096

    return sorted(set(boundaries))


# ─────────────────────────────────────────────────────────────────────────────
# 并行预分词工作函数（模块顶层，可被 multiprocessing 序列化）
# ─────────────────────────────────────────────────────────────────────────────

def _worker(args: tuple) -> dict:
    """
    多进程工作函数：读取文件一个字节区间，返回预分词单元频次字典。

    输入（通过元组传入，兼容 Pool.map 的单参数接口）：
        args[0] (str): 文件路径
        args[1] (int): 起始字节偏移量
        args[2] (int): 结束字节偏移量
        args[3] (list[str]): 特殊 token 列表

    输出：
        dict[tuple[bytes, ...], int]:
            键：预分词单元的单字节元组，如 (b'h', b'i') 表示 "hi"
            值：该单元在本块中的出现次数

    流程：
        1. 读取字节 → UTF-8 解码（容错）
        2. 按特殊 token 分割 → 过滤掉特殊 token 本身
        3. 对普通文本预分词 → 统计字节元组频次
    """
    file_path, start, end, special_tokens = args

    with open(file_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")

    special_set = set(special_tokens)
    counts: dict[tuple, int] = defaultdict(int)

    for part in _split_on_specials(text, special_tokens):
        if not part or part in special_set:
            continue
        for word in _pretokenize(part):
            if word:
                # 将预分词单元拆成单字节元组，作为 BPE 的初始表示
                # 例："hi" → (b'h', b'i')
                counts[tuple(bytes([b]) for b in word.encode("utf-8"))] += 1

    return dict(counts)


# ─────────────────────────────────────────────────────────────────────────────
# BPE 训练
# ─────────────────────────────────────────────────────────────────────────────

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练字节级 BPE 分词器。

    输入：
        input_path: 训练语料文件路径
        vocab_size: 目标词表大小（特殊 token + 256 字节 + 合并 token）
        special_tokens: 特殊 token 列表，训练时作为文档边界

    输出：
        vocab (dict[int, bytes]):
            token ID → 字节串。分配顺序：特殊 token → 256 字节值 → 合并 token。
        merges (list[tuple[bytes, bytes]]):
            BPE 合并列表，按创建顺序排列。编码时必须按此顺序应用。

    算法：
        Step 1  初始化词表：特殊 token + 256 字节值
        Step 2  并行预分词：multiprocessing 加速，按特殊 token 对齐分块边界
        Step 3  BPE 合并循环：
                  a. 找全局频次最高的相邻字节对（同频取字典序最大，保证确定性）
                  b. 合并该对，加入词表和 merges
                  c. 增量更新：只更新受影响的预分词单元，而非重扫全部语料

        增量更新依赖两个索引：
          pair_counts    (a, b) → 全局频次
          pair_to_words  (a, b) → 含该对的预分词单元 ID 集合
        合并后只需更新 pair_to_words[best_pair] 中的单元，代价远低于全量扫描。
    """
    # ── Step 1：初始化词表 ────────────────────────────────────────────────
    vocab: dict[int, bytes] = {}
    idx = 0
    for s in special_tokens:
        vocab[idx] = s.encode("utf-8"); idx += 1
    for b in range(256):
        vocab[idx] = bytes([b]); idx += 1

    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []

    # ── Step 2：并行预分词 ────────────────────────────────────────────────
    split_bytes = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
    n_proc = max(1, multiprocessing.cpu_count())

    with open(input_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, n_proc, split_bytes)

    chunk_args = [
        (str(input_path), boundaries[i], boundaries[i + 1], special_tokens)
        for i in range(len(boundaries) - 1)
        if boundaries[i] < boundaries[i + 1]
    ]

    pretoken_counts: dict[tuple, int] = defaultdict(int)
    pool_fn = multiprocessing.Pool if len(chunk_args) > 1 else None
    if pool_fn:
        with multiprocessing.Pool(n_proc) as pool:
            for res in pool.map(_worker, chunk_args):
                for k, v in res.items():
                    pretoken_counts[k] += v
    else:
        for res in map(_worker, chunk_args):
            for k, v in res.items():
                pretoken_counts[k] += v

    # ── Step 3：初始化 BPE 索引 ───────────────────────────────────────────
    # words: word_id → (当前 token 列表, 频次)
    words: dict[int, tuple[list[bytes], int]] = {
        wid: (list(tup), cnt)
        for wid, (tup, cnt) in enumerate(pretoken_counts.items())
    }

    # pair_counts: (a, b) → 全局加权频次
    # pair_to_words: (a, b) → 含该对的 word_id 集合
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for wid, (tokens, cnt) in words.items():
        for k in range(len(tokens) - 1):
            p = (tokens[k], tokens[k + 1])
            pair_counts[p] += cnt
            pair_to_words[p].add(wid)

    # ── Step 3：合并循环 ──────────────────────────────────────────────────
    merges: list[tuple[bytes, bytes]] = []

    for _ in range(num_merges):
        if not pair_counts:
            break

        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        if pair_counts[best] <= 0:
            break

        merged = best[0] + best[1]
        merges.append(best)
        vocab[idx] = merged; idx += 1

        for wid in list(pair_to_words.get(best, set())):
            tokens, cnt = words[wid]

            # 撤销旧贡献
            for k in range(len(tokens) - 1):
                p = (tokens[k], tokens[k + 1])
                pair_counts[p] -= cnt
                pair_to_words[p].discard(wid)
                if pair_counts[p] <= 0:
                    del pair_counts[p]

            # 应用合并
            new: list[bytes] = []
            k = 0
            while k < len(tokens):
                if k + 1 < len(tokens) and tokens[k] == best[0] and tokens[k + 1] == best[1]:
                    new.append(merged); k += 2
                else:
                    new.append(tokens[k]); k += 1
            words[wid] = (new, cnt)

            # 加入新贡献
            for k in range(len(new) - 1):
                p = (new[k], new[k + 1])
                pair_counts[p] = pair_counts.get(p, 0) + cnt
                pair_to_words[p].add(wid)

        pair_to_words.pop(best, None)

    return vocab, merges


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer 类
# ─────────────────────────────────────────────────────────────────────────────

class Tokenizer:
    """
    字节级 BPE 分词器，支持编码（str → list[int]）和解码（list[int] → str）。

    编码流程：
        文本 → 按特殊 token 分割 → 普通段预分词 → 对每个预分词单元应用 BPE → token ID

    解码流程：
        token ID → 查词表得字节串 → 拼接 → UTF-8 解码
        先拼接后解码，保证跨 token 的多字节 UTF-8 字符完整还原。
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """
        输入：
            vocab: token ID → 字节串，通常来自 train_bpe 的返回值
            merges: BPE 合并列表，按创建顺序排列
            special_tokens: 特殊 token 字符串列表，编码时整体处理、不拆分
        """
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens) if special_tokens else []

        # 反向映射：字节串 → token ID，用于编码时查找 ID
        self.bytes_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        # 若特殊 token 不在词表中，追加到末尾
        for s in self.special_tokens:
            sb = s.encode("utf-8")
            if sb not in self.bytes_to_id:
                new_id = max(self.vocab) + 1 if self.vocab else 0
                self.vocab[new_id] = sb
                self.bytes_to_id[sb] = new_id

        # 合并优先级：merge → rank（越小越早创建，编码时越优先应用）
        self.merge_rank: dict[tuple[bytes, bytes], int] = {
            m: r for r, m in enumerate(self.merges)
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        从 GPT-2 格式的词表文件和合并文件加载分词器。

        vocab_filepath: JSON 文件，键为 GPT-2 字节编码的 token 字符串，值为 token ID。
        merges_filepath: 文本文件，每行 "token1 token2"，以 '#' 开头的行为注释。

        GPT-2 字节编码将 256 个字节值映射到可打印 Unicode 字符，以避免
        JSON 中出现控制字符。加载时需要反向映射：Unicode 字符 → 字节值。
        """
        c2b = cls._char_to_byte()

        with open(vocab_filepath, encoding="utf-8") as f:
            raw: dict[str, int] = json.load(f)
        vocab: dict[int, bytes] = {
            int(v): bytes([c2b[c] for c in k]) for k, v in raw.items()
        }

        with open(merges_filepath, encoding="utf-8") as f:
            lines = f.readlines()
        merges: list[tuple[bytes, bytes]] = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b = line.split(" ", 1)
            merges.append((
                bytes([c2b[c] for c in a]),
                bytes([c2b[c] for c in b]),
            ))

        return cls(vocab, merges, special_tokens)

    @staticmethod
    def _char_to_byte() -> dict[str, int]:
        """
        GPT-2 字节编码反向映射：Unicode 字符 → 原始字节值（0-255）。

        GPT-2 将可打印字节（33-126, 161-172, 174-255）映射到自身 Unicode 码点，
        其余字节从 256 开始依次分配 Unicode 码点。
        本函数返回反向映射，用于还原 vocab.json / merges.txt 中的字节串。
        """
        printable = (
            list(range(33, 127))      # 可打印 ASCII
            + list(range(161, 173))   # Latin-1 可打印
            + list(range(174, 256))   # Latin-1 可打印
        )
        b2u: dict[int, int] = {b: b for b in printable}
        extra = 256
        for b in range(256):
            if b not in b2u:
                b2u[b] = extra; extra += 1
        return {chr(u): b for b, u in b2u.items()}

    def _apply_bpe(self, word_bytes: list[bytes]) -> list[bytes]:
        """
        对单个预分词单元的字节列表应用 BPE 合并。

        输入：
            word_bytes: 单字节列表，如 [b'h', b'e', b'l', b'l', b'o']

        输出：
            合并后的字节 token 列表，如 [b'hello']（若词表学到了对应合并）

        算法：
            每轮找 rank 最小（最早创建）的可用相邻对，将所有该对替换为合并结果。
            重复直到无可用合并。等价于按合并创建顺序重放 BPE 训练过程。
        """
        tokens = list(word_bytes)

        while len(tokens) > 1:
            # 找 rank 最小的可用对
            best_rank, best_pair = float("inf"), None
            for k in range(len(tokens) - 1):
                r = self.merge_rank.get((tokens[k], tokens[k + 1]), float("inf"))
                if r < best_rank:
                    best_rank, best_pair = r, (tokens[k], tokens[k + 1])

            if best_pair is None:
                break

            merged = best_pair[0] + best_pair[1]
            new: list[bytes] = []
            k = 0
            while k < len(tokens):
                if k + 1 < len(tokens) and tokens[k] == best_pair[0] and tokens[k + 1] == best_pair[1]:
                    new.append(merged); k += 2
                else:
                    new.append(tokens[k]); k += 1
            tokens = new

        return tokens

    def encode(self, text: str) -> list[int]:
        """
        将文本编码为 token ID 列表。

        流程：
            1. 按特殊 token 分割文本（特殊 token 直接映射为固定 ID，不拆分）
            2. 普通文本段：预分词 → 对每个单元应用 BPE → 转 ID

        输入：任意 Unicode 字符串
        输出：token ID 整数列表
        """
        ids: list[int] = []
        special_set = set(self.special_tokens)

        for part in _split_on_specials(text, self.special_tokens):
            if not part:
                continue
            if part in special_set:
                ids.append(self.bytes_to_id[part.encode("utf-8")])
            else:
                for word in _pretokenize(part):
                    if word:
                        word_bytes = [bytes([b]) for b in word.encode("utf-8")]
                        for tb in self._apply_bpe(word_bytes):
                            ids.append(self.bytes_to_id[tb])

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        对字符串可迭代对象惰性编码，逐个 yield token ID。

        输入：字符串可迭代对象（文件句柄、列表等）
        产出：token ID（int）

        内存高效：适用于大型文件，逐块处理而非整体载入。
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        """
        将 token ID 列表解码为文本字符串。

        输入：token ID 列表（每个 ID 必须在词表中）
        输出：UTF-8 解码后的字符串，无效字节用 U+FFFD 替换

        先拼接所有字节串再解码，保证跨 token 的多字节字符完整还原。
        """
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
