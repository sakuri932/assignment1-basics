"""
BPE (Byte-Pair Encoding) Tokenizer: Training and Encoding/Decoding.

本文件实现了两个核心功能：
1. train_bpe: 从零开始训练字节级 BPE 分词器
2. Tokenizer 类: 使用训练好的 BPE 词表进行文本编码和解码

BPE 算法原理 (Sennrich et al., 2016):
- 初始词表由 256 个字节值组成
- 迭代地找出语料库中最频繁的字节对并合并
- 最终词表大小 = 256 + 特殊 token 数 + 合并次数

References:
- Sennrich et al., 2016: Neural Machine Translation of Rare Words with Subword Units
- Radford et al., 2019: Language Models are Unsupervised Multitask Learners (GPT-2)
"""

import os
import re
import json
import regex
import multiprocessing
from collections import defaultdict
from typing import Any, Iterator, Iterable

# GPT-2 风格的预分词正则表达式
# 该模式将文本分割为"词"，BPE 不会跨越这些边界合并
# 来源: github.com/openai/tiktoken/pull/234/files
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _find_chunk_boundaries(
    file, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    """
    将文件分割为以特殊 token 对齐的块，返回字节偏移量边界列表。
    确保每个 chunk 的边界都落在特殊 token 的起始位置，
    这样并行预分词时不会在文档中间截断。
    """
    # 1.按特殊 token 分割文本块（粒度：多个短篇故事），防止 BPE 跨文档边界合并
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)] #给出初始化文本块分割
    chunk_boundaries[-1] = file_size #读取文件末尾，用于修正最后一个块的边界

    mini_chunk_size = 4096  # 每次向前查找 4KB
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _pretokenize_chunk(args: tuple) -> dict:
    """
    Worker 函数: 读取文件的一个 chunk，进行预分词并统计 pre-token 频次。

    策略:
    1. 按特殊 token 分割 chunk（防止跨文档合并）
    2. 对每个非特殊 token 部分用 GPT-2 正则进行预分词
    3. 将每个 pre-token 转为 UTF-8 字节元组，统计出现次数

    返回: dict[tuple[bytes, ...], int] - pre-token 字节元组到频次的映射
    """
    file_path, start, end, special_tokens = args

    with open(file_path, "rb") as f:
        f.seek(start)
        raw_bytes = f.read(end - start)

    chunk = raw_bytes.decode("utf-8", errors="ignore")

    # 2.按特殊 token 分割文本块（粒度：单个短篇故事），防止 BPE 跨文档边界合并
    if special_tokens:
        #正则表达式特殊 token 用escape转义，按长度降序确保最长匹配优先，这样不会匹配出错从而报错
        escaped = [re.escape(st) for st in sorted(special_tokens, key=len, reverse=True)]
        split_pat = "|".join(escaped)
        # split 保留分隔符（特殊 token）在结果中，并切割文本
        parts = re.split(f"({split_pat})", chunk)
    else:
        parts = [chunk]

    pretoken_counts: dict[tuple, int] = defaultdict(int)
    special_tokens_set = set(special_tokens) if special_tokens else set()

    #3.对每篇小故事，按特殊 token 分割文本块（粒度：故事的每个单词）为pre-token，统计 pre-token 频次
    for part in parts:
        if not part:
            continue
        # 跳过特殊 token 本身（它们不参与 BPE 合并统计）
        if part in special_tokens_set:
            continue
        # 使用 GPT-2 正则进行预分词
        #regex.finditer(pattern, string) 会在 string 里从左到右扫描，找出所有匹配 pattern 的地方，返回一个迭代器，每次产出一个匹配对象 match。
        for match in regex.finditer(PAT, part):
            word = match.group(0) #group(0) 表示匹配到的pre-token，如 "hello"
            # 将每个字符的 UTF-8 编码拆为单独字节
            word_bytes_tuple = tuple(bytes([b]) for b in word.encode("utf-8"))
            pretoken_counts[word_bytes_tuple] += 1

    return dict(pretoken_counts)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练字节级 BPE 分词器。

    算法流程:
    1. 初始化词表: 特殊 token + 256 个字节值
    2. 并行预分词: 使用 multiprocessing 加速，按特殊 token 划分 chunk
    3. BPE 合并循环:
       a. 统计所有相邻字节对的频次
       b. 找出最频繁的对（频率相同时取字典序最大的）
       c. 将该对合并为新 token，加入词表
       d. 增量更新频次统计

    关键优化: 维护 pair_to_words 索引，每次合并只更新受影响的词，
    而不是重新扫描全部语料，时间复杂度从 O(N) 降至 O(W*L)，
    其中 W 是包含该对的词数，L 是平均词长。

    Args:
        input_path: 训练语料文本文件路径
        vocab_size: 最终词表大小（含特殊 token 和字节词表）
        special_tokens: 特殊 token 列表，训练时作为硬边界

    Returns:
        vocab: dict[int, bytes] - token ID 到字节串的映射
        merges: list[tuple[bytes, bytes]] - BPE 合并列表，按创建顺序排列
    """
    # --- Step 1: 初始化词表 ---
    # 特殊 token 先分配 ID，然后是 256 个字节值
    vocab: dict[int, bytes] = {}
    idx = 0
    for token in special_tokens:
        vocab[idx] = token.encode("utf-8")
        idx += 1
    for byte_val in range(256):
        vocab[idx] = bytes([byte_val])
        idx += 1

    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []

    # --- Step 2: 并行预分词 ---
    # 用第一个特殊 token 作为文档边界，便于 chunking
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
    num_processes = max(1, multiprocessing.cpu_count())

    with open(input_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, num_processes, split_token)

    chunk_args = [
        (str(input_path), boundaries[i], boundaries[i + 1], special_tokens)
        for i in range(len(boundaries) - 1)
        if boundaries[i] < boundaries[i + 1]
    ]

    pretoken_counts: dict[tuple, int] = defaultdict(int)

    #多进程设置
    if len(chunk_args) > 1:
        with multiprocessing.Pool(num_processes) as pool:
            results = pool.map(_pretokenize_chunk, chunk_args)
    else:
        results = [_pretokenize_chunk(a) for a in chunk_args]
    #多进程结果合并为总字典
    for result in results:
        for word_tuple, count in result.items():
            pretoken_counts[word_tuple] += count

    # --- Step 3: BPE 合并循环 ---
    # words: word_id -> (可变字节列表, 频次)，目的是转换“单词表”字典为更好的结构，以便增量更新
    words: dict[int, tuple[list[bytes], int]] = {
        i: (list(word_tuple), count)
        for i, (word_tuple, count) in enumerate(pretoken_counts.items())
    }

    # pair_counts: (a, b) -> 在所有词中出现的总次数
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int) #收集最终相邻对数量
    # pair_to_words: (a, b) -> 包含该对的 word ID 集合（用于增量更新）
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    # 初始化对频次
    for word_id, (tokens, count) in words.items():
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += count #双for扫描加和包含该对的词频总和
            pair_to_words[pair].add(word_id) #双for记录包含该对的词ID集合

    merges: list[tuple[bytes, bytes]] = []

    #根据对频次合并单词文本集
    for _ in range(num_merges):
        if not pair_counts:
            break

        # 选出最高频对；频率相同时取字典序最大的（确定性行为）
        #对于每个字节对 p，用 (pair_counts[p], p) 这个元组来比较大小。
        # Python 比较元组时，先比第一个元素，第一个相同再比第二个。所以这相当于：先按频次排，频次一样就按字节对本身的字典序排。
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))

        if pair_counts[best_pair] <= 0:
            break

        # 创建合并后的新 token
        merged = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[idx] = merged
        idx += 1

        # 对所有包含 best_pair 的词进行增量更新
        affected_words = list(pair_to_words.get(best_pair, set()))

        for word_id in affected_words:
            tokens, count = words[word_id]

            #这个单词不存在了，成了一个新单词，所以它对所有相邻对的贡献都要减去，然后再加上新单词对的贡献
            # 先从 pair_counts 中减去该词对所有对的贡献
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] -= count
                pair_to_words[pair].discard(word_id)
                if pair_counts[pair] <= 0 and pair in pair_counts:
                    del pair_counts[pair]

            # 对该词应用合并，构建新 token 列表
            new_tokens: list[bytes] = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            words[word_id] = (new_tokens, count)

            # 将新词对的贡献加回 pair_counts
            for i in range(len(new_tokens) - 1):
                pair = (new_tokens[i], new_tokens[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count
                pair_to_words[pair].add(word_id)

        # 清理已合并对的索引
        if best_pair in pair_to_words:
            del pair_to_words[best_pair]

    return vocab, merges


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    返回 GPT-2 的字节到 Unicode 字符映射表。
    GPT-2 用可打印 Unicode 字符表示所有 256 个字节值，
    以便在 JSON 中存储词表而不引入控制字符。
    """
    # bpe编码器核心部分编解码对特殊符号使用正常编解码，然而文本存储时无法存储特殊符号，这里是将特殊符号映射为可打印字符
    # 比如空格、换行等特殊符号，剩下的字节值则直接映射为对应的 Unicode 字符，这样保存词表时就能保存为txt等格式。
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class Tokenizer:
    """
    字节级 BPE 分词器，支持文本编码（str -> List[int]）和解码（List[int] -> str）。

    编码流程:
    1. 按特殊 token 分割文本（特殊 token 保持完整，不拆分）
    2. 对普通文本段用 GPT-2 正则进行预分词
    3. 对每个 pre-token 应用 BPE 合并（按合并创建顺序）
    4. 将字节序列映射为整数 token ID

    解码: 将 token ID 还原为字节，再解码为 Unicode 字符串。
    遇到无效 UTF-8 字节序列时，用 U+FFFD 替换字符代替。
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        初始化分词器。

        Args:
            vocab: token ID -> token 字节串 的映射
            merges: BPE 合并列表，按创建顺序排列
            special_tokens: 特殊 token 列表，这些字符串永远不会被拆分
        """
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens) if special_tokens else []

        # 构建反向查找: 字节串 -> token ID
        self.bytes_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        #注：这说明总共有三个表，一个是vocab，一个是merges，最后一个是bytes_to_id

        # 将不在词表中的特殊 token 追加到词表末尾
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.bytes_to_id:
                new_id = max(self.vocab.keys()) + 1 if self.vocab else 0
                self.vocab[new_id] = token_bytes
                self.bytes_to_id[token_bytes] = new_id

        # 合并优先级: merge -> rank (rank 越小 = 越早应用)
        self.merge_rank: dict[tuple[bytes, bytes], int] = {
            merge: rank for rank, merge in enumerate(self.merges)
        }

        # 构建特殊 token 分割正则（按长度降序，确保最长匹配优先）
        if self.special_tokens:
            escaped = [re.escape(st) for st in sorted(self.special_tokens, key=len, reverse=True)]
            self.special_token_pattern = re.compile(f"({'|'.join(escaped)})")
        else:
            self.special_token_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        从序列化的词表和合并文件加载分词器。

        文件格式与 GPT-2/tiktoken 兼容：
        - vocab_filepath: JSON 文件，键为 GPT-2 编码的 token 字符串，值为整数 ID
        - merges_filepath: 文本文件，每行一个合并 "token1 token2"

        Args:
            vocab_filepath: 词表 JSON 文件路径
            merges_filepath: 合并文件路径
            special_tokens: 可选的特殊 token 列表

        Returns:
            Tokenizer 实例
        """
        gpt2_byte_decoder = {v: k for k, v in _gpt2_bytes_to_unicode().items()}

        with open(vocab_filepath, encoding="utf-8") as f:
            gpt2_vocab = json.load(f)

        vocab = {
            int(v): bytes([gpt2_byte_decoder[c] for c in k])
            for k, v in gpt2_vocab.items()
        } # 注意这里的bytes能直接把list合成回一个单独的字节串而不是一个字节列表，这样就能正确构建vocab表了

        with open(merges_filepath, encoding="utf-8") as f:
            lines = f.readlines()

        merges = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(" ")
            if len(parts) == 2:
                token1 = bytes([gpt2_byte_decoder[c] for c in parts[0]])
                token2 = bytes([gpt2_byte_decoder[c] for c in parts[1]])
                merges.append((token1, token2))

        return cls(vocab, merges, special_tokens)

    def _apply_bpe(self, word_bytes: list[bytes]) -> list[bytes]:
        """
        对一个字节列表应用 BPE 合并。

        使用贪心策略：每次找优先级最高（rank 最小）的可用合并并应用，
        直到没有更多可用合并为止。这等价于按合并顺序重播 BPE 训练过程。

        Args:
            word_bytes: 单字节组成的列表，如 [b'h', b'e', b'l', b'l', b'o']

        Returns:
            合并后的字节列表，如 [b'he', b'llo']
        """
        if len(word_bytes) <= 1:
            return word_bytes

        tokens = list(word_bytes)

        while len(tokens) > 1:
            # 找优先级最高（rank 最小）的相邻对
            best_rank = float("inf")
            best_idx = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_rank.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i

            if best_idx == -1 or best_rank == float("inf"):
                break  # 没有更多可用合并

            # 应用此合并
            merged = tokens[best_idx] + tokens[best_idx + 1]
            new_tokens: list[bytes] = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == tokens[best_idx]
                    and tokens[i + 1] == tokens[best_idx + 1]
                    and i == best_idx
                ):
                    # 只合并第一个找到的最佳对（然后重新扫描）
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def _apply_bpe_fast(self, word_bytes: list[bytes]) -> list[bytes]:
        """
        更高效的 BPE 应用：每次应用一种合并，直到全部应用。

        与 _apply_bpe 不同，这里每次找到最高优先级合并后，
        将所有出现位置都替换，然后继续。
        """
        if len(word_bytes) <= 1:
            return word_bytes

        tokens = list(word_bytes)

        while len(tokens) > 1:
            # 找优先级最高的对
            best_pair = None
            best_rank = float("inf")

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_rank.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None or best_rank == float("inf"):
                break

            # 将所有出现的 best_pair 都合并
            merged = best_pair[0] + best_pair[1]
            new_tokens: list[bytes] = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str) -> list[int]:
        """
        将文本字符串编码为 token ID 序列。

        处理特殊 token 的关键：先按特殊 token 分割文本，
        确保特殊 token 不会被进一步拆分，始终映射为单个整数 ID。

        Args:
            text: 输入文本字符串

        Returns:
            整数 token ID 列表
        """
        token_ids: list[int] = []

        if self.special_token_pattern:
            parts = self.special_token_pattern.split(text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue
            # 检查是否为特殊 token
            if part in set(self.special_tokens):
                token_id = self.bytes_to_id[part.encode("utf-8")]
                token_ids.append(token_id)
            else:
                # 普通文本：预分词 + BPE
                for match in regex.finditer(PAT, part):
                    word = match.group(0)
                    word_bytes = [bytes([b]) for b in word.encode("utf-8")]
                    merged_bytes = self._apply_bpe_fast(word_bytes)
                    for token_bytes in merged_bytes:
                        token_ids.append(self.bytes_to_id[token_bytes])

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        对字符串可迭代对象进行惰性编码，逐个产出 token ID。

        内存效率高：适用于无法整体加载到内存的大文件。
        通过逐行处理（或逐块处理）避免内存爆炸。

        Args:
            iterable: 字符串可迭代对象（如文件句柄）

        Yields:
            整数 token ID
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        将 token ID 序列解码回文本字符串。

        解码方法: 拼接所有 token 的字节串，再解码为 Unicode。
        对于无效 UTF-8 序列（如用户传入任意整数序列时可能出现），
        使用 errors='replace' 用 U+FFFD 替换字符替代，而非抛出异常。

        Args:
            ids: 整数 token ID 列表

        Returns:
            解码后的字符串，无效字节用 U+FFFD 替换
        """
        all_bytes = b"".join(self.vocab[id_] for id_ in ids)
        return all_bytes.decode("utf-8", errors="replace")
