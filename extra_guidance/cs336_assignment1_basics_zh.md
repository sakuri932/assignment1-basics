# CS336 作业 1（基础）：从零构建 Transformer 语言模型

**版本 26.0.3**  
CS336 教学团队  
2026 年春季

---

## 1 作业概述

在本作业中，你将从零开始构建训练标准 Transformer 语言模型（LM）所需的全部组件，并训练若干模型。

### 你需要实现的内容

1. 字节对编码（BPE）分词器（第 2 节）
2. Transformer 语言模型（第 3 节）
3. 交叉熵损失函数与 AdamW 优化器（第 4 节）
4. 训练循环，支持序列化和加载模型及优化器状态（第 5 节）

### 你需要运行的内容

1. 在 TinyStories 数据集上训练 BPE 分词器。
2. 使用训练好的分词器对数据集进行编码，将其转换为整数 ID 序列。
3. 在 TinyStories 数据集上训练 Transformer LM。
4. 使用训练好的 Transformer LM 生成样本并评估困惑度。
5. 在 OpenWebText 上训练模型，并将所获得的困惑度提交至排行榜。

### 你可以使用的内容

我们期望你从零开始构建每个组件。特别是，除以下内容外，你**不得**使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的任何定义：

- `torch.nn.Parameter`
- `torch.nn` 中的容器类（例如 `Module`、`ModuleList`、`Sequential` 等）。¹
- `torch.optim.Optimizer` 基类

你可以使用其他任何 PyTorch 定义。如果你想使用某个函数或类但不确定是否被允许，欢迎在 Slack 上提问。如有疑问，请考虑使用它是否有悖于本作业"从零开始"的精神。

> ¹完整列表见 pytorch.org/docs/stable/nn.html#containers。

### 关于 AI 工具的声明

AI 可以完全自主地解决作业的许多部分，这使得深入参与并从课程材料中学习变得更加困难。

允许使用 AI 工具回答高层次的概念性问题，或提供低层次的编程文档（如函数签名和库 API）。但是，**不允许**使用 AI 工具实现任何作业的任何部分。这包括代码智能体（如 Cursor Agents、Codex、Claude Code）和 AI 自动补全（如 Cursor Tab、GitHub Copilot）。使用 AI 智能体时，请确保其使用所提供的 `AGENTS.md` 文件；使用聊天机器人时，也应包含提示词（prompt）。

我们强烈建议你在完成作业时禁用 IDE 中的 AI 自动补全（非 AI 自动补全，例如自动补全函数名是完全可以的）。以往的学生反映，禁用 AI 自动补全有助于更深入地理解材料。

完整的 AI 政策请参阅[该文档](https://ai-policy-link)。

### 代码结构

作业代码及本文档均可在 GitHub 上获取：

[github.com/stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)

请使用 `git clone` 克隆仓库。如有更新，我们会通知你，你可以使用 `git pull` 获取最新内容。

1. `cs336_basics/*`：这是你编写代码的地方。注意此目录下没有任何代码——你可以完全从零开始！
2. `adapters.py`：你的代码需要具备一系列功能。对于每项功能（例如缩放点积注意力），通过简单调用你的代码来填写其实现（例如 `run_scaled_dot_product_attention`）。注意：你对 `adapters.py` 的修改不应包含任何实质性逻辑；这只是"胶水代码"。
3. `test_*.py`：包含你必须通过的所有测试（例如 `test_scaled_dot_product_attention`），这些测试将调用 `adapters.py` 中定义的钩子。请勿编辑测试文件。

### 如何提交

提交时，运行 `make_submission.sh` 生成提交 zip 文件。如果你有不想包含在提交 zip 中的大型数据文件或检查点，请务必将它们添加到脚本的排除列表中。

你需要向 Gradescope 提交以下文件：

- `writeup.pdf`：回答所有书面问题，请使用排版工具。
- `code.zip`：包含你编写的所有代码。

要提交至排行榜，请向以下仓库提交 PR：

[github.com/stanford-cs336/assignment1-basics-leaderboard](https://github.com/stanford-cs336/assignment1-basics-leaderboard)

详细的提交说明见排行榜仓库的 README.md。

### 数据集获取

本作业将使用两个预处理数据集：TinyStories [R. Eldan 等, 2023] 和 OpenWebText [A. Gokaslan 等, 2019]。两个数据集均为单个大型纯文本文件。

如果你随班级一起完成本作业，可在计算指南中找到下载数据集的说明。

如果你在家自学，可以使用 README.md 中的命令下载这些文件。

---

> **低资源提示：初始化**
>
> 在课程作业讲义中，我们将为使用较少或无 GPU 资源完成作业提供建议。例如，我们有时会建议缩小数据集或模型规模，或说明如何在 Mac 集成 GPU 或 CPU 上运行训练代码。你会在蓝色框（如本框）中看到这些"低资源提示"。即使你是有权访问课程机器的在校斯坦福学生，这些提示也可能帮助你更快地迭代并节省时间，因此我们建议阅读！

---

> **低资源提示：在 Apple Silicon 或 CPU 上完成作业 1**
>
> 使用工作人员的解决方案代码，我们可以在配备 36 GB RAM 的 Apple M4 Max 芯片上训练一个能生成相当流畅文本的 LM：在 Metal GPU（MPS）上不到 5 分钟，在 CPU 上约 30 分钟。如果这些词对你来说没有太多意义，不用担心！只需知道，如果你有一台相当新的笔记本电脑，且你的实现正确且高效，你将能够训练一个能生成简单儿童故事（且具有相当流畅性）的小型 LM。
>
> 在作业的后续部分，我们将说明如果你使用 CPU 或 MPS 需要做哪些修改。

---

## 2 字节对编码（BPE）分词器

在作业的第一部分，我们将训练并实现一个字节级字节对编码（BPE）分词器 [R. Sennrich 等, 2016; C. Wang 等, 2019]。具体来说，我们将任意（Unicode）字符串表示为字节序列，并在此字节序列上训练 BPE 分词器。之后，我们将使用该分词器将文本（字符串）编码为用于语言建模的 token（整数序列）。

### 2.1 Unicode 标准

Unicode 是一种将字符映射到整数码点的文本编码标准。截至 Unicode 17.0（2025 年 9 月发布），该标准定义了跨 172 种文字的 159,801 个字符。例如，字符 `"s"` 的码点为 115（通常记作 U+0073，其中 U+ 是惯用前缀，0073 是 115 的十六进制表示），字符 `"牛"` 的码点为 29275。在 Python 中，你可以使用 `ord()` 函数将单个 Unicode 字符转换为其整数表示，`chr()` 函数则将整数 Unicode 码点转换为对应字符的字符串。

```python
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

---

> **问题（unicode1）：理解 Unicode（1 分）**
>
> (a) `chr(0)` 返回什么 Unicode 字符？
> **交付物**：一句话回答。
>
> (b) 该字符的字符串表示（`__repr__()`）与其打印表示有何不同？
> **交付物**：一句话回答。
>
> (c) 当该字符出现在文本中时会发生什么？在 Python 解释器中试试以下代码，看看是否符合你的预期：
>
> ```python
> >>> chr(0)
> >>> print(chr(0))
> >>> "this is a test" + chr(0) + "string"
> >>> print("this is a test" + chr(0) + "string")
> ```
>
> **交付物**：一句话回答。

---

### 2.2 Unicode 编码

虽然 Unicode 标准定义了字符到码点（整数）的映射，但直接在 Unicode 码点上训练分词器是不切实际的，因为词汇表会非常庞大（约 150K 条目）且稀疏（许多字符极为罕见）。因此，我们将使用 Unicode 编码，将 Unicode 字符转换为字节序列。Unicode 标准本身定义了三种编码：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网上最主流的编码（超过 98% 的网页使用）。

要将 Unicode 字符串编码为 UTF-8，可以使用 Python 中的 `encode()` 函数。要访问 Python `bytes` 对象的底层字节值，可以对其进行迭代（例如调用 `list()`）。最后，可以使用 `decode()` 函数将 UTF-8 字节字符串解码为 Unicode 字符串。

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # 获取编码字符串的字节值（0 到 255 的整数）
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129,
161, 227, 129, 175, 33]
>>> # 一个字节不一定对应一个 Unicode 字符！
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

通过将 Unicode 码点转换为字节序列（例如通过 UTF-8 编码），我们实质上是将一系列码点（21 位整数，有 159,801 个有效值）转换为字节值序列（0 到 255 的整数）。256 长度的字节词汇表更易于处理。使用字节级分词时，我们无需担心未登录词（out-of-vocabulary）的问题，因为我们知道任何输入文本都可以表示为 0 到 255 的整数序列。

---

> **问题（unicode2）：Unicode 编码（3 分）**
>
> (a) 相比 UTF-16 或 UTF-32，有哪些理由更倾向于在 UTF-8 编码的字节上训练分词器？比较各种输入字符串下这些编码的输出可能会有所帮助。
> **交付物**：一到两句话回答。
>
> (b) 考虑以下（错误的）函数，其本意是将 UTF-8 字节字符串解码为 Unicode 字符串。为何该函数是错误的？请提供一个产生错误结果的输入字节字符串示例。
>
> ```python
> def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
>     return "".join([bytes([b]).decode("utf-8") for b in bytestring])
> >>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
> 'hello'
> ```
>
> **交付物**：一个使 `decode_utf8_bytes_to_str_wrong` 产生错误输出的示例输入字节字符串，以及一句话解释该函数为何错误。
>
> (c) 给出一个无法解码为任何 Unicode 字符的两字节序列。
> **交付物**：一个示例，以及一句话解释。

---

### 2.3 子词分词

虽然字节级分词可以缓解词级分词器面临的未登录词问题，但将文本分词为字节会产生极长的输入序列。这会减慢模型训练速度，因为在词级语言模型中，一个 10 个词的句子可能只有 10 个 token，但在字符级模型中可能有 50 个或更多 token（取决于词的长度）。处理这些更长的序列在模型的每个步骤都需要更多计算。此外，在字节序列上进行语言建模很困难，因为更长的输入序列会在数据中产生长程依赖关系。

子词分词是词级分词器和字节级分词器之间的折中方案。注意，字节级分词器的词汇表有 256 个条目（字节值为 0 到 255）。子词分词器以更大的词汇量换取对输入字节序列更好的压缩率。例如，如果字节序列 `b'the'` 在我们的原始训练数据中频繁出现，为其分配一个词汇表条目将把这个 3 token 序列缩减为单个 token。

我们如何选择要添加到词汇表中的子词单元？R. Sennrich 等 [3] 提出使用字节对编码（BPE；P. Gage [5]），这是一种压缩算法，它迭代地将最频繁的字节对替换（"合并"）为一个新的未使用索引。注意，该算法向词汇表中添加子词 token 以最大化对输入序列的压缩——如果某个词在输入文本中出现足够多次，它将被表示为单个子词单元。

通过 BPE 构建词汇表的子词分词器通常称为 BPE 分词器。在本作业中，我们将实现一个字节级 BPE 分词器，其中词汇表项是字节或合并后的字节序列，兼具未登录词处理和可管理的输入序列长度两方面的优势。构建 BPE 分词器词汇表的过程称为"训练"BPE 分词器。

### 2.4 BPE 分词器训练

BPE 分词器训练过程包含三个主要步骤。

#### 词汇表初始化

分词器词汇表是字节串 token 到整数 ID 的一一映射。由于我们训练的是字节级 BPE 分词器，初始词汇表就是所有字节的集合。因为有 256 种可能的字节值，初始词汇表大小为 256。

#### 预分词

拥有词汇表后，原则上你可以统计语料库中字节相邻出现的频率，然后从最频繁的字节对开始合并。然而，这计算代价很高，因为每次合并都需要对整个语料库进行完整扫描。此外，直接在语料库中跨词合并字节可能会产生仅在标点上不同的 token（例如 `dog!` 与 `dog.`）。这些 token 将获得完全不同的 token ID，尽管它们在语义上高度相似（仅在标点上不同）。

为了避免这种情况，我们对语料库进行预分词。你可以将其理解为对语料库进行粗粒度分词，帮助我们统计字符对出现的频率。例如，单词 `'text'` 可能是一个出现 10 次的预 token。在这种情况下，当我们统计字符 `'t'` 和 `'e'` 相邻出现的次数时，我们会看到单词 `'text'` 中 `'t'` 和 `'e'` 相邻，可以将其计数增加 10，而不必遍历整个语料库。由于我们训练的是字节级 BPE 模型，每个预 token 表示为 UTF-8 字节序列。

R. Sennrich 等 [3] 的原始 BPE 实现通过简单地在空白处分割（即 `s.split(" ")`）来进行预分词。这种方法仍在基于 SentencePiece 的分词器中使用（例如 Llama 1 和 2 的分词器）。

大多数现代分词器使用基于正则表达式的预分词器，这一做法源自 GPT-2 [A. Radford 等, 2019]。我们将使用从以下链接获取的原始正则表达式的稍微美化版本：

[github.com/openai/tiktoken/pull/234/files](https://github.com/openai/tiktoken/pull/234/files)

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

交互式地用这个预分词器拆分一些文本可能有助于更好地理解其行为：

```python
>>> # 需要 `regex` 包
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

然而，在代码中使用时，应该使用 `re.finditer` 而非将预分词后的词存储为列表，以便在构建预 token 到计数的映射时避免存储预分词结果。

#### 计算 BPE 合并

将输入文本转换为预 token 并将每个预 token 表示为 UTF-8 字节序列后，我们可以计算 BPE 合并（即训练 BPE 分词器）。从高层次来看，BPE 算法迭代地统计每个字节对并确定频率最高的对 `("A", "B")`。然后，将该最频繁对 `("A", "B")` 的每次出现都进行合并，即替换为新 token `"AB"`。这个新的合并 token 被添加到词汇表中；因此，BPE 训练后的最终词汇表大小等于初始词汇表大小（本例中为 256）加上训练期间执行的 BPE 合并操作次数。为了提高 BPE 训练效率，我们不考虑跨越预 token 边界的字节对。² 计算合并时，通过优先选择词典序更大的对来确定性地打破频率并列的情况。例如，如果对 `("A", "B")`、`("A", "C")`、`("B", "ZZ")` 和 `("BA", "A")` 都具有最高频率，我们将合并 `("BA", "A")`：

```python
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

#### 特殊 token

通常，某些字符串（例如 `<|endoftext|>`）用于编码元数据（例如文档之间的边界）。在编码文本时，通常希望将某些字符串视为"特殊 token"，这些 token 不应被拆分为多个 token（即始终保留为单个 token）。例如，序列结束字符串 `<|endoftext|>` 应始终保留为单个 token（即单个整数 ID），以便我们知道何时停止从语言模型生成。这些特殊 token 必须添加到词汇表中，以便它们具有相应的固定 token ID。

R. Sennrich 等 [3] 的算法 1 包含一个低效的 BPE 分词器训练实现（本质上遵循我们上面概述的步骤）。作为第一个练习，实现并测试这个函数以检验你的理解可能很有用。

> ² 注意，R. Sennrich 等 [3] 的原始 BPE 公式规定包含词尾 token。我们在训练字节级 BPE 模型时不添加词尾 token，因为所有字节（包括空白和标点符号）都包含在模型词汇表中。由于我们明确表示空格和标点符号，学习到的 BPE 合并将自然反映这些词边界。

---

> **示例（bpe_example）：BPE 训练示例**
>
> 以下是 R. Sennrich 等 [3] 的一个风格化示例。考虑由以下文本组成的语料库：
>
> ```
> low low low low low
> lower lower widest widest widest
> newest newest newest newest newest newest
> ```
>
> 词汇表中有一个特殊 token `<|endoftext|>`。
>
> **词汇表**
>
> 我们用特殊 token `<|endoftext|>` 和 256 个字节值初始化词汇表。
>
> **预分词**
>
> 为简单起见，我们在此示例中假设预分词只是按空白分割。预分词并统计后，得到频率表：
>
> ```python
> {low: 5, lower: 2, widest: 3, newest: 6}
> ```
>
> 方便起见，将其表示为 `dict[tuple[bytes, ...], int]`，例如 `{(l,o,w): 5, …}`。注意，Python 中即使是单个字节也是 `bytes` 对象。Python 中没有 `byte` 类型来表示单个字节，就像 Python 中没有 `char` 类型来表示单个字符一样。
>
> **合并**
>
> 我们首先查看每对相邻字节并对它们出现的词的频率求和：`{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}`。对 `('e', 's')` 和 `('s', 't')` 并列，因此我们取词典序更大的对 `('s', 't')`。然后合并预 token，得到 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}`。
>
> 第二轮中，`(e, st)` 是最常见的对（计数为 9），合并后得到 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}`。继续这个过程，最终得到的合并序列为 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']`。
>
> 如果我们取 6 次合并，得到 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']`，词汇表元素为 `[<|endoftext|>, [...256 字节字符], st, est, ow, low, west, ne]`。
>
> 使用此词汇表和合并集，单词 `newest` 将被分词为 `[ne, west]`。

---

### 2.5 BPE 分词器训练实验

让我们在 TinyStories 数据集上训练字节级 BPE 分词器。数据集的获取/下载说明见第 1 节。开始之前，我们建议先浏览 TinyStories 数据集，了解其内容。

#### 并行化预分词

你会发现预分词步骤是主要瓶颈。可以使用内置库 `multiprocessing` 并行化代码来加速预分词。具体来说，我们建议在并行实现预分词时，将语料库分块，并确保块的边界出现在特殊 token 的开头处。你可以直接使用以下链接的启动代码来获取块边界，然后将工作分配到各个进程：

[https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py)

这种分块方式始终有效，因为我们永远不希望跨文档边界进行合并。就本作业而言，你始终可以这样分割。不用担心接收到不含 `<|endoftext|>` 的超大语料库的边界情况。

#### 预分词前移除特殊 token

在使用正则表达式模式运行预分词（使用 `re.finditer`）之前，你应该从语料库（或块，如果使用并行实现）中去除所有特殊 token。确保在特殊 token 处分割，这样就不会在它们分隔的文本之间发生合并。例如，如果语料库（或块）类似于 `[文档 1]<|endoftext|>[文档 2]`，你应该在特殊 token `<|endoftext|>` 处分割，分别对 `[文档 1]` 和 `[文档 2]` 进行预分词，这样就不会在文档边界处发生合并。换言之，特殊 token 在训练期间定义了硬分割边界，但它们本身不应贡献合并计数。可以使用 `re.split` 以 `"|".join(special_tokens)` 作为分隔符来实现（注意谨慎使用 `re.escape`，因为 `|` 可能出现在特殊 token 中）。测试 `test_train_bpe_special_tokens` 将对此进行测试。

#### 优化合并步骤

上述风格化示例中 BPE 训练的朴素实现很慢，因为每次合并都需要遍历所有字节对以找出最频繁的对。然而，每次合并后只有与合并对重叠的对的计数会发生变化。因此，可以通过索引所有对的计数并增量更新这些计数来提高 BPE 训练速度，而不是显式遍历每对字节来统计对频率。这种缓存过程可以获得显著的加速，但请注意 BPE 训练的合并部分在 Python 中是不可并行化的。

---

> **低资源提示：性能分析**
>
> 你应该使用 `cProfile` 或 `py-spy` 等性能分析工具来识别实现中的瓶颈，并专注于优化这些部分。

---

> **低资源提示："缩小规模"**
>
> 不要直接在完整的 TinyStories 数据集上训练分词器，我们建议先在一小部分数据上训练：一个"调试数据集"。例如，可以使用 TinyStories 验证集（22K 文档，而非 2.12M）进行训练。这体现了一种通用策略：尽可能缩小规模以加快开发速度——例如使用更小的数据集、更小的模型规模等。选择调试数据集的大小或超参数配置需要仔细考量：你希望调试集足够大，以便与完整配置具有相同的瓶颈（从而使你的优化能够推广），但又不能太大以至于运行时间过长。

---

> **问题（train_bpe）：BPE 分词器训练（15 分）**
>
> **交付物**：编写一个函数，给定输入文本文件的路径，训练一个（字节级）BPE 分词器。你的 BPE 训练函数应至少处理以下输入参数：
>
> **输入**
> - `input_path: str`  BPE 分词器训练数据的文本文件路径。
> - `vocab_size: int`  定义最终词汇表最大大小的正整数（包括初始字节词汇表、合并产生的词汇表项以及任何特殊 token）。
> - `special_tokens: list[str]`  要添加到词汇表的字符串列表。训练期间，将它们视为阻止跨越其范围合并的硬边界，但不应将其纳入合并统计计算。
>
> 你的 BPE 训练函数应返回结果词汇表和合并列表：
>
> **输出**
> - `vocab: dict[int, bytes]`  分词器词汇表，从整数（词汇表中的 token ID）到字节（token 字节）的映射。
> - `merges: list[tuple[bytes, bytes]]`  训练产生的 BPE 合并列表。每个列表项是一个字节元组 `(<token1>, <token2>)`，表示 `<token1>` 与 `<token2>` 进行了合并。合并应按创建顺序排列。
>
> 要对照提供的测试检验你的 BPE 训练函数，首先需要在 `[adapters.run_train_bpe]` 实现测试适配器，然后运行 `uv run pytest tests/test_train_bpe.py`。你的实现应能通过所有测试。（可选）你可以使用某些系统语言（例如 C++ 使用 cppyy 或 nanobind，或 Rust 使用 PyO3）实现训练方法的关键部分。如果这样做，请注意哪些操作需要复制 Python 内存而哪些可以直接读取，并确保留下构建说明，或确保仅使用 `pyproject.toml` 即可构建。还要注意，GPT-2 正则表达式在大多数正则引擎中支持不佳，运行速度会太慢。我们已验证 Oniguruma 速度合理且支持否定前瞻，但 Python 的 `regex` 包甚至更快。

---

> **问题（train_bpe_tinystories）：在 TinyStories 上训练 BPE（2 分）**
>
> (a) 在 TinyStories 数据集上训练字节级 BPE 分词器，最大词汇表大小为 10,000。确保将 TinyStories 的 `<|endoftext|>` 特殊 token 添加到词汇表中。将结果词汇表和合并序列化到磁盘以供进一步检查。训练花了多长时间和多少内存？词汇表中最长的 token 是什么？它有意义吗？
>
> **资源要求**：≤30 分钟（无 GPU），≤30 GB RAM
>
> **提示**：使用预分词期间的多进程处理以及以下两个事实，你应该能将 BPE 训练时间控制在 2 分钟以内：
> - `<|endoftext|>` token 用于分隔数据文件中的文档。
> - `<|endoftext|>` token 在 BPE 合并应用之前作为特殊情况处理。
>
> **交付物**：一到两句话回答。
>
> (b) 对你的代码进行性能分析。分词器训练过程中哪个部分耗时最多？
> **交付物**：一到两句话回答。

---

接下来，我们将尝试在 OpenWebText 数据集上训练字节级 BPE 分词器。与之前一样，我们建议先浏览数据集以更好地理解其内容。

---

> **问题（train_bpe_expts_owt）：在 OpenWebText 上训练 BPE（2 分）**
>
> (a) 在 OpenWebText 数据集上训练字节级 BPE 分词器，最大词汇表大小为 32,000。将结果词汇表和合并序列化到磁盘以供进一步检查。词汇表中最长的 token 是什么？它有意义吗？
>
> **资源要求**：≤12 小时（无 GPU），≤100 GB RAM
>
> **交付物**：一到两句话回答。
>
> (b) 比较在 TinyStories 和 OpenWebText 上训练得到的分词器，分析异同。
> **交付物**：一到两句话回答。

---

### 2.6 BPE 分词器：编码与解码

在作业的前一部分，我们实现了一个函数，用于在输入文本上训练 BPE 分词器，得到分词器词汇表和 BPE 合并列表。现在，我们将实现一个 BPE 分词器，它加载提供的词汇表和合并列表，并使用它们将文本编码为 token ID 及将 token ID 解码为文本。

#### 2.6.1 文本编码

BPE 编码文本的过程与我们训练 BPE 词汇表的方式相对应，主要包含以下几个步骤。

**步骤 1：预分词。** 我们首先对序列进行预分词，并将每个预 token 表示为 UTF-8 字节序列，就像在 BPE 训练中所做的那样。我们将在每个预 token 内将这些字节合并为词汇表元素，独立处理每个预 token（不跨预 token 边界合并）。

**步骤 2：应用合并。** 然后，按照 BPE 训练期间创建的词汇表元素合并序列，按相同的创建顺序应用到我们的预 token 上。

---

> **示例（bpe_encoding）：BPE 编码示例**
>
> 例如，假设输入字符串为 `'the cat ate'`，词汇表为 `{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}`，学到的合并为 `[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]`。首先，预分词器将字符串拆分为 `['the', ' cat', ' ate']`。然后，对每个预 token 应用 BPE 合并。
>
> 第一个预 token `'the'` 初始表示为 `[b't', b'h', b'e']`。查看合并列表，找到第一个适用的合并 `(b't', b'h')`，将预 token 转换为 `[b'th', b'e']`。然后回到合并列表，找到下一个适用的合并 `(b'th', b'e')`，将预 token 转换为 `[b'the']`。最后，查看合并列表，发现没有更多适用的合并（整个预 token 已合并为单个 token），处理完成。对应的整数序列为 `[9]`。
>
> 对其余预 token 重复此过程：预 token `' cat'` 应用 BPE 合并后表示为 `[b' c', b'a', b't']`，对应整数序列 `[7, 1, 5]`。最后的预 token `' ate'` 应用 BPE 合并后为 `[b' at', b'e']`，对应整数序列 `[10, 3]`。因此，编码输入字符串的最终结果为 `[9, 7, 1, 5, 10, 3]`。

---

**特殊 token**

你的分词器应能在编码文本时正确处理用户定义的特殊 token（在构建分词器时提供）。

**内存注意事项**

假设我们要对一个无法完全加载到内存的大型文本文件进行分词。为了高效地分词这个大型文件（或任何其他数据流），我们需要将其分割成可管理的块并依次处理，使内存复杂度为常数而非与文本大小线性相关。在此过程中，我们需要确保 token 不会跨越块边界，否则将得到与朴素方法（将整个序列加载到内存中分词）不同的分词结果。

#### 2.6.2 文本解码

要将整数 token ID 序列解码回原始文本，只需查找每个 ID 在词汇表中对应的条目（字节序列），将它们连接在一起，然后将字节解码为 Unicode 字符串。注意，输入 ID 不保证映射到有效的 Unicode 字符串（因为用户可能输入任意整数 ID 序列）。如果输入 token ID 无法产生有效的 Unicode 字符串，你应该用官方 Unicode 替换字符 U+FFFD 替换格式错误的字节。³ `bytes.decode` 的 `errors` 参数控制如何处理 Unicode 解码错误，使用 `errors='replace'` 将自动用替换标记替换格式错误的数据。

> ³ 关于 Unicode 替换字符的更多信息，见 en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character。

---

> **问题（tokenizer）：实现分词器（15 分）**
>
> **交付物**：实现一个 `Tokenizer` 类，给定词汇表和合并列表，将文本编码为整数 ID，并将整数 ID 解码为文本。你的分词器还应支持用户提供的特殊 token（如果尚未存在，则将其追加到词汇表中）。建议使用以下接口：
>
> `def __init__(self, vocab, merges, special_tokens=None)` 从给定的词汇表、合并列表和（可选的）特殊 token 列表构造分词器。此函数应接受以下参数：
> - `vocab: dict[int, bytes]`
> - `merges: list[tuple[bytes, bytes]]`
> - `special_tokens: list[str] | None = None`
>
> `def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)` 类方法，从序列化的词汇表和合并列表（与 BPE 训练代码输出格式相同）以及（可选的）特殊 token 列表构造并返回 `Tokenizer`。此方法应接受以下附加参数：
> - `vocab_filepath: str`
> - `merges_filepath: str`
> - `special_tokens: list[str] | None = None`
>
> `def encode(self, text: str) -> list[int]` 将输入文本编码为 token ID 序列。
>
> `def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]` 给定一个可迭代的字符串（例如 Python 文件句柄），返回一个懒惰地生成 token ID 的生成器。这是对无法直接加载到内存的大型文件进行内存高效分词所必需的。
>
> `def decode(self, ids: list[int]) -> str` 将 token ID 序列解码为文本。
>
> 要对照提供的测试检验你的 `Tokenizer`，首先需要在 `[adapters.get_tokenizer]` 实现测试适配器，然后运行 `uv run pytest tests/test_tokenizer.py`。你的实现应能通过所有测试。

---

### 2.7 实验

---

> **问题（tokenizer_experiments）：分词器实验（4 分）**
>
> (a) 从 TinyStories 和 OpenWebText 各抽取 10 个文档。使用之前训练的 TinyStories 和 OpenWebText 分词器（词汇表大小分别为 10K 和 32K），将这些抽取的文档编码为整数 ID。每个分词器的压缩比（字节/token）是多少？
> **交付物**：一到两句话回答。
>
> (b) 如果用 TinyStories 分词器对 OpenWebText 样本进行分词会发生什么？比较压缩比和/或定性描述发生的情况。
> **交付物**：一到两句话回答。
>
> (c) 估计你的分词器的吞吐量（例如以字节/秒为单位）。对 Pile 数据集（825GB 文本）进行分词需要多长时间？
> **交付物**：一到两句话回答。
>
> (d) 使用你的 TinyStories 和 OpenWebText 分词器，将各自的训练集和开发集编码为整数 token ID 序列。我们将在后续训练语言模型时使用这些数据。建议将 token ID 序列化为 NumPy `uint16` 数据类型的数组。为什么 `uint16` 是合适的选择？
> **交付物**：一到两句话回答。

---

## 3 Transformer 语言模型架构

语言模型接受批量整数 token ID 序列（即形状为 `(batch_size, sequence_length)` 的 `torch.Tensor`）作为输入，并返回词汇表上的（批量）归一化概率分布（即形状为 `(batch_size, sequence_length, vocab_size)` 的 PyTorch 张量），其中预测分布是对每个输入 token 的下一个词的预测。训练语言模型时，我们使用这些下一词预测来计算实际下一词与预测下一词之间的交叉熵损失。在推理期间从语言模型生成文本时，我们从最后一个时间步（即序列中的最后一项）的预测下一词分布中生成下一个 token（例如取概率最高的 token、从分布中采样等），将生成的 token 添加到输入序列，然后重复此过程。

在作业的这一部分，你将从零开始构建这个 Transformer 语言模型。我们将从模型的高层描述开始，然后逐步详细说明各个组件。

### 3.1 Transformer LM

给定 token ID 序列，Transformer 语言模型使用输入嵌入将 token ID 转换为密集向量，将嵌入后的 token 通过 `num_layers` 个 Transformer 块，然后应用学习的线性投影（"输出嵌入"或"LM 头"）来产生预测的下一 token logit。见图 1 的示意图。

```
输入
  |
Token 嵌入层
  |
Transformer Block
...（共 num_layers 个）
Transformer Block
  |
Norm（归一化）
  |
Linear（输出嵌入）
  |
Softmax
  |
输出概率
```

**图 1**：Transformer 语言模型概览。

```
输入张量，形状为 (batch_size, seq_len, d_model)
  |
Norm → 因果多头自注意力（含 RoPE）
  |  +---- 残差连接 ---|
  |
Norm → 位置前馈网络
  |  +---- 残差连接 ---|
  |
输出张量，形状为 (batch_size, seq_len, d_model)
```

**图 2**：前置归一化（pre-norm）Transformer 块。

#### Token 嵌入

第一步，Transformer 将（批量）token ID 序列嵌入为包含 token 身份信息的向量序列（图 1 中的红色块）。

更具体地说，给定 token ID 序列，Transformer 语言模型使用 token 嵌入层生成向量序列。每个嵌入层接受形状为 `(batch_size, sequence_length)` 的整数张量，并生成形状为 `(batch_size, sequence_length, d_model)` 的向量序列。

#### 前置归一化 Transformer 块

嵌入后，激活值由若干结构相同的神经网络层处理。标准的仅解码器 Transformer 语言模型由 `num_layers` 个相同的层（通常称为 Transformer "块"）组成。每个 Transformer 块接受形状为 `(batch_size, sequence_length, d_model)` 的输入，并返回形状为 `(batch_size, sequence_length, d_model)` 的输出。每个块通过自注意力在序列中聚合信息，并通过前馈层对其进行非线性变换。

经过 `num_layers` 个 Transformer 块后，我们将利用最终激活值生成词汇表上的分布。

我们将实现"前置归一化"Transformer 块（详见第 3.4 节），它还要求在最终 Transformer 块之后使用层归一化，以确保其输出得到适当缩放。

归一化后，我们将使用标准的学习线性变换将 Transformer 块的输出转换为预测的下一 token logit（参见 A. Radford 等 [7] 等式 2）。

### 3.2 注：批处理、Einsum 与高效计算

在整个 Transformer 中，我们将对许多批量输入执行相同的计算。以下是几个例子：

- **批次元素**：对每个批次元素应用相同的 Transformer 前向运算。
- **序列长度**：RMSNorm 和前馈网络等"位置独立"运算对序列的每个位置独立执行。
- **注意力头**：注意力运算在"多头"注意力操作中跨注意力头批处理。

以一种符合人体工程学的方式执行这些操作——充分利用 GPU 且易于阅读和理解——是很有用的。许多 PyTorch 操作可以在张量开头接受额外的"批量维度"，并在这些维度上高效地重复/广播操作。

例如，假设我们正在进行位置独立的批量操作。我们有一个形状为 `(batch_size, sequence_length, d_model)` 的"数据张量" $D$，我们希望对形状为 `(d_model, d_model)` 的矩阵 $A$ 进行批量向量-矩阵乘法。在这种情况下，`D @ A` 将执行批量矩阵乘法（PyTorch 中的高效原语），其中 `(batch_size, sequence_length)` 维度被批量处理。

因此，假设你的函数可能会获得额外的批量维度并将这些维度保留在 PyTorch 形状的开头是很有帮助的。为了以这种方式组织张量以便批处理，可能需要通过 `view`、`reshape` 和 `transpose` 的许多步骤来调整形状，这可能有些繁琐，而且代码的意图和张量形状往往变得难以阅读。

更符合人体工程学的选择是在 `torch.einsum` 中使用 einsum 表示法，或者使用与框架无关的库如 `einops` 或 `einx`。两个关键操作是 einsum（可以对输入张量的任意维度进行张量收缩）和 rearrange（可以重新排序、连接和分割任意维度）。事实证明，机器学习中几乎所有操作都是维度调整和张量收缩的某种组合，偶尔带有（通常是逐点的）非线性函数。这意味着使用 einsum 表示法可以使大量代码更具可读性和灵活性。

**我们强烈建议在课程中学习和使用 einsum 表示法。** 之前未接触过 einsum 表示法的学生应使用 `einops`（文档[在此](https://einops.rocks/)），已熟悉 `einops` 的学生应学习更通用的 `einx`（[在此](https://github.com/fferflo/einx)）。⁴ 两个包都已安装在我们提供的环境中。

以下是一些 einsum 表示法使用示例。这些是 `einops` 文档的补充，请先阅读文档。

> ⁴ 值得注意的是，虽然 `einops` 有大量支持，但 `einx` 尚未经过充分测试。如果你发现 `einx` 有任何限制或错误，欢迎退回使用带有更多普通 PyTorch 的 `einops`。

---

> **示例（einstein_example1）：使用 einops.einsum 进行批量矩阵乘法**
>
> ```python
> import torch
> from einops import rearrange, einsum
>
> ## 基本实现
> Y = D @ A.T
> # 难以判断输入和输出形状及其含义。
> # D 和 A 可以有什么形状，是否有任何意外行为？
>
> ## Einsum 自文档化且健壮
> #                          D                A     ->          Y
> Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")
>
> ## 或者，批量版本，D 可以有任意前导维度，但 A 受约束。
> Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
> ```

---

> **示例（einstein_example2）：使用 einops.rearrange 的广播操作**
>
> 我们有一批图像，对于每张图像，我们希望根据某个缩放因子生成 10 个调暗版本：
>
> ```python
> images = torch.randn(64, 128, 128, 3)  # (batch, height, width, channel)
> dim_by = torch.linspace(start=0.0, end=1.0, steps=10)
>
> ## 调整形状并相乘
> dim_value = rearrange(dim_by,    "dim_value              -> 1 dim_value 1 1 1")
> images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
> dimmed_images = images_rearr * dim_value
>
> ## 或者一步完成：
> dimmed_images = einsum(
>     images, dim_by,
>     "batch height width channel, dim_value -> batch dim_value height width channel"
> )
> ```

---

> **示例（einstein_example3）：使用 einops.rearrange 的像素混合**
>
> 假设我们有一批图像，表示为形状 `(batch, height, width, channel)` 的张量，我们希望对图像的所有像素进行线性变换，但此变换对每个通道独立进行。线性变换表示为形状 `(height * width, height * width)` 的矩阵 $B$。
>
> ```python
> channels_last = torch.randn(64, 32, 32, 3)  # (batch, height, width, channel)
> B = torch.randn(32*32, 32*32)
>
> ## 重排图像张量以混合所有像素
> channels_last_flat = channels_last.view(
>     -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
> )
> channels_first_flat = channels_last_flat.transpose(1, 2)
> channels_first_flat_transformed = channels_first_flat @ B.T
> channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
> channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)
> ```
>
> 使用 einops 代替：
>
> ```python
> height = width = 32
> ## rearrange 替代笨拙的 torch view + transpose
> channels_first = rearrange(
>     channels_last,
>     "batch height width channel -> batch channel (height width)"
> )
> channels_first_transformed = einsum(
>     channels_first, B,
>     "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
> )
> channels_last_transformed = rearrange(
>     channels_first_transformed,
>     "batch channel (height width) -> batch height width channel",
>     height=height, width=width
> )
> ```
>
> 或者，如果你想更进一步：使用 `einx.dot`（`einops.einsum` 的 `einx` 等价物）一步完成：
>
> ```python
> height = width = 32
> channels_last_transformed = einx.dot(
>     "batch row_in col_in channel, (row_out col_out) (row_in col_in)"
>     "-> batch row_out col_out channel",
>     channels_last, B,
>     col_in=width, col_out=width
> )
> ```
>
> 第一种实现可以通过在前后添加注释来说明输入和输出形状，但这很笨拙且容易出错。使用 einsum 表示法，文档即是实现！

---

Einsum 表示法可以处理任意输入批处理维度，还有一个关键优势：自文档化。在使用 einsum 表示法的代码中，输入和输出张量的相关形状更加清晰。对于其余张量，你可以考虑使用张量类型提示，例如使用 `jaxtyping` 库（不特定于 JAX）。

我们将在作业 2 中进一步讨论 einsum 表示法的性能影响，但现在知道它们几乎总是优于替代方案！

#### 3.2.1 数学符号与内存顺序

许多机器学习论文在其符号中使用行向量，这与 NumPy 和 PyTorch 默认使用的行主序内存排列良好契合。使用行向量时，线性变换如下所示：

$$y = xW^\top, \tag{1}$$

其中 $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ 为行主序，$x \in \mathbb{R}^{1 \times d_\text{in}}$ 为行向量。注意，这允许我们通过增加 $x$ 的最外层维度来批处理输入，即可以用矩阵输入 $X \in \mathbb{R}^{\text{batch} \times d_\text{in}}$ 替换向量输入 $x$。

在线性代数中，通常更常见使用列向量，线性变换如下所示：

$$y = Wx, \tag{2}$$

其中 $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ 为行主序，$x \in \mathbb{R}^{d_\text{in}}$ 为列向量。在此设置下，对输入进行批处理时，批处理维度必须放在最后，因此 $x$ 需要替换为矩阵 $\tilde{X} \in \mathbb{R}^{d_\text{in} \times \text{batch}}$。

**在本作业中，我们将主要使用列向量进行数学符号表示**，因为数学通常遵循这种惯例。你应注意，如果你想使用普通矩阵乘法表示法，由于 PyTorch 使用行主序内存排列，你需要按照等式 1 中的行向量惯例使用转置矩阵。如果你使用 einsum 进行线性代数运算，只要正确标记轴，这将不成问题。顺便提一下，Matlab、Julia 和 Fortran 等其他语言/线性代数包都使用列主序内存排列，意味着批处理维度放在最后，但 Python 及相关包采用了行主序的 C 标准。

### 3.3 基本构建模块：Linear 和 Embedding 模块

#### 3.3.1 参数初始化

有效训练神经网络通常需要仔细初始化模型参数——不良的初始化可能导致梯度消失或爆炸等不良行为。前置归一化 Transformer 对初始化具有异常的鲁棒性，但初始化仍然可以对训练速度和收敛产生显著影响。由于本作业已经很长，我们将把细节留到作业 3，并提供一些在大多数情况下应该工作良好的近似初始化。现在，使用：

- **线性权重**：$\mathcal{N}(\mu=0, \sigma^2 = \frac{2}{d_\text{in}+d_\text{out}})$，截断在 $[-3\sigma, 3\sigma]$。
- **嵌入**：$\mathcal{N}(\mu=0, \sigma^2=1)$，截断在 $[-3, 3]$。
- **RMSNorm**：$\mathbf{1}$（全 1）。

你应使用 `torch.nn.init.trunc_normal_` 来初始化截断正态权重。

#### 3.3.2 Linear 模块

线性层是 Transformer 和神经网络的基本构建模块。首先，你将实现自己的 `Linear` 类，该类继承自 `torch.nn.Module` 并执行线性变换：

$$y = Wx. \tag{3}$$

注意，遵循大多数现代 LLM，我们不包含偏置项。

---

> **问题（linear）：实现 Linear 模块（1 分）**
>
> **交付物**：实现一个继承自 `torch.nn.Module` 的 `Linear` 类，执行线性变换。你的实现应遵循 PyTorch 内置 `nn.Linear` 模块的接口，除了没有 `bias` 参数或参数项。建议使用以下接口：
>
> `def __init__(self, in_features, out_features, device=None, dtype=None)` 构造线性变换模块，接受以下参数：
> - `in_features: int`  输入的最后一个维度
> - `out_features: int`  输出的最后一个维度
> - `device: torch.device | None = None`  存储参数的设备
> - `dtype: torch.dtype | None = None`  参数的数据类型
>
> `def forward(self, x: torch.Tensor) -> torch.Tensor` 对输入应用线性变换。
>
> 确保：
> - 子类化 `nn.Module`
> - 调用超类构造函数
> - 将参数构造并存储为 $W$（而非 $W^\top$），放入 `nn.Parameter`
> - 当然，不要使用 `nn.Linear` 或 `nn.functional.linear`
>
> 初始化使用上述设置，并使用 `torch.nn.init.trunc_normal_` 初始化权重。
>
> 要测试你的 `Linear` 模块，在 `[adapters.run_linear]` 实现测试适配器（适配器应将给定权重加载到你的 `Linear` 模块中，可使用 `Module.load_state_dict`），然后运行 `uv run pytest -k test_linear`。

---

#### 3.3.3 Embedding 模块

如上所述，Transformer 的第一层是嵌入层，将整数 token ID 映射到维度为 `d_model` 的向量空间。我们将实现一个继承自 `torch.nn.Module` 的自定义 `Embedding` 类（因此不应使用 `nn.Embedding`）。`forward` 方法应通过使用形状为 `(batch_size, sequence_length)` 的 `torch.LongTensor` token ID 对形状为 `(vocab_size, d_model)` 的嵌入矩阵进行索引来选择每个 token ID 的嵌入向量。

---

> **问题（embedding）：实现 Embedding 模块（1 分）**
>
> **交付物**：实现继承自 `torch.nn.Module` 的 `Embedding` 类，执行嵌入查找。你的实现应遵循 PyTorch 内置 `nn.Embedding` 模块的接口。建议使用以下接口：
>
> `def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)` 构造嵌入模块，接受以下参数：
> - `num_embeddings: int`  词汇表大小
> - `embedding_dim: int`  嵌入向量的维度，即 $d_\text{model}$
> - `device: torch.device | None = None`  存储参数的设备
> - `dtype: torch.dtype | None = None`  参数的数据类型
>
> `def forward(self, token_ids: torch.Tensor) -> torch.Tensor` 查找给定 token ID 的嵌入向量。
>
> 确保：
> - 子类化 `nn.Module`
> - 调用超类构造函数
> - 将嵌入矩阵初始化为 `nn.Parameter`
> - 存储嵌入矩阵时，`d_model` 为最后一个维度
> - 当然，不要使用 `nn.Embedding` 或 `nn.functional.embedding`
>
> 同样，使用上述初始化设置，并使用 `torch.nn.init.trunc_normal_` 初始化权重。
>
> 要测试你的实现，在 `[adapters.run_embedding]` 实现测试适配器，然后运行 `uv run pytest -k test_embedding`。

---

### 3.4 前置归一化 Transformer 块

每个 Transformer 块有两个子层：多头自注意力机制和位置独立前馈网络（[A. Vaswani 等, 2017]，第 3.1 节）。

在原始 Transformer 论文中，模型在每两个子层周围使用残差连接，后跟层归一化。这种架构通常称为"后置归一化"Transformer，因为层归一化应用于子层输出。然而，大量研究发现，将层归一化从每个子层的输出移至每个子层的输入（并在最后一个 Transformer 块后额外添加一个层归一化）可以提高 Transformer 训练稳定性 [T. Q. Nguyen 等, 2019; R. Xiong 等, 2020]——见图 2 的"前置归一化"Transformer 块可视化表示。然后，每个 Transformer 块子层的输出通过残差连接添加到子层输入（A. Vaswani 等 [8]，第 5.4 节）。前置归一化的一个直觉是：从输入嵌入到 Transformer 最终输出，存在一条干净的"残差流"，没有任何归一化，这据说可以改善梯度流。这种前置归一化 Transformer 现在是当今语言模型的标准（例如 GPT-3、LLaMA、PaLM 等），因此我们将实现这个变体。我们将逐步介绍前置归一化 Transformer 块的每个组件，依次实现它们。

#### 3.4.1 均方根层归一化

A. Vaswani 等 [8] 的原始 Transformer 实现使用层归一化 [J. L. Ba 等, 2016] 来归一化激活值。遵循 H. Touvron 等 [12]，我们将使用均方根层归一化（RMSNorm；B. Zhang 等 [13]，等式 4）进行层归一化。给定激活值向量 $a \in \mathbb{R}^{d_\text{model}}$，RMSNorm 将按如下方式重新缩放每个激活值 $a_i$：

$$\text{RMSNorm}(a_i) = \frac{a_i}{\text{RMS}(a)} g_i, \tag{4}$$

其中 $\text{RMS}(a) = \sqrt{\frac{1}{d_\text{model}} \sum_{i=1}^{d_\text{model}} a_i^2 + \varepsilon}$。这里，$g_i$ 是可学习的"增益"参数（总共有 `d_model` 个这样的参数），$\varepsilon$ 是通常固定为 `1e-5` 的超参数。

你应该将输入转换（upcast）为 `torch.float32` 以防止对输入平方时溢出。总体上，你的 `forward` 方法应如下所示：

```python
  in_dtype = x.dtype
  x = x.to(torch.float32)
  # 在此执行 RMSNorm 的代码
  ...
  result = ...
  # 以原始 dtype 返回结果
  return result.to(in_dtype)
```

---

> **问题（rmsnorm）：均方根层归一化（1 分）**
>
> **交付物**：将 RMSNorm 实现为 `torch.nn.Module`。建议使用以下接口：
>
> `def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)` 构造 RMSNorm 模块，接受以下参数：
> - `d_model: int`  模型的隐藏维度
> - `eps: float = 1e-5`  用于数值稳定性的 epsilon 值
> - `device: torch.device | None = None`  存储参数的设备
> - `dtype: torch.dtype | None = None`  参数的数据类型
>
> `def forward(self, x: torch.Tensor) -> torch.Tensor` 处理形状为 `(batch_size, sequence_length, d_model)` 的输入张量，返回相同形状的张量。
>
> **注意**：记得在执行归一化之前将输入转换为 `torch.float32`（之后再转回原始 dtype），如上所述。
>
> 要测试你的实现，在 `[adapters.run_rmsnorm]` 实现测试适配器，然后运行 `uv run pytest -k test_rmsnorm`。

---

#### 3.4.2 位置独立前馈网络

在原始 Transformer 论文（A. Vaswani 等 [8] 第 3.3 节）中，Transformer 前馈网络由两个线性变换组成，两者之间有一个 ReLU 激活（$\text{ReLU}(x) = \max(0, x)$）。在原始架构中，内层前馈层的维度通常是输入维度的 4 倍。

然而，现代语言模型相比原始设计倾向于融入两个主要变化：使用另一种激活函数，并采用门控机制。具体来说，我们将实现 Llama 3 [A. Grattafiori 等, 2024] 和 Qwen 2.5 [A. Yang 等, 2024] 等 LLM 中采用的"SwiGLU"激活函数，它将 SiLU（通常称为 Swish）激活与称为门控线性单元（GLU）的门控机制结合起来。我们还将省略线性层中有时使用的偏置项，遵循 PaLM [A. Chowdhery 等, 2022] 和 LLaMA [H. Touvron 等, 2023] 之后的大多数现代 LLM。

SiLU 或 Swish 激活函数 [D. Hendrycks 等, 2016; S. Elfwing 等, 2017] 定义如下：

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}} \tag{5}$$

如图 3 所示，SiLU 激活函数类似于 ReLU 激活函数，但在零处是光滑的。

门控线性单元（GLU）最初由 Y. N. Dauphin 等 [19] 定义为通过 sigmoid 函数的线性变换与另一个线性变换的逐元素乘积：

$$\text{GLU}(x, W_1, W_2) = \sigma(W_1 x) \odot W_2 x, \tag{6}$$

其中 $\odot$ 表示逐元素乘法。门控线性单元被建议"通过为梯度提供线性路径同时保留非线性能力来减少深层架构的梯度消失问题"。

将 SiLU/Swish 和 GLU 结合起来，我们得到 SwiGLU，用于我们的前馈网络：

$$\text{FFN}(x) = \text{SwiGLU}(x, W_1, W_2, W_3) = W_2(\text{SiLU}(W_1 x) \odot W_3 x), \tag{7}$$

其中 $x \in \mathbb{R}^{d_\text{model}}$，$W_1, W_3 \in \mathbb{R}^{d_\text{ff} \times d_\text{model}}$，$W_2 \in \mathbb{R}^{d_\text{model} \times d_\text{ff}}$，规范上 $d_\text{ff} = \frac{8}{3} d_\text{model}$。对于具体实现，可以将其四舍五入到最近的 64 的倍数以提高硬件效率。

N. Shazeer [20] 首先提出将 SiLU/Swish 激活与 GLU 结合，并进行实验表明 SwiGLU 在语言建模任务上优于 ReLU 和 SiLU（无门控）等基准。在作业的后续部分，你将比较 SwiGLU 和 SiLU。尽管我们提到了这些组件的一些启发式论点（论文提供了更多支持证据），但保持实证视角是好的：Shazeer 论文中有一句著名的引言：

> "我们无法解释为什么这些架构似乎有效；我们将其成功归因于神圣的仁慈，一如其他一切。"

---

> **问题（positionwise_feedforward）：实现位置独立前馈网络（2 分）**
>
> **交付物**：实现 SwiGLU 前馈网络，由 SiLU 激活函数和 GLU 组成。
>
> **注意**：在这种特殊情况下，你可以在实现中使用 `torch.sigmoid` 以保证数值稳定性。
>
> 在实现中，将 $d_\text{ff}$ 设置为约 $\frac{8}{3} \times d_\text{model}$，同时确保内层前馈层的维度是 64 的倍数以充分利用硬件。要对照提供的测试检验你的实现，需要在 `[adapters.run_swiglu]` 实现测试适配器，然后运行 `uv run pytest -k test_swiglu` 测试你的实现。

---

#### 3.4.3 相对位置嵌入

为了向模型注入位置信息，我们将实现旋转位置嵌入 [J. Su 等, 2021]，通常称为 RoPE。对于在 token 位置 $i$ 处的给定查询 token $q^{(i)} = W_q x^{(i)} \in \mathbb{R}^d$，我们将应用成对旋转矩阵 $R_i$，得到 $q'^{(i)} = R_i q^{(i)} = R_i W_q x^{(i)}$。这里，$R_i$ 将嵌入元素对 $q^{(i)}_{2k-1:2k}$ 作为 2d 向量旋转角度 $\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}$，其中 $k \in \{1, \ldots, d/2\}$，$\Theta$ 为某个常数。因此，我们可以将 $R_i$ 视为大小为 $d \times d$ 的块对角矩阵，对 $k \in \{1, \ldots, \frac{d}{2}\}$ 有块 $R_i^k$：

$$R_i^k = \begin{pmatrix} \cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\ \sin(\theta_{i,k}) & \cos(\theta_{i,k}) \end{pmatrix} \tag{8}$$

完整旋转矩阵 $R_i$ 为对角块矩阵，其中 $0$ 表示 $2 \times 2$ 零矩阵。虽然可以构造完整的 $d \times d$ 矩阵，但好的解决方案应利用该矩阵的属性更高效地实现变换。由于我们只关心给定序列内 token 的相对旋转，因此可以在各层和不同批次之间重用 $\cos(\theta_{i,k})$ 和 $\sin(\theta_{i,k})$ 的计算值。如果你想优化它，可以使用单个 RoPE 模块被所有层引用，它可以在初始化时使用 `self.register_buffer(persistent=False)` 创建一个预计算的 cos 和 sin 值的 2D 缓冲区，而不是 `nn.Parameter`（因为我们不想学习这些固定的余弦和正弦值）。对 $k^{(j)}$ 执行完全相同的旋转过程，按相应的 $R_j$ 旋转。注意此层没有可学习参数。

---

> **问题（rope）：实现 RoPE（2 分）**
>
> **交付物**：实现一个类 `RotaryPositionalEmbedding`，将 RoPE 应用于输入张量。
>
> 建议使用以下接口：
>
> `def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)` 构造 RoPE 模块并在需要时创建缓冲区，接受以下参数：
> - `theta: float`  RoPE 的 $\Theta$ 值
> - `d_k: int`  查询和键向量的维度
> - `max_seq_len: int`  将输入的最大序列长度
> - `device: torch.device | None = None`  存储缓冲区的设备
>
> `def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor` 处理形状为 `(..., seq_len, d_k)` 的输入张量，返回相同形状的张量。注意你应该容忍具有任意数量批处理维度的 $x$。应假设 token 位置是形状为 `(..., seq_len)` 的张量，指定 $x$ 沿序列维度的 token 位置。
>
> 你应使用 token 位置沿序列维度切片你的（可能预计算的）cos 和 sin 张量。
>
> 要测试你的实现，完成 `[adapters.run_rope]` 并确保通过 `uv run pytest -k test_rope`。

---

#### 3.4.4 缩放点积注意力

我们现在将按照 A. Vaswani 等 [8]（第 3.2.1 节）实现缩放点积注意力。作为初步步骤，注意力操作的定义将使用 softmax，该操作接受未归一化的分数向量并将其转换为归一化分布：

$$\text{softmax}(v)_i = \frac{\exp(v_i)}{\sum_{j=1}^n \exp(v_j)}. \tag{10}$$

注意，对于大值，$\exp(v_i)$ 可能变为 $\inf$（然后 $\inf / \inf = \text{NaN}$）。我们可以注意到 softmax 操作对所有输入加上任意常数 $c$ 是不变的，从而避免这个问题。我们可以利用这个属性来保证数值稳定性——通常，我们将从 $v$ 的所有元素中减去 $v$ 的最大值，使新的最大值为 0。你现在将使用此技巧实现 softmax。

---

> **问题（softmax）：实现 softmax（1 分）**
>
> **交付物**：编写一个函数对张量应用 softmax 操作。你的函数应接受两个参数：张量和维度 $i$，并将 softmax 应用于输入张量的第 $i$ 个维度。输出张量应与输入张量形状相同，但其第 $i$ 个维度现在将具有归一化的概率分布。使用从第 $i$ 个维度中所有元素中减去最大值的技巧来避免数值稳定性问题。
>
> 要测试你的实现，完成 `[adapters.run_softmax]` 并确保通过 `uv run pytest -k test_softmax_matches_pytorch`。

---

我们现在可以如下数学定义注意力操作：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V \tag{11}$$

其中 $Q \in \mathbb{R}^{n \times d_k}$，$K \in \mathbb{R}^{m \times d_k}$，$V \in \mathbb{R}^{m \times d_v}$。这里，$Q$、$K$ 和 $V$ 都是该操作的输入——注意这些不是可学习参数。

**掩码**：有时掩盖注意力操作的输出很方便。掩码应具有形状 $M \in \{\text{True}, \text{False}\}^{n \times m}$，每行 $i$ 表示查询 $i$ 应该关注哪些键。规范地（并且稍有混淆地），位置 $(i, j)$ 处的 `True` 值表示查询 $i$ 确实关注键 $j$，`False` 值表示查询不关注该键。换言之，在值为 `True` 的 $(i, j)$ 对处"信息流动"。例如，考虑条目为 `[[True, True, False]]` 的 $1 \times 3$ 掩码矩阵。单个查询向量只关注前两个键。

从计算上看，使用掩码比对子序列计算注意力效率高得多，我们可以通过取预 softmax 值（$QK^\top / \sqrt{d_k}$）并向掩码矩阵中任何 `False` 条目添加 $-\infty$ 来实现这一点。

---

> **问题（scaled_dot_product_attention）：实现缩放点积注意力（5 分）**
>
> **交付物**：实现缩放点积注意力函数。你的实现应处理形状为 `(batch_size, ..., seq_len, d_k)` 的键和查询，以及形状为 `(batch_size, ..., seq_len, d_v)` 的值，其中 `...` 表示任意数量的其他批量维度（如果提供）。实现应返回形状为 `(batch_size, ..., seq_len, d_v)` 的输出。有关批量维度的讨论，见第 3.2 节。
>
> 你的实现还应支持可选的用户提供的形状为 `(seq_len, seq_len)` 的布尔掩码。掩码值为 `True` 的位置的注意力概率之和应为 1，掩码值为 `False` 的位置的注意力概率应为零。
>
> 要对照提供的测试检验你的实现，需要在 `[adapters.run_scaled_dot_product_attention]` 实现测试适配器。`uv run pytest -k test_scaled_dot_product_attention` 在三阶输入张量上测试你的实现，`uv run pytest -k test_4d_scaled_dot_product_attention` 在四阶输入张量上测试你的实现。

---

#### 3.4.5 因果多头自注意力

我们将按照 A. Vaswani 等 [8]（第 3.2.2 节）实现多头自注意力。回想一下，数学上应用多头注意力的操作定义如下：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \tag{12}$$

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) \tag{13}$$

其中 $Q_i$、$K_i$、$V_i$ 分别是 $Q$、$K$、$V$ 的嵌入维度中大小为 $d_k$ 或 $d_v$ 的第 $i \in \{1, \ldots, h\}$ 个切片，`Attention` 是第 3.4.4 节定义的缩放点积注意力操作。由此我们可以形成多头自注意力操作：

$$\text{MultiHeadSelfAttention}(x) = W^O \text{MultiHead}(W^Q x, W^K x, W^V x) \tag{14}$$

这里，可学习参数为 $W^Q \in \mathbb{R}^{hd_k \times d_\text{model}}$，$W^K \in \mathbb{R}^{hd_k \times d_\text{model}}$，$W^V \in \mathbb{R}^{hd_v \times d_\text{model}}$，$W^O \in \mathbb{R}^{d_\text{model} \times hd_v}$。由于在多头注意力操作中 $Q$、$K$、$V$ 沿输出维度切片，我们可以将 $W^Q$、$W^K$ 和 $W^V$ 视为在每个头上沿输出维度分离的。当你实现好这个后，应该以总共三次矩阵乘法来计算键、值和查询投影。⁵

**因果掩码**

你的实现应防止模型关注序列中的未来 token。换言之，如果模型给定 token 序列 $t_1, \ldots, t_n$，并且我们想计算前缀 $t_1, \ldots, t_i$（其中 $i < n$）的下一词预测，模型不应该能访问（关注）位置 $t_{i+1}, \ldots, t_n$ 处的 token 表示，因为在推理期间生成文本时它不会访问这些 token（而且这些未来 token 会泄露真实下一词的身份，使语言建模预训练目标变得平凡）。对于输入 token 序列 $t_1, \ldots, t_n$，我们可以朴素地通过运行 $n$ 次多头自注意力（对于序列中 $n$ 个唯一前缀）来防止访问未来 token。相反，我们将使用因果注意力掩码，允许 token $i$ 关注序列中所有位置 $j \leq i$。你可以使用 `torch.triu` 或广播的索引比较来构造此掩码，并应利用第 3.4.4 节中的缩放点积注意力实现已经支持注意力掩码这一事实。

**应用 RoPE**

RoPE 应应用于查询和键向量，但不应用于值向量。此外，头维度应作为批处理维度处理，因为在多头注意力中，注意力对每个头独立应用。这意味着完全相同的 RoPE 旋转应应用于每个头的查询和键向量。

> ⁵ 作为进阶目标，尝试将键、查询和值投影合并为单个权重矩阵，使你只需要一次矩阵乘法。

---

> **问题（multihead_self_attention）：实现因果多头自注意力（5 分）**
>
> **交付物**：将因果多头自注意力实现为 `torch.nn.Module`。你的实现应至少接受以下参数：
>
> - `d_model: int`  Transformer 块输入的维度。
> - `num_heads: int`  多头自注意力中要使用的头数。
>
> 遵循 A. Vaswani 等 [8]，设置 $d_k = d_v = \frac{d_\text{model}}{h}$。要对照提供的测试检验你的实现，在 `[adapters.run_multihead_self_attention]` 实现测试适配器，然后运行 `uv run pytest -k test_multihead_self_attention` 测试你的实现。

---

### 3.5 完整 Transformer LM

让我们开始组装 Transformer 块（参考图 2 会有帮助）。Transformer 块包含两个"子层"，一个用于多头自注意力，另一个用于 SwiGLU 前馈网络。在每个子层中，我们首先执行 RMSNorm，然后是主要操作（MHA/FF），最后添加残差连接。

具体来说，Transformer 块的第一半（第一个"子层"）应实现以下更新，从输入 $x$ 产生输出 $y$：

$$y = x + \text{MultiHeadSelfAttention}(\text{RMSNorm}(x)). \tag{15}$$

---

> **问题（transformer_block）：实现 Transformer 块（3 分）**
>
> 按照第 3.4 节的描述和图 2 的说明实现前置归一化 Transformer 块。你的 Transformer 块应至少接受以下参数：
>
> - `d_model: int`  Transformer 块输入的维度。
> - `num_heads: int`  多头自注意力中要使用的头数。
> - `d_ff: int`  位置独立前馈内层的维度。
>
> 要测试你的实现，实现适配器 `[adapters.run_transformer_block]`，然后运行 `uv run pytest -k test_transformer_block` 测试你的实现。
>
> **交付物**：通过提供测试的 Transformer 块代码。

---

现在我们将各块组合在一起，遵循图 1 中的高层图示。按照第 3.1 节中对嵌入的描述，将其送入 `num_layers` 个 Transformer 块，然后通过最终层归一化和 LM 头，以获得词汇表上的未归一化分布（logit）。

---

> **问题（transformer_lm）：实现 Transformer LM（3 分）**
>
> 是时候把所有内容整合在一起了！按照第 3.1 节的描述和图 1 的说明实现 Transformer 语言模型。你的实现至少应接受 Transformer 块的所有上述构造参数，以及以下附加参数：
>
> - `vocab_size: int`  词汇表大小，用于确定 token 嵌入矩阵的维度。
> - `context_length: int`  最大上下文长度，用于确定 RoPE sin 和 cos 缓冲区的维度。
> - `num_layers: int`  要使用的 Transformer 块数量。
>
> 要对照提供的测试检验你的实现，首先需要在 `[adapters.run_transformer_lm]` 实现测试适配器，然后运行 `uv run pytest -k test_transformer_lm` 测试你的实现。
>
> **交付物**：通过上述测试的 Transformer LM 模块。

---

#### 资源核算

了解 Transformer 的各个部分如何消耗计算和内存是很有用的。我们将完成一些基本的"FLOPs 核算"。Transformer 中绝大多数 FLOPs 来自矩阵乘法，因此我们的核心方法很简单：

1. 写出 Transformer 前向传播中所有的矩阵乘法。
2. 将每个矩阵乘法转换为所需的 FLOPs。

对于第二步，以下事实将很有用：

**规则**：给定 $A \in \mathbb{R}^{m \times n}$ 和 $B \in \mathbb{R}^{n \times p}$，矩阵乘积 $AB$ 需要 $2mnp$ 个 FLOPs。

为了看到这一点，注意 $(AB)[i, j] = A[i, :] \cdot B[:, j]$，而这个点积需要 $n$ 次加法和 $n$ 次乘法（$2n$ 个 FLOPs）。然后，由于矩阵乘积 $AB$ 有 $m \times p$ 个元素，总 FLOPs 数为 $(2n)(mp) = 2mnp$。

现在，在你做下一个问题之前，遍历 Transformer 块和 Transformer LM 的每个组件并列出所有矩阵乘法及其关联的 FLOPs 成本会很有帮助。

---

> **问题（transformer_accounting）：Transformer LM 资源核算（5 分）**
>
> (a) 考虑使用我们的作业架构的 GPT-2 XL 大小的模型，具有以下配置：
>
> | 参数 | 值 |
> |------|-----|
> | `vocab_size` | 50,257 |
> | `context_length` | 1,024 |
> | `num_layers` | 48 |
> | `d_model` | 1,600 |
> | `num_heads` | 25 |
> | `d_ff` | 4,288（$\frac{8}{3} \times 1600$ 最近的 64 倍数） |
>
> 假设我们使用此配置构造我们的模型。我们的模型有多少可训练参数？假设每个参数使用单精度浮点表示，仅加载此模型需要多少内存？
>
> **交付物**：一到两句话回答。
>
> (b) 确定完成 GPT-2 XL 形状模型前向传播所需的矩阵乘法。这些矩阵乘法总共需要多少 FLOPs？假设输入序列有 `context_length` 个 token。
>
> **交付物**：矩阵乘法列表（含描述），以及所需的总 FLOPs 数。
>
> (c) 根据你的上述分析，模型的哪些部分需要最多 FLOPs？
>
> **交付物**：一到两句话回答。
>
> (d) 对 GPT-2 small（12 层，768 d_model，12 头）、GPT-2 medium（24 层，1024 d_model，16 头）和 GPT-2 large（36 层，1280 d_model，20 头）重复你的分析。随着模型规模增大，Transformer LM 的哪些部分占总 FLOPs 的比例增加或减少？
>
> **交付物**：对于每个模型，提供模型组件及其关联 FLOPs（占前向传播所需总 FLOPs 的比例）的分解。此外，提供一到两句话描述改变模型规模如何改变每个组件的 FLOPs 比例。
>
> (e) 将 GPT-2 XL 的上下文长度增加到 16,384。一次前向传播的总 FLOPs 如何变化？模型组件的 FLOPs 相对贡献如何变化？
>
> **交付物**：一到两句话回答。

---

## 4 训练 Transformer LM

现在我们已经有了预处理数据（通过分词器）和模型（Transformer）的步骤。剩下的是构建所有支持训练的代码，包括：

- **损失函数**：我们需要定义损失函数（交叉熵）。
- **优化器**：我们需要定义优化器来最小化这个损失（AdamW）。
- **训练循环**：我们需要所有支持基础设施来加载数据、保存检查点和管理训练。

### 4.1 交叉熵损失

回想一下，Transformer 语言模型为长度为 $m+1$ 的每个序列 $x$ 和 $i = 1, \ldots, m$ 定义了分布 $p_\theta(x_{i+1} | x_{1:i})$。给定由长度为 $m+1$ 的序列组成的训练集 $\mathcal{D}$，我们定义标准交叉熵（负对数似然）损失函数：

$$\ell(\theta; \mathcal{D}) = \frac{1}{|\mathcal{D}|m} \sum_{x \in \mathcal{D}} \sum_{i=1}^{m} -\log p_\theta(x_{i+1} | x_{1:i}). \tag{16}$$

（注意，Transformer 中的单次前向传播对所有 $i = 1, \ldots, m$ 均可得到 $p_\theta(x_{i+1} | x_{1:i})$。）

特别地，Transformer 为每个位置 $i$ 计算 logit $o_i \in \mathbb{R}^\text{vocab\_size}$，从而：⁶

$$p(x_{i+1} | x_{1:i}) = \text{softmax}(o_i)[x_{i+1}] = \frac{\exp(o_i[x_{i+1}])}{\sum_{a=1}^{\text{vocab\_size}} \exp(o_i[a])}. \tag{17}$$

交叉熵损失通常针对 logit 向量 $o_i \in \mathbb{R}^\text{vocab\_size}$ 和目标 $x_{i+1}$ 定义。⁷

> ⁶ 注意，$o_i[k]$ 指向量 $o_i$ 的索引 $k$ 处的值。
> ⁷ 这对应于 $x_{i+1}$ 上的 Dirac delta 分布与预测的 $\text{softmax}(o_i)$ 分布之间的交叉熵。

实现交叉熵损失需要像 softmax 一样仔细处理数值问题。

---

> **问题（cross_entropy）：实现交叉熵（1 分）**
>
> **交付物**：编写一个函数来计算交叉熵损失，该函数接受预测 logit（$o_i$）和目标（$x_{i+1}$），并计算交叉熵 $\ell_i = -\log \text{softmax}(o_i)[x_{i+1}]$。你的函数应处理以下事项：
>
> - 为数值稳定性减去最大元素。
> - 尽可能消除 log 和 exp。
> - 处理任何额外的批处理维度，并返回批次平均值。与第 3.2 节一样，我们假设批量维度始终放在前面，位于词汇表大小维度之前。
>
> 实现 `[adapters.run_cross_entropy]`，然后运行 `uv run pytest -k test_cross_entropy` 测试你的实现。

---

**困惑度**

交叉熵对于训练已足够，但评估模型时，我们还想报告困惑度。对于长度为 $m$ 的序列，交叉熵损失为 $\ell_1, \ldots, \ell_m$：

$$\text{perplexity} = \exp\left(\frac{1}{m} \sum_{i=1}^{m} \ell_i\right). \tag{18}$$

### 4.2 SGD 优化器

现在我们有了损失函数，开始探索优化器。最简单的基于梯度的优化器是随机梯度下降（SGD）。我们从随机初始化的参数 $\theta_0$ 开始，然后对每个步骤 $t = 0, \ldots, T-1$ 执行以下更新：

$$\theta_{t+1} \leftarrow \theta_t - \alpha_t \nabla \mathcal{L}(\theta_t; \mathcal{B}_t), \tag{19}$$

其中 $\mathcal{B}_t$ 是从数据集 $\mathcal{D}$ 中随机采样的一批数据，学习率 $\alpha_t$ 和批大小 $|\mathcal{B}_t|$ 是超参数。

#### 4.2.1 在 PyTorch 中实现 SGD

为了实现我们的优化器，我们将子类化 PyTorch 的 `torch.optim.Optimizer` 类。`Optimizer` 子类必须实现两个方法：

`def __init__(self, params, ...)` 应初始化你的优化器。这里，`params` 将是要优化的参数集合（或参数组，以防用户想对模型的不同部分使用不同的超参数，例如学习率）。确保将 `params` 传递给基类的 `__init__` 方法，该方法将存储这些参数以供 `step` 使用。你可以根据优化器接受额外参数（例如学习率是一个常见参数），并将它们以字典形式（键为你为这些参数选择的名称字符串）传递给基类构造函数。

`def step(self)` 应对参数进行一次更新。在训练循环中，这将在反向传播后被调用，因此你可以访问最后一批的梯度。此方法应遍历每个参数张量 `p` 并就地修改它们，即根据梯度 `p.grad`（如果存在，持有损失相对于该参数的梯度的张量）设置 `p.data`（持有与该参数关联的张量）。

PyTorch 优化器 API 有几个微妙之处，所以用示例解释更容易。为了使示例更丰富，我们将实现 SGD 的一个轻微变体，其中学习率在训练过程中衰减，从初始学习率 $\alpha$ 开始，随时间逐步减小步骤：

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{t+1}} \nabla \mathcal{L}(\theta_t; \mathcal{B}_t) \tag{20}$$

让我们看看如何将这个版本的 SGD 实现为 PyTorch Optimizer：

```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # 获取学习率
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # 获取与 p 关联的状态
                t = state.get("t", 0)  # 从状态获取迭代次数，或默认为 0
                grad = p.grad.data  # 获取损失对 p 的梯度
                p.data -= lr / math.sqrt(t + 1) * grad  # 就地更新权重张量
                state["t"] = t + 1  # 递增迭代次数
        return loss
```

在 `__init__` 中，我们将参数以及默认超参数传递给基类构造函数（参数可能以组的形式出现，每组具有不同的超参数）。如果参数只是单个 `torch.nn.Parameter` 对象的集合，基类构造函数将创建一个单组并分配默认超参数。然后，在 `step` 中，我们遍历每个参数组，再遍历该组中的每个参数，并应用等式 20。这里，我们将迭代次数作为与每个参数关联的状态保存：我们首先读取该值，在梯度更新中使用它，然后更新它。API 规定用户可能传入一个可调用的 `closure` 在优化器步骤之前重新计算损失。我们的优化器不需要这个，但我们添加它以符合 API。

要查看此实现，可以使用以下最小训练循环示例：

```python
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)
for t in range(100):
    opt.zero_grad()  # 重置所有可学习参数的梯度
    loss = (weights**2).mean()  # 计算标量损失值
    print(loss.cpu().item())
    loss.backward()  # 运行反向传播，计算梯度
    opt.step()  # 运行优化器步骤
```

这是训练循环的典型结构：在每次迭代中，我们将计算损失并运行一步优化器。训练语言模型时，我们的可学习参数将来自模型（在 PyTorch 中，`m.parameters()` 给出这个集合）。损失将在采样的数据批次上计算，但训练循环的基本结构将是相同的。

---

> **问题（learning_rate_tuning）：调整学习率（1 分）**
>
> 我们将看到，影响训练最大的超参数之一是学习率。让我们在我们的玩具示例中实际体验一下。使用另外三个学习率值运行上述 SGD 示例：`1e1`、`1e2` 和 `1e3`，只运行 10 次训练迭代。对于每个学习率，损失会发生什么？它是衰减得更快、更慢，还是发散（即在训练过程中增加）？
>
> **交付物**：一到两句话回答，描述你观察到的行为。

---

### 4.3 AdamW

现代语言模型通常使用更复杂的优化器而非 SGD 进行训练。最近使用的大多数优化器都是 Adam 优化器 [D. P. Kingma 等, 2015] 的派生。我们将使用 AdamW [I. Loshchilov 等, 2019]，这在最近的工作中被广泛使用。AdamW 提出了对 Adam 的修改，通过添加权重衰减（在每次迭代中，我们将参数拉向 0）来改善正则化，这与梯度更新解耦。我们将按照 I. Loshchilov 等 [23] 算法 2 的描述实现 AdamW。

AdamW 是有状态的：对于每个参数，它跟踪其一阶矩和二阶矩的运行估计。因此，AdamW 以额外的内存换取改善的稳定性和收敛性。除了学习率 $\alpha$，AdamW 还有一对超参数 $(\beta_1, \beta_2)$，用于控制矩估计的更新，以及权重衰减率 $\lambda$。典型应用将 $(\beta_1, \beta_2)$ 设置为 $(0.9, 0.999)$，但大型语言模型如 LLaMA [H. Touvron 等, 2023] 和 GPT-3 [T. B. Brown 等, 2020] 通常使用 $(0.9, 0.95)$。算法可以写成如下形式，其中 $\varepsilon$ 是一个小值（例如 $10^{-8}$），用于在 $v$ 中出现极小值时改善数值稳定性：

---

**算法 1：AdamW 优化器**

```
init(θ)                      ▷ 初始化可学习参数
m ← 0                        ▷ 一阶矩向量初始值；与 θ 形状相同
v ← 0                        ▷ 二阶矩向量初始值；与 θ 形状相同
for t = 1, …, T do
    采样一批数据 B_t
    g ← ∇_θ ℓ(θ; B_t)        ▷ 计算损失的梯度
    α_t ← α * √(1-β₂ᵗ) / (1-β₁ᵗ)   ▷ 计算迭代 t 的调整 α
    θ ← θ - αλθ              ▷ 应用权重衰减
    m ← β₁m + (1-β₁)g       ▷ 更新一阶矩估计
    v ← β₂v + (1-β₂)g²      ▷ 更新二阶矩估计
    θ ← θ - α_t * m / (√v + ε)  ▷ 应用矩调整的权重更新
end for
```

注意 $t$ 从 1 开始。你现在将实现这个优化器。

---

> **问题（adamw）：实现 AdamW（2 分）**
>
> **交付物**：将 AdamW 优化器实现为 `torch.optim.Optimizer` 的子类。你的类应在 `__init__` 中接受学习率 $\alpha$，以及 $\beta$、$\varepsilon$ 和 $\lambda$ 超参数。为了帮助你保存状态，基础 `Optimizer` 类提供了字典 `self.state`，它将 `nn.Parameter` 对象映射到存储该参数所需信息的字典（对于 AdamW，这将是矩估计）。实现 `[adapters.get_adamw_cls]` 并确保通过 `uv run pytest -k test_adamw`。

---

> **问题（adamw_accounting）：使用 AdamW 训练的资源核算（2 分）**
>
> 让我们计算运行 AdamW 需要多少内存和计算量。假设对所有张量都使用 `float32`。
>
> (a) 运行 AdamW 需要多少峰值内存？根据参数、激活值、梯度和优化器状态的内存使用量分解你的答案。用 `batch_size` 和模型超参数（`vocab_size`、`context_length`、`num_layers`、`d_model`、`num_heads`）来表达你的答案。假设 $d_\text{ff} = \frac{8}{3} \times d_\text{model}$。
>
> 为简单起见，在计算激活值内存使用时，仅考虑以下组件：
>
> - Transformer 块
>   - RMSNorm（多个）
>   - 多头自注意力子层：$QKV$ 投影、$QK^\top$ 矩阵乘法、softmax、值的加权求和、输出投影
>   - 位置独立前馈（SwiGLU）：$W_1$、$W_2$、门控分支上的 SiLU、逐元素乘积、$W_3$
> - 最终 RMSNorm
> - 输出嵌入
> - logit 上的交叉熵
>
> **交付物**：参数、激活值、梯度和优化器状态的各自代数表达式，以及总计。
>
> (b) 将你的答案实例化为 GPT-2 XL 形状的模型，得到仅依赖于 `batch_size` 的表达式。你能使用的最大批大小是多少，同时仍然能在 80GB 内存中装下？
>
> **交付物**：形如 $a \cdot \text{batch\_size} + b$ 的表达式（$a$、$b$ 为数值），以及代表最大批大小的数字。
>
> (c) 运行 AdamW 的一步需要多少 FLOPs？
>
> **交付物**：代数表达式，附简要说明。
>
> (d) 模型 FLOPs 利用率（MFU）定义为观测到的吞吐量（tokens/秒）与硬件理论峰值 FLOP 吞吐量之比 [A. Chowdhery 等, 2022]。NVIDIA H100 GPU 的"float32"（实际上是 TensorFloat-32，实质上是"bfloat19"）操作理论峰值为 495 teraFLOP/s。假设你能达到 50% MFU，在单个 H100 上训练 GPT-2 XL 进行 400K 步、批大小为 1024 需要多长时间？遵循 J. Kaplan 等 [25] 和 J. Hoffmann 等 [26]，假设反向传播的 FLOPs 是前向传播的两倍。
>
> **交付物**：训练所需的小时数，附简要说明。

---

### 4.4 学习率调度

在训练过程中，导致损失最快下降的学习率值通常会有所变化。在训练 Transformer 时，通常使用学习率调度，一开始使用较大的学习率进行更快的更新，然后随着模型训练逐渐衰减到较小的值。⁸ 在本作业中，我们将实现用于训练 LLaMA [H. Touvron 等, 2023] 的余弦退火调度。

调度器简单来说是一个函数，接受当前步骤 $t$ 和其他相关参数（如初始学习率和最终学习率），并返回步骤 $t$ 处梯度更新要使用的学习率。最简单的调度是常数函数，对任何 $t$ 返回相同的学习率。

余弦退火学习率调度接受 (i) 当前迭代 $t$，(ii) 最大学习率 $\alpha_\text{max}$，(iii) 最小（最终）学习率 $\alpha_\text{min}$，(iv) 预热迭代次数 $T_w$，以及 (v) 余弦退火的最终迭代 $T_c$。迭代 $t$ 处的学习率定义为：

- **（预热）** 若 $t < T_w$，则 $\alpha_t = \frac{t}{T_w} \alpha_\text{max}$。
- **（余弦退火）** 若 $T_w \leq t \leq T_c$，则 $\alpha_t = \alpha_\text{min} + \frac{1}{2}\left(1 + \cos\left(\frac{t - T_w}{T_c - T_w}\pi\right)\right)(\alpha_\text{max} - \alpha_\text{min})$。
- **（退火后）** 若 $t > T_c$，则 $\alpha_t = \alpha_\text{min}$。

> ⁸ 有时会使用学习率重新上升（重启）的调度，以帮助跳出局部极小值。

---

> **问题（learning_rate_schedule）：实现带预热的余弦学习率调度（1 分）**
>
> 编写一个函数，接受 $t$、$\alpha_\text{max}$、$\alpha_\text{min}$、$T_w$ 和 $T_c$，并按照上面定义的调度器返回学习率 $\alpha_t$。然后实现 `[adapters.get_lr_cosine_schedule]` 并确保通过 `uv run pytest -k test_get_lr_cosine_schedule`。

---

### 4.5 梯度裁剪

在训练过程中，有时会遇到产生大梯度的训练样本，这可能会导致训练不稳定。为了缓解这一问题，实践中常用的一种技术是梯度裁剪。其思路是在每次反向传播后、执行优化器步骤之前，对梯度的范数施加限制。

给定（所有参数的）梯度 $g$，我们计算其 $\ell_2$ 范数 $\|g\|_2$。如果该范数小于最大值 $M$，则保持 $g$ 不变；否则，将 $g$ 按因子 $\frac{M}{\|g\|_2 + \varepsilon}$ 缩小（其中加入小 $\varepsilon$（如 $10^{-6}$）以保证数值稳定性）。注意，结果范数将略小于 $M$。

---

> **问题（gradient_clipping）：实现梯度裁剪（1 分）**
>
> 编写一个函数实现梯度裁剪。你的函数应接受参数列表和最大 $\ell_2$ 范数，并就地修改每个参数梯度。使用 $\varepsilon = 10^{-6}$（PyTorch 默认值）。然后，实现适配器 `[adapters.run_gradient_clipping]` 并确保通过 `uv run pytest -k test_gradient_clipping`。

---

## 5 训练循环

我们现在终于可以将迄今为止构建的主要组件整合在一起：分词后的数据、模型和优化器。

### 5.1 数据加载器

分词后的数据（例如你在 `tokenizer_experiments` 中准备的数据）是单个 token 序列 $x = (x_1, \ldots, x_n)$。尽管源数据可能由单独的文档组成（例如不同的网页或源代码文件），但常见做法是将所有这些数据连接成一个单一的 token 序列，在它们之间添加分隔符（如 `<|endoftext|>` token）。

数据加载器将此转换为批次流，其中每批包含 $B$ 个长度为 $m$ 的序列，以及对应的下一 token，也是长度 $m$。例如，对于 $B=1$，$m=3$，$([x_2, x_3, x_4], [x_3, x_4, x_5])$ 将是一个潜在批次。

以这种方式加载数据简化了训练的诸多方面。首先，任何 $1 \leq i \leq n - m$ 都给出一个有效的训练序列，因此采样训练序列是平凡的。由于所有训练序列长度相同，无需填充输入序列，这提高了硬件利用率（也通过增加批大小 $B$）。最后，我们也不需要加载完整数据集来采样训练数据，使大型数据集的处理变得容易，即使这些数据集可能无法全部装入内存。

---

> **问题（data_loading）：实现数据加载（2 分）**
>
> **交付物**：编写一个函数，接受 NumPy 数组 $x$（包含 token ID 的整数数组）、`batch_size`、`context_length` 和 PyTorch 设备字符串（例如 `'cpu'` 或 `'cuda:0'`），并返回一对张量：采样的输入序列和对应的下一 token 目标。两个张量的形状都应为 `(batch_size, context_length)`，包含 token ID，且都应放置在请求的设备上。要对照提供的测试检验你的实现，首先需要在 `[adapters.run_get_batch]` 实现测试适配器，然后运行 `uv run pytest -k test_get_batch` 测试你的实现。

---

> **低资源提示：在 CPU 或 Apple Silicon 上加载数据**
>
> 如果你计划在 CPU 或 Apple Silicon 上训练 LM，需要将数据移动到正确的设备（类似地，模型也应使用相同设备）。
>
> 在 CPU 上，可以使用 `'cpu'` 设备字符串；在 Apple Silicon（M* 芯片）上，可以使用 `'mps'` 设备字符串。
>
> 有关 MPS 的更多信息，请查看以下资源：
> - [https://docs.pytorch.org/docs/stable/mps.html](https://docs.pytorch.org/docs/stable/mps.html)
> - [https://docs.pytorch.org/docs/stable/notes/mps.html](https://docs.pytorch.org/docs/stable/notes/mps.html)
> - [https://developer.apple.com/documentation/metalperformanceshaders](https://developer.apple.com/documentation/metalperformanceshaders)

---

如果数据集太大无法加载到内存中怎么办？我们可以使用名为 `mmap` 的 Unix 系统调用，它将磁盘上的文件映射到虚拟内存，并在访问该内存位置时懒加载文件内容。这样，你可以"假装"整个数据集都在内存中。NumPy 通过 `np.memmap`（或在 `np.load` 时使用标志 `mmap_mode='r'`，如果你最初用 `np.save` 保存了数组）实现这一功能，它返回一个按需加载条目的类 NumPy 数组对象。在训练期间从数据集中采样时（即 NumPy 数组），确保以内存映射模式加载数据集（通过 `np.memmap` 或 `np.load` 的标志 `mmap_mode='r'`，取决于你如何保存数组）。还要确保指定与正在加载的数组匹配的 `dtype`。显式验证内存映射数据看起来正确（例如，不包含超出预期词汇表大小的值）可能会有所帮助。

### 5.2 检查点

除了加载数据，我们还需要在训练时保存模型。运行任务时，我们通常希望能够恢复中途停止的训练（例如由于任务超时、机器故障等）。即使一切顺利，我们也可能希望之后访问中间模型（例如事后研究训练动态、从训练不同阶段的模型中采样等）。

检查点应包含我们恢复训练所需的所有状态。我们当然希望至少能够恢复模型权重。如果使用有状态优化器（如 AdamW），我们还需要保存优化器状态（例如对于 AdamW，是矩估计）。最后，为了恢复学习率调度，我们需要知道停止时的迭代次数。PyTorch 使这一切变得容易：每个 `nn.Module` 都有一个 `state_dict()` 方法，返回包含所有可学习权重的字典；我们之后可以使用对应的 `load_state_dict()` 方法恢复这些权重。`torch.optim.Optimizer` 也一样。最后，`torch.save(obj, dest)` 可以将对象（例如，包含张量作为某些值、但也包含整数等常规 Python 对象的字典）转储到文件（路径）或类文件对象，然后可以使用 `torch.load(src)` 将其加载回内存。

---

> **问题（checkpointing）：实现模型检查点（1 分）**
>
> 实现以下两个函数来加载和保存检查点：
>
> `def save_checkpoint(model, optimizer, iteration, out)` 应将模型、优化器和迭代次数的所有状态转储到类文件对象 `out` 中。你可以使用模型和优化器的 `state_dict` 方法获取其相关状态，并使用 `torch.save(obj, out)` 将 `obj` 转储到 `out`（PyTorch 在这里支持路径或类文件对象）。典型选择是让 `obj` 为字典，但你可以使用任何格式，只要你之后能加载检查点。
>
> 此函数期望以下参数：
> - `model: torch.nn.Module`
> - `optimizer: torch.optim.Optimizer`
> - `iteration: int`
> - `out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
>
> `def load_checkpoint(src, model, optimizer)` 应从 `src`（路径或类文件对象）加载检查点，然后从该检查点恢复模型和优化器状态。你的函数应返回保存到检查点的迭代次数。你可以使用 `torch.load(src)` 恢复你在 `save_checkpoint` 实现中保存的内容，以及模型和优化器中的 `load_state_dict` 方法将它们恢复到之前的状态。
>
> 此函数期望以下参数：
> - `src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
> - `model: torch.nn.Module`
> - `optimizer: torch.optim.Optimizer`
>
> 实现 `[adapters.run_save_checkpoint]` 和 `[adapters.run_load_checkpoint]` 适配器，并确保通过 `uv run pytest -k test_checkpointing`。

---

### 5.3 训练循环

现在，终于是时候将你实现的所有组件整合到主训练脚本中了。花时间让不同超参数的训练运行易于启动是值得的（例如，通过将它们作为命令行参数接受），因为你之后将多次进行这些操作以研究不同选择对训练的影响。

---

> **问题（training_together）：整合在一起（4 分）**
>
> **交付物**：编写一个脚本，运行训练循环以在用户提供的输入上训练你的模型。特别地，我们建议你的训练脚本至少允许以下内容：
>
> - 配置和控制各种模型和优化器超参数的能力。
> - 使用 `np.memmap` 高效加载大型训练和验证数据集。
> - 将检查点序列化到用户提供的路径。
> - 定期记录训练和验证性能（例如，到控制台和/或外部服务如 Weights and Biases）。⁹

> ⁹ wandb.ai

---

## 6 生成文本

现在我们可以训练模型了，最后一块拼图是从模型生成文本的能力。回想一下，语言模型接受（可能批处理的）长度为 `sequence_length` 的整数序列作为输入，并生成大小为 `(sequence_length, vocab_size)` 的矩阵，其中序列的每个元素都是预测该位置之后下一个 token 的概率分布。我们现在将编写几个函数，将其转化为新序列的采样方案。

**Softmax**

按照标准惯例，语言模型输出是最终线性层的输出（"logit"），因此我们必须通过 softmax 操作（我们在等式 10 中看到的）将其转换为归一化概率。

**解码**

要从模型生成文本（解码），我们将为模型提供一系列前缀 token（"提示词"），并要求其生成词汇表上的概率分布，预测序列中的下一个 token。然后，我们将从该词汇表项的分布中采样以确定下一个输出 token。

具体地，解码过程的一步应接受序列 $x_{1\ldots t}$ 并通过以下方程返回 token $x_{t+1}$：

$$P(x_{t+1} = i | x_{1\ldots t}) = \frac{\exp(v_i)}{\sum_j \exp(v_j)} \tag{21}$$

$$v = \text{TransformerLM}(x_{1\ldots t})_t \in \mathbb{R}^\text{vocab\_size} \tag{22}$$

其中 $\text{TransformerLM}$ 是我们的模型，接受长度为 `sequence_length` 的序列作为输入，生成大小为 `(sequence_length, vocab_size)` 的矩阵，我们取该矩阵的最后一个元素，因为我们寻找第 $t$ 个位置的下一 token 预测。

这为我们提供了一个基本解码器，通过反复从这些单步条件分布中采样（将之前生成的输出 token 添加到下一解码时间步的输入中），直到我们生成序列结束 token `<|endoftext|>`（或达到用户指定的最大生成 token 数量）。

**解码技巧**

我们将在小型模型上进行实验，小型模型有时会生成质量很低的文本。两个简单的解码技巧可以帮助解决这些问题。首先，在**温度缩放**中，我们用温度参数 $\tau$ 修改 softmax，新的 softmax 为：

$$\text{softmax}(v, \tau)_i = \frac{\exp(v_i / \tau)}{\sum_{j=1}^{\text{vocab\_size}} \exp(v_j / \tau)}. \tag{23}$$

注意，设置 $\tau \to 0$ 使得 $v$ 的最大元素占主导，softmax 的输出变为集中在该最大元素处的独热向量。

其次，另一个技巧是**核采样**（nucleus）或 top-p 采样，我们通过截断低概率 token 来修改采样分布。设 $q$ 是从（温度缩放的）softmax 得到的大小为 `vocab_size` 的概率分布。核采样（超参数为 $p$）按以下方程生成下一个 token：

$$P(x_{t+1} = i | q) = \begin{cases} \frac{q_i}{\sum_{j \in V(p)} q_j} & \text{若 } i \in V(p) \\ 0 & \text{否则} \end{cases} \tag{24}$$

其中 $V(p)$ 是最小的索引集合，使得 $\sum_{j \in V(p)} q_j \geq p$。你可以通过首先按大小对概率分布 $q$ 排序，然后选择最大的词汇表元素直到达到目标水平 $p$ 来轻松计算这个量。

---

> **问题（decoding）：解码（3 分）**
>
> **交付物**：实现一个从你的语言模型解码的函数。我们建议支持以下功能：
>
> - 为用户提供的提示生成补全（即接受某个 $x_{1\ldots t}$ 并采样补全，直到遇到 `<|endoftext|>` token）。
> - 允许用户控制生成 token 的最大数量。
> - 给定所需的温度值，在采样之前对预测的下一 token 分布应用 softmax 温度缩放。
> - Top-$p$ 采样（[A. Holtzman 等, 2020]，也称为核采样），给定用户指定的阈值。

---

## 7 实验

现在是时候把所有内容整合在一起，在预训练数据集上训练（小型）语言模型了。

### 7.1 如何运行实验和交付物

理解 Transformer 架构组件背后原理的最佳方式是实际修改它并自己运行。没有什么能替代亲身实践的经验。

为此，快速、一致地进行实验并保留记录非常重要。为了快速实验，我们将在小规模模型（约 1700 万总参数）和简单数据集（TinyStories）上运行许多实验。为了一致性，你将以系统的方式消融组件和变化超参数；为了保留记录，我们将要求你提交实验日志和与每个实验相关的学习曲线。

为了能够提交损失曲线，确保定期评估验证损失并记录步骤数和挂钟时间。你可能会发现像 Weights and Biases 这样的日志基础设施很有帮助。

---

> **问题（experiment_log）：实验记录（3 分）**
>
> 对于你的训练和评估代码，创建实验追踪基础设施，允许你追踪实验和相对于梯度步骤和挂钟时间的损失曲线。
>
> **交付物**：你的实验的日志基础设施代码，以及本节以下作业问题的实验日志（记录你尝试的所有内容的文档）。

---

### 7.2 TinyStories

我们将从一个非常简单的数据集（TinyStories；R. Eldan 等 [1]）开始，模型将快速训练，我们可以看到一些有趣的行为。获取此数据集的说明在第 1 节。以下是该数据集的一个示例。

---

> **示例（tinystories_example）：TinyStories 中的一个示例**
>
> Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase. When Ben saw it he was amazed! He said, "Wow, that is a really amazing vase! Can I buy it?" The shopkeeper smiled and said, "Of course you can. You can take it home and show all your friends how amazing it is!" So Ben took the vase home and he was so proud of it! He called his friends over and showed them the amazing vase. All his friends thought the vase was beautiful and couldn't believe how lucky Ben was. And that's how Ben found an amazing vase in the store!

---

#### 7.2.1 超参数调整

我们会告诉你一些非常基本的超参数，并要求你找到其他一些效果好的设置。

- **词汇表大小** 10000。典型的词汇表大小在数万到数十万之间。你应该改变这个参数，看看词汇表和模型行为如何变化。
- **上下文长度** 256。像 TinyStories 这样的简单数据集可能不需要很长的序列长度，但对于后续的 OpenWebText 数据，你可能需要改变这个参数。尝试改变它，看看对每次迭代运行时间和最终困惑度的影响。
- **d_model** 512。这比许多小型 Transformer 论文中使用的 768 维稍小，但这会使速度更快。
- **d_ff** 1344。这大约是 $\frac{8}{3} d_\text{model}$，同时是 64 的倍数，这对 GPU 性能有益。
- **RoPE theta 参数** $\Theta$ 10000。
- **层数和头数** 4 层，16 头。合在一起，这将给出约 1700 万非嵌入参数，是一个相当小的 Transformer。
- **处理的总 token 数** 327,680,000（你的批大小 × 总步数 × 上下文长度应大致等于这个值）。

你应该进行一些试错，为以下其他超参数找到好的默认值：学习率、学习率预热、其他 AdamW 超参数（$\beta_1$、$\beta_2$、$\varepsilon$）以及权重衰减。你可以在 D. P. Kingma 等 [22] 中找到这些超参数的一些典型选择。

#### 7.2.2 整合在一起

现在你可以通过获取训练好的 BPE 分词器、对训练数据集进行分词，以及在你编写的训练循环中运行，将所有内容整合在一起。**重要说明**：如果你的实现正确且高效，上述超参数在 1 块 B200 GPU 上的运行时间应约为 20–30 分钟。如果你的运行时间远长于此，请检查并确保你的数据加载、检查点保存或验证损失代码没有成为瓶颈，并确保你的实现得到了适当的批处理。

#### 7.2.3 调试模型架构的技巧

我们强烈建议你熟悉 IDE 内置的调试器（例如 VSCode/Zed），这将比用打印语句调试节省时间。如果你使用文本编辑器，可以使用类似 `ipdb` 的工具。调试模型架构时的其他一些好做法：

- 开发任何神经网络架构时，常见的第一步是对单个小批次过拟合。如果你的实现正确，你应该能够很快地将训练损失驱动到接近零。
- 在各个模型组件中设置调试断点，检查中间张量的形状，确保它们符合你的预期。
- 监控激活值、模型权重和梯度的范数，确保它们没有爆炸或消失。

---

> **问题（learning_rate）：调整学习率（2 B200 小时）（3 分）**
>
> 学习率是最重要的需要调整的超参数之一。使用你训练好的基础模型，回答以下问题：
>
> (a) 对学习率进行超参数搜索，报告最终损失（或如果优化器发散则注明发散）。
>
> **交付物**：与多个学习率相关的学习曲线。解释你的超参数搜索策略。
>
> **交付物**：在 TinyStories 上验证损失（每 token）不超过 1.45 的模型。

---

> **低资源提示：在 CPU 或 Apple Silicon 上训练几步**
>
> 如果你在 cpu 或 mps 上运行，应将处理的总 token 数减少到 40,000,000，这足以产生相当流畅的文本。你也可以将目标验证损失从 1.45 提高到 2.00。
>
> 在 M4 Max 芯片和 36 GB RAM 上使用调整过学习率的解决方案代码，我们使用批大小 × 总步数 × 上下文长度 = 32 × 5000 × 256 = 40,960,000 个 token，在 cpu 上耗时 1 小时 22 分钟，在 mps 上耗时 36 分钟。在步骤 5000 时，我们达到验证损失 1.80。
>
> 一些额外提示：
> - 使用 $N$ 个训练步骤时，建议调整余弦学习率衰减调度，使其衰减（即达到最小学习率）恰好在步骤 $N$ 时终止。
> - 使用 mps 时，不要使用 TF32 内核，即不要设置 `torch.set_float32_matmul_precision('high')`（你可能会对 cuda 设备这样做）。我们尝试在 mps 上启用 TF32 内核（torch 版本 2.9.0），发现后端有时使用无声出错的内核，导致训练不稳定。
> - 你可以通过使用 `torch.compile` 对模型进行 JIT 编译来加速训练。具体来说：
>   - 在 cpu 上，用 `model = torch.compile(model)` 编译你的模型
>   - 在 mps 上，你可以用 `model = torch.compile(model, backend="aot_eager")` 优化反向传播
>   - 截至 torch 版本 2.9.0，mps 上不支持使用 Inductor 编译。

---

> (b) 民间智慧是最佳学习率处于"稳定性边缘"。研究学习率发散点与你最佳学习率的关系。
>
> **交付物**：包含至少一个发散运行的递增学习率学习曲线，以及分析这与收敛率的关系。

---

现在让我们改变批大小，看看训练会发生什么。批大小很重要——它们让我们通过进行更大的矩阵乘法从 GPU 获得更高效率，但我们是否总是想要大批大小是真的吗？让我们运行一些实验来找出答案。

---

> **问题（batch_size_experiment）：批大小变化（1 B200 小时）（1 分）**
>
> 将批大小从 1 一直变化到 GPU 内存限制。在中间尝试至少几个批大小，包括典型大小如 64 和 128。
>
> **交付物**：不同批大小运行的学习曲线。如有必要，应再次优化学习率。
>
> **交付物**：几句话讨论你对批大小及其对训练影响的发现。

---

有了解码器，我们现在可以生成文本了！我们将从模型生成并看看效果如何。作为参考，你应该得到至少和以下示例一样好的输出：

---

> **示例（ts_generate_example）：TinyStories 语言模型的样本输出**
>
> Once upon a time, there was a pretty girl named Lily. She loved to eat gum, especially the big black one. One day, Lily's mom asked her to help cook dinner. Lily was so excited! She loved to help her mom. Lily's mom made a big pot of soup for dinner. Lily was so happy and said, "Thank you, Mommy! I love you." She helped her mom pour the soup into a big bowl. After dinner, Lily's mom made some yummy soup. Lily loved it! She said, "Thank you, Mommy! This soup is so yummy!" Her mom smiled and said, "I'm glad you like it, Lily." They finished cooking and continued to cook together. The end.

---

> **低资源提示：在 CPU 或 Apple Silicon 上生成文本**
>
> 如果你使用了处理 40M token 的低资源配置，你应该看到仍然类似英语但不如上述流畅的生成结果。例如，我们在训练了 40M token 的 TinyStories 语言模型上的样本输出如下：
>
> *Once upon a time, there was a little girl named Sue. Sue had a tooth that she loved very much. It was his best head. One day, Sue went for a walk and met a ladybug! They became good friends and played on the path together.*
>
> *"Hey, Polly! Let's go out!" said Tim. Sue looked at the sky and saw that it was difficult to find a way to dance shining. She smiled and agreed to help the talking!"*
>
> *As Sue watched the sky moved, what it was. She*

---

> **问题（generate）：生成文本（1 分）**
>
> 使用你的解码器和训练好的检查点，报告你的模型生成的文本。你可能需要调整解码参数（温度、top-p 等）来获得流畅的输出。
>
> **交付物**：至少 256 个 token 的文本转储（或直到第一个 `<|endoftext|>` token），以及关于此输出流畅性的简短评论和至少两个影响此输出好坏的因素。

---

### 7.3 消融实验与架构修改

理解 Transformer 的最佳方式是实际修改它并观察其行为。我们现在将进行几个简单的消融实验和修改。

**消融 1：层归一化**

人们常说层归一化对 Transformer 训练的稳定性很重要。但也许我们想冒险一试。让我们从每个 Transformer 块中移除 RMSNorm，看看会发生什么。

---

> **问题（layer_norm_ablation）：移除 RMSNorm 并训练（0.5 B200 小时）（1 分）**
>
> 从你的 Transformer 中移除所有 RMSNorm 并进行训练。在之前最优学习率下会发生什么？使用较低的学习率能获得稳定性吗？
>
> **交付物**：移除 RMSNorm 后训练的学习曲线，以及最佳学习率的学习曲线。
>
> **交付物**：几句话评论 RMSNorm 的影响。

---

现在让我们研究另一个乍看随意的层归一化选择。前置归一化 Transformer 块定义为：

$$z = x + \text{MultiHeadSelfAttention}(\text{RMSNorm}(x)) \tag{25}$$
$$y = z + \text{FFN}(\text{RMSNorm}(z)). \tag{26}$$

这是对原始 Transformer 架构（使用后置归一化方法）为数不多的"共识"修改之一：

$$z = \text{RMSNorm}(x + \text{MultiHeadSelfAttention}(x)) \tag{27}$$
$$y = \text{RMSNorm}(z + \text{FFN}(z)). \tag{28}$$

让我们回退到后置归一化方法，看看会发生什么。

---

> **问题（pre_norm_ablation）：实现后置归一化并训练（0.5 B200 小时）（1 分）**
>
> 将你的前置归一化 Transformer 实现修改为后置归一化。使用后置归一化模型训练，看看会发生什么。
>
> **交付物**：后置归一化 Transformer 的学习曲线，与前置归一化的进行比较。

---

我们看到层归一化对 Transformer 的行为有重大影响，甚至层归一化的位置也很重要。

**消融 2：位置嵌入**

接下来我们将研究位置嵌入对模型性能的影响。具体来说，我们将比较我们的基础模型（含 RoPE）与完全不包含位置嵌入（NoPE）的情况。事实证明，仅解码器 Transformer（即我们实现的带因果掩码的 Transformer）理论上可以在没有显式提供位置嵌入的情况下推断相对或绝对位置信息 [Y.-H. H. Tsai 等, 2019; A. Kazemnejad 等, 2023]。我们现在将实证测试 NoPE 与 RoPE 相比的表现。

---

> **问题（no_pos_emb）：实现 NoPE（0.5 B200 小时）（1 分）**
>
> 修改含 RoPE 的 Transformer 实现，完全移除位置嵌入信息，看看会发生什么。
>
> **交付物**：比较 RoPE 和 NoPE 性能的学习曲线。

---

**消融 3：SwiGLU 与 SiLU**

接下来，我们将遵循 N. Shazeer [20] 并测试前馈网络中门控的重要性，通过比较 SwiGLU 前馈网络与使用 SiLU 激活但没有门控线性单元（GLU）的前馈网络的性能：

$$\text{FFN}_\text{SiLU}(x) = W_2 \text{SiLU}(W_1 x). \tag{29}$$

回想一下，在我们的 SwiGLU 实现中，内层前馈层的维度设置为约 $d_\text{ff} = \frac{8}{3} d_\text{model}$（同时确保 $d_\text{ff} \mod 64 = 0$，以利用 GPU 张量核）。在这个消融基准中，你的 $\text{FFN}_\text{SiLU}$ 实现应将 $d_\text{ff}$ 设置为 $4 \times d_\text{model}$，以大致匹配默认 SwiGLU 前馈网络的参数数量（后者有三个而非两个权重矩阵）。

---

> **问题（swiglu_ablation）：SwiGLU 与 SiLU（0.5 B200 小时）（1 分）**
>
> **交付物**：比较 SwiGLU 和 SiLU 前馈网络性能的学习曲线，参数数量大致匹配。
>
> **交付物**：几句话讨论你的发现。

---

> **低资源提示：GPU 资源有限的在线学生应在 TinyStories 上测试修改**
>
> 在作业的剩余部分，我们将转向更大规模、噪声更多的网络数据集（OpenWebText），进行架构修改实验，以及（可选地）向课程排行榜提交。
>
> 在 OpenWebText 上训练 LM 到流畅需要很长时间，因此我们建议 GPU 访问有限的在线学生继续在 TinyStories 上测试修改（使用验证损失作为衡量性能的指标）。

---

### 7.4 在 OpenWebText 上运行

我们现在将转向从网络爬虫创建的更标准的预训练数据集。OpenWebText [A. Gokaslan 等, 2019] 的一个小样本也作为单个文本文件提供：获取此文件的方法见第 1 节。

以下是 OpenWebText 的一个示例。注意文本更加真实、复杂和多样。你可能需要浏览训练数据集，以了解网络爬取语料库的训练数据是什么样子的。

---

> **示例（owt_example）：OWT 中的一个示例**
>
> Baseball Prospectus director of technology Harry Pavlidis took a risk when he hired Jonathan Judge.
>
> Pavlidis knew that, as Alan Schwarz wrote in The Numbers Game, "no corner of American culture is more precisely counted, more passionately quantified, than performances of baseball players." With a few clicks here and there, you can find out that Noah Syndergaard's fastball revolves more than 2,100 times per minute on its way to the plate, that Nelson Cruz had the game's highest average exit velocity among qualified hitters in 2016 and myriad other tidbits that seem ripped from a video game or science fiction novel. The rising ocean of data has empowered an increasingly important actor in baseball's culture: the analytical hobbyist.
>
> That empowerment comes with added scrutiny – on the measurements, but also on the people and publications behind them. With Baseball Prospectus, Pavlidis knew all about the backlash that accompanies quantitative imperfection. He also knew the site's catching metrics needed to be reworked, and that it would take a learned mind – someone who could tackle complex statistical modeling problems – to complete the job.
>
> "He freaks us out." Harry Pavlidis
>
> Pavlidis had a hunch that Judge "got it" based on the latter's writing and their interaction at a site-sponsored ballpark event. [...]

---

**注意**：你可能需要为此实验重新调整超参数，例如学习率或批大小。

---

> **问题（main_experiment）：在 OWT 上实验（2 B200 小时）（2 分）**
>
> 使用与 TinyStories 相同的模型架构和总训练迭代次数，在 OpenWebText 上训练你的语言模型。这个模型表现如何？
>
> **交付物**：你的语言模型在 OpenWebText 上的学习曲线。描述与 TinyStories 的损失差异——我们应该如何解释这些损失？
>
> **交付物**：以与 TinyStories 输出相同格式的 OpenWebText LM 生成文本。这段文本的流畅性如何？为什么即使我们使用相同的模型和计算预算，输出质量却比 TinyStories 更差？

---

### 7.5 你自己的修改 + 排行榜

恭喜你走到这一步，你快完成了！你现在将尝试改进 Transformer 架构，并看看你的超参数和架构与班级其他同学相比如何。

**排行榜规则**

除以下内容外没有限制：

- **运行时间**：你的提交在 B200 上最多可以运行 45 分钟。如果你使用 SLURM 或 Modal，你可能需要在提交脚本中强制执行此限制。
- **数据**：你只能使用我们提供的 OpenWebText 训练数据集。

除此之外，你可以随心所欲地做任何事。

如果你在寻找一些实现思路，可以查看以下资源：

- 最先进的开源 LLM 系列，如 Llama 3 [A. Grattafiori 等, 2024] 或 Qwen 2.5 [A. Yang 等, 2024]。
- NanoGPT 速跑仓库（[github.com/KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)），社区成员在此发布了许多有趣的"速跑"小规模语言模型预训练的修改。例如，一个可以追溯到原始 Transformer 论文的常见修改是将输入和输出嵌入的权重绑定在一起（见 A. Vaswani 等 [8]（第 3.4 节）和 A. Chowdhery 等 [16]（第 2 节））。如果你尝试权重绑定，你可能需要减小嵌入/LM 头初始化的标准差。

在尝试完整的 45 分钟运行之前，你应该先在 OpenWebText 的一个小子集或 TinyStories 上测试这些修改。

需要注意的是，你在这个排行榜中发现效果好的一些修改可能无法推广到更大规模的预训练。我们将在课程的缩放定律单元中进一步探讨这个想法。

---

> **问题（leaderboard）：排行榜（10 B200 小时）（6 分）**
>
> 你将在上述排行榜规则下训练一个模型，目标是在 0.75 B200 小时内最小化语言模型的验证损失。
>
> **交付物**：记录的最终验证损失，相关学习曲线（清楚显示挂钟时间 x 轴不超过 45 分钟），以及你所做事情的描述。我们期望排行榜提交至少超过 5.0 损失的朴素基线。提交至排行榜：[github.com/stanford-cs336/assignment1-basics-leaderboard](https://github.com/stanford-cs336/assignment1-basics-leaderboard)。

---

## 参考文献

[1] R. Eldan and Y. Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?." 2023.

[2] A. Gokaslan, V. Cohen, E. Pavlick, and S. Tellex, "OpenWebText corpus." 2019.

[3] R. Sennrich, B. Haddow, and A. Birch, "Neural Machine Translation of Rare Words with Subword Units," in *Proc. of ACL*, 2016.

[4] C. Wang, K. Cho, and J. Gu, "Neural Machine Translation with Byte-Level Subwords." 2019.

[5] P. Gage, "A new algorithm for data compression," *C Users Journal*, vol. 12, no. 2, pp. 23–38, Feb. 1994.

[6] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, "Language Models are Unsupervised Multitask Learners." 2019.

[7] A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever, "Improving Language Understanding by Generative Pre-Training." 2018.

[8] A. Vaswani et al., "Attention is All you Need," in *Proc. of NeurIPS*, 2017.

[9] T. Q. Nguyen and J. Salazar, "Transformers without Tears: Improving the Normalization of Self-Attention," in *Proc. of IWSWLT*, 2019.

[10] R. Xiong et al., "On Layer Normalization in the Transformer Architecture," in *Proc. of ICML*, 2020.

[11] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer Normalization." 2016.

[12] H. Touvron et al., "LLaMA: Open and Efficient Foundation Language Models." 2023.

[13] B. Zhang and R. Sennrich, "Root Mean Square Layer Normalization," in *Proc. of NeurIPS*, 2019.

[14] A. Grattafiori et al., "The Llama 3 Herd of Models." Available: [https://arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783)

[15] A. Yang et al., "Qwen2.5 Technical Report," *arXiv preprint arXiv:2412.15115*, 2024.

[16] A. Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways." 2022.

[17] D. Hendrycks and K. Gimpel, "Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units." 2016.

[18] S. Elfwing, E. Uchibe, and K. Doya, "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning." Available: [https://arxiv.org/abs/1702.03118](https://arxiv.org/abs/1702.03118)

[19] Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier, "Language Modeling with Gated Convolutional Networks." Available: [https://arxiv.org/abs/1612.08083](https://arxiv.org/abs/1612.08083)

[20] N. Shazeer, "GLU Variants Improve Transformer." 2020.

[21] J. Su, Y. Lu, S. Pan, B. Wen, and Y. Liu, "RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021.

[22] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in *Proc. of ICLR*, 2015.

[23] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in *Proc. of ICLR*, 2019.

[24] T. B. Brown et al., "Language Models are Few-Shot Learners," in *Proc. of NeurIPS*, 2020.

[25] J. Kaplan et al., "Scaling Laws for Neural Language Models." 2020.

[26] J. Hoffmann et al., "Training Compute-Optimal Large Language Models." 2022.

[27] A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi, "The Curious Case of Neural Text Degeneration," in *Proc. of ICLR*, 2020.

[28] Y.-H. H. Tsai, S. Bai, M. Yamada, L.-P. Morency, and R. Salakhutdinov, "Transformer Dissection: An Unified Understanding for Transformer's Attention via the Lens of Kernel," in *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, K. Inui, J. Jiang, V. Ng, and X. Wan, Eds., Hong Kong, China: Association for Computational Linguistics, Nov. 2019, pp. 4344–4353. doi: 10.18653/v1/D19-1443.

[29] A. Kazemnejad, I. Padhi, K. Natesan, P. Das, and S. Reddy, "The Impact of Positional Encoding on Length Generalization in Transformers," in *Thirty-seventh Conference on Neural Information Processing Systems*, 2023. Available: [https://openreview.net/forum?id=Drrl2gcjzl](https://openreview.net/forum?id=Drrl2gcjzl)
