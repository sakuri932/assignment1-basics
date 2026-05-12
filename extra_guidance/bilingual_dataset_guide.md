# 中英双语预训练语料选择指南

> 适用场景：作业规模（10~30GB）的中英双语语言模型预训练，含一定翻译能力。

---

## 核心前提

### Tokenizer 词表大小
纯英文模型用 32K vocab 基本够用。加入中文后，常用汉字约 3500 个，常用英文子词也需要数千个位置，**建议将 vocab_size 提升至 50,000~64,000**，否则中文覆盖率会不足，压缩比差。

### 语料配比原则
- 中英比例建议约 1:1（按字节），避免一种语言压倒另一种
- 如果需要翻译能力，额外加入 5~10% 的平行语料（过多会让模型偏向翻译任务）

---

## 单语语料

### 英文

#### OpenWebText（已有）
- **来源**：Reddit 上被点赞的外链网页正文，由社区复现自 GPT-2 的训练集 WebText
- **规模**：约 12GB
- **质量**：★★★★ 内容多样，以英文讨论、新闻、博客为主
- **获取**：作业中已下载（`owt_train.txt` / `owt_valid.txt`），直接复用

---

### 中文

#### 中文 Wikipedia（首选）
- **来源**：维基百科中文版全量 dump，内容为百科条目，书面语为主，结构规整
- **规模**：约 2GB（解压并提取纯文本后）
- **质量**：★★★★★ 极干净，无广告/垃圾内容
- **获取**：
```bash
# 1. 下载 XML dump（约 2GB 压缩包）
wget https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2

# 2. 用 wikiextractor 转纯文本
pip install wikiextractor
python -m wikiextractor.WikiExtractor zhwiki-latest-pages-articles.xml.bz2 -o zh_wiki/

# 3. 合并为单文件（用 <|endoftext|> 分隔文章）
find zh_wiki/ -name "wiki_*" -exec cat {} \; | \
  sed 's/<doc[^>]*>//g; s/<\/doc>/<|endoftext|>/g' > zh_wiki.txt
```

#### mC4（中文切片）
- **来源**：Google 从 Common Crawl（每月全网爬虫快照）中清洗出的多语言语料，中文切片数百GB，按需截取
- **规模**：可截取任意量，建议取 5~8GB 补充多样性
- **质量**：★★★ 网页来源，有噪声，但经过语言识别和基本过滤
- **获取**：通过 HuggingFace Datasets 流式下载，无需全量下载：
```python
from datasets import load_dataset

ds = load_dataset("mc4", "zh", streaming=True, split="train")

with open("mc4_zh.txt", "w", encoding="utf-8") as f:
    for i, example in enumerate(ds):
        if i >= 500_000:  # 约 5~8GB，按需调整
            break
        f.write(example["text"] + "\n<|endoftext|>\n")
```

#### CLUECorpus2020
- **来源**：中文语言理解测评基准（CLUE）发布的预训练语料，来源涵盖新闻、百科、问答、评论等，经人工筛选
- **规模**：约 14GB
- **质量**：★★★★ 较干净，领域多样
- **获取**：填表申请，1~2 天审批，地址：[github.com/CLUEbenchmark/CLUECorpus2020](https://github.com/CLUEbenchmark/CLUECorpus2020)

---

## 平行语料（提供翻译能力）

纯单语混合语料中模型看到的是两堆分开的语言，不知道对应关系。要具备翻译能力，必须加入平行语料（同一句话的中英对照）。

### 推荐数据集

| 数据集 | 规模 | 内容 | 获取 |
|--------|------|------|------|
| **OPUS CCAligned** | 约 2800 万句对 | 网页对齐句对，领域广泛 | [opus.nlpl.eu](https://opus.nlpl.eu) 直接下载 |
| **WMT 中英（新闻）** | 约 2000 万句对 | 新闻翻译，质量高 | [statmt.org/wmt22](https://statmt.org/wmt22) |
| **WikiMatrix** | 约 890 万句对 | 跨语言 Wikipedia 对齐 | HuggingFace `wikimedia/WikiMatrix` |
| **BELLE 翻译数据** | 约 50 万条 | 指令式翻译对，适合 SFT 阶段 | HuggingFace `BelleGroup/train_0.5M_CN` |

### 格式化方式

平行语料需要格式化后才能混入预训练语料，常见两种做法：

**方法一：拼接（最简单，推荐）**

中英句对直接拼接，用特殊分隔符连接，让模型从上下文学对应关系：
```
今天天气很好。<|sep|>The weather is nice today.<|endoftext|>
深度学习改变了自然语言处理。<|sep|>Deep learning transformed NLP.<|endoftext|>
```

**方法二：交替排列**

同一篇文章的中文版和英文版交替出现（Wikipedia 两种语言都有，天然对齐），模型在预训练中隐式学习对齐关系，无需显式分隔符。

---

## 推荐最终配比

```
英文单语：OpenWebText                    12 GB
中文单语：中文 Wikipedia + mC4-zh 截片    10 GB（2GB + 8GB）
中英平行：OPUS CCAligned 或 WMT           2 GB（格式化后）
──────────────────────────────────────────────
总计：约 24 GB    vocab_size: 50,000~64,000
```

平行语料占总量约 8%，在提供翻译能力的同时不影响一般语言建模质量。这是 mBERT、XLM-R 等多语言模型的通行做法。

---

## 处理流程

1. 下载各语料，统一转换为 UTF-8 纯文本
2. 文档之间用 `<|endoftext|>` 分隔（与现有 tokenizer 脚本一致）
3. 将平行语料格式化后混入（按方法一或方法二）
4. 用修改后的 `train_bpe`（已支持中日韩预分词）训练 50K~64K vocab 的分词器
5. 用 `encode_iterable` 将全部语料编码为 `.npy` 文件

---

## 对现有代码的影响

本作业的 tokenizer 已在 `cs336_basics/tokenizer.py` 中扩展了中日韩预分词支持（`\p{Han}|\p{Hiragana}|\p{Katakana}|\p{Hangul}` 分支），**无需修改代码**，直接用中文语料训练 BPE 即可。
