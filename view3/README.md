## 项目概述

本项目用于对 Git 提交的 `diff` 数据进行语义检索。主要功能包括：

1. **生成 query 集**：从 JSONL 数据中随机抽取 diff 用作检索测试。
2. **构建检索索引**：

   * **BM25 索引**：基于 diff 文本，快速检索。
   * **CodeBERT 向量索引**：语义嵌入，支持余弦相似度检索。
   * **Jina Embeddings 向量索引**：使用 `jina-embeddings-v2-base-code` 模型，支持语义检索。
3. **检索与相似度计算**：

   * BM25 + 向量混合检索
   * 余弦相似度用于向量检索

---

## 环境搭建

### 1 创建 Conda 环境

项目提供 `environment.yml`，可创建干净环境：

```bash
conda env create -f environment.yml
```

### 2️ 激活环境

```bash
conda activate myproject
```

### 3️ 验证安装

```bash
python -c "import torch; import transformers; import jina; import sentence_transformers; print('All modules loaded')"
```

---

---

## 生成 Query 集

从 `test.jsonl` 中随机抽取 1000 条 diff 作为 query 测试集：

```python
import json, random

N_QUERIES = 1000
with open("resource/apachecm/full.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

queries = random.sample(lines, N_QUERIES)

with open("query.jsonl", "w", encoding="utf-8") as f:
    f.writelines(queries)

print(f"Saved {N_QUERIES} queries to query.jsonl")
```

---

## 构建 BM25 索引

使用 `bm25_diff_retrieval.py`：

```bash
python bm25_diff_retrieval.py full.jsonl
```

* 会加载 diff 字段并构建 BM25 索引
* 可用于快速基于文本匹配的检索

---

## 构建 CodeBERT 向量索引

使用 `codebert_build_index.py`：

```bash
python codebert_build_index.py full.jsonl
```

* 输出文件：`../resource/codebert_diff_index.pkl`


---

## 构建 Jina Embeddings 向量索引

使用 `jina_build_diff_index.py`：

```bash
python jina_build_diff_index.py full.jsonl
```

* 输出文件：`jina_diff_index.pkl`


---

## 注意事项

1. **GPU 支持**

   * CodeBERT 可在 GPU 上加速，需安装 `torch` 对应 CUDA 版本
2. **批量处理**

   * 建议大规模 diff 使用 batch 编码，防止显存爆满
3. **索引归一化**

   * 构建索引时已做 L2 归一化，点积可直接作为余弦相似度

---

## 推荐流程

1. 创建干净 Conda 环境
2. 生成 query 测试集
3. 构建 BM25 索引（快速匹配）
4. 构建 CodeBERT / Jina 向量索引（语义检索）
5. 混合检索：先 BM25 Top-K → 向量 rerank


---
