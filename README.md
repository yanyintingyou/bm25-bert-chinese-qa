# bm25-bert-chinese-qa

**基于 "BM25 粗排 + Cross-Encoder BERT 重排" 的两阶段中文问答检索系统**

> 本仓库为本人**计算机科学与技术专业（自考助学本科二学历）**毕业设计项目的代码留档仓库。仅上传核心代码部分。非科班出身，水平有限，烦请谅解。

A two-stage Chinese Question Answering retrieval system based on **BM25 Coarse Ranking + BERT Reranking**. Achieves significant quality improvement with minimal latency overhead on consumer-grade hardware.

---

## 项目简介 / Project Overview

### 中文
针对大规模中文问答场景下“语义匹配精度”与“检索实时性”难以兼顾的难题，本项目提出并实现了一种 **“BM25 粗排 + BERT 重排”** 的两阶段检索架构。

系统首先利用 BM25 算法结合并行倒排索引，从百万级文档中快速召回 Top-50 候选集；随后接入轻量级 Cross-Encoder BERT 模型（基于 MiniLM 蒸馏），通过全层语义交互对候选集进行重排序。

本项目基于 **T2Ranking** 大规模中文基准数据集进行验证，并使用 Streamlit 开发了可视化交互演示系统。

### English
To address the challenge of balancing semantic matching accuracy and retrieval latency in large-scale Chinese QA scenarios, this project implements a **two-stage retrieval architecture**: BM25 for coarse candidate retrieval + Cross-Encoder BERT for semantic reranking.

The system first uses a parallel BM25 inverted index to quickly retrieve a Top-50 candidate set from millions of documents, then applies a lightweight Cross-Encoder BERT model (MiniLM-distilled) to perform deep semantic interaction and rerank the candidates.

The project is validated on the **T2Ranking** large-scale Chinese passage ranking benchmark and includes a Streamlit-based interactive visualization demo.

---

## 核心亮点 / Key Highlights

### 中文
- **两阶段架构**：BM25 负责高召回快速筛选，BERT 负责语义精度精排
- **显著性能提升**：在 T2Ranking 验证集上，相比纯 BM25 基线：
  - MRR@10 提升 **28.68%**
  - Recall@10 提升 **20.77%**
  - 额外平均延迟仅 **46.2 毫秒**
- **工程实用性**：在消费级硬件（RTX 5070 Ti）上实现百万级文档的实时响应
- **完整实现**：包含并行索引构建、检索引擎、语义重排模块与 Streamlit 可视化界面

### English
- **Two-stage Architecture**: BM25 for high-recall fast filtering + BERT for semantic precision reranking
- **Significant Improvements** (on T2Ranking validation set vs. BM25 baseline):
  - MRR@10: **+28.68%**
  - Recall@10: **+20.77%**
  - Additional latency: only **+46.2 ms**
- **Practical on Consumer Hardware**: Handles million-scale documents with real-time response on RTX 5070 Ti
- **Complete Implementation**: Parallel index building, retrieval engine, semantic reranker, and Streamlit interactive demo

---

## 系统架构 / System Architecture

```
用户查询
    │
    ▼
[BM25 粗排]  →  Top-50 候选集（快速、高召回）
    │
    ▼
[BERT Cross-Encoder 重排] → 语义打分 + 重排序
    │
    ▼
Top-10 结果 + 可视化界面 (Streamlit)
```

**核心模块**：
- `data_loader.py`: 数据清洗、预处理与并行分词
- `run_bm25.py`: BM25 倒排索引构建与序列化
- `search_engine.py`: 核心检索引擎（BM25 + BERT 推理）
- `app.py`: Streamlit 可视化交互界面
- `run_bert.py` / 相关脚本: BERT 重排序相关逻辑

---

## 技术栈 / Tech Stack

- **Python 3.11**
- **Jieba** (中文分词)
- **rank_bm25** (BM25 实现)
- **sentence-transformers** + **cross-encoder/mmarco-mMiniLMv2** (轻量级语义模型)
- **Streamlit** (可视化界面)
- **PyTorch** + **multiprocessing** (并行加速)
- **T2Ranking** 数据集

---

## 如何运行 / How to Run

### 1. 环境准备

```bash
git clone https://github.com/yanyintingyou/bm25-bert-chinese-qa.git
cd bm25-bert-chinese-qa
pip install -r requirements.txt
```

### 2. 数据准备

本项目使用 **T2Ranking** 数据集（约 230 万中文段落）。由于数据集较大，仓库中未包含完整数据文件。你需要自行下载 T2Ranking 的 `collection.tsv`、`queries.dev.tsv` 和 `qrels.retrieval.dev.tsv` 文件，并放置在合适目录。

### 3. 构建索引

```bash
python work_code/run_bm25.py
```

（首次运行会进行并行分词与索引构建，并序列化保存）

### 4. 启动可视化界面

```bash
streamlit run work_code/app.py
```

启动后可在浏览器中交互测试 “纯 BM25” 与 “BM25 + BERT” 两种模式，并实时查看排序变化与延迟。

---

## 项目说明 / Project Note

**重要提醒**：

本仓库代码为本人**计算机科学与技术专业自考助学本科二学历**毕业设计项目的留档版本。项目旨在探索 “BM25 + BERT” 两阶段架构在中文检索场景下的可行性与工程实现。

由于本人为非全日制自考学生，正式计算机科班训练有限，代码实现、工程规范性和部分细节可能存在不足。上传此仓库仅用于个人学习成果存档与交流，**烦请谅解代码水平**。

如果您对两阶段检索、BM25 与神经重排序结合感兴趣，欢迎参考或改进。

---

## 参考 / References

- 毕业设计论文：《基于BM25的中文问答系统检索方法研究与实现》
- T2Ranking: A large-scale Chinese Benchmark for Passage Ranking
- 原理解释与实验分析详见论文第四章

---

## License

Apache-2.0 License

---

**感谢阅读**。如果这个项目对你有帮助，欢迎 Star 或提出改进建议！
