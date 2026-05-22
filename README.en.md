[中文](README.md) | [English](README.en.md)

# bm25-bert-chinese-qa

**Two-Stage Chinese Question Answering Retrieval System**

**"BM25 Coarse Ranking + Cross-Encoder BERT Reranking"**

> ⚠️ **Project Note**: This repository is an archive of the author's graduation project for a **second bachelor's degree (self-taught / 自考助学本科二学历)** in Computer Science and Technology. Only core code is uploaded. The author has a non-traditional educational background with limited formal CS training. Please understand any limitations in code quality or engineering practices.

A practical engineering project that achieves real-time semantic retrieval over million-scale Chinese documents on consumer-grade hardware.

---

## Project Overview

To address the trade-off between semantic matching accuracy and retrieval latency in large-scale Chinese QA scenarios, this project implements a **two-stage retrieval architecture**: BM25 for fast candidate retrieval + lightweight Cross-Encoder BERT for semantic reranking.

The system first uses BM25 with a parallel inverted index to quickly retrieve a Top-50 candidate set from millions of documents, then applies a distilled Cross-Encoder BERT model (MiniLM-based) to perform deep semantic interaction and rerank the results.

The project is validated on the **T2Ranking** large-scale Chinese passage ranking benchmark and includes a Streamlit interactive visualization demo.

## Key Results

Experimental results on the T2Ranking validation set (compared to pure BM25 baseline):

| Metric        | Improvement   | Description                     |
|---------------|---------------|---------------------------------|
| **MRR@10**    | **+28.68%**   | Significant improvement in ranking quality |
| **Recall@10** | **+20.77%**   | Better coverage of relevant documents     |
| Added Latency | **+46.2 ms**  | Marginal cost of semantic reranking       |

The system delivers millisecond-level responses for million-scale Chinese document retrieval on consumer hardware such as RTX 5070 Ti.

## System Architecture

```
User Natural Language Query
        │
        ▼
[Stage 1] BM25 Coarse Ranking (Parallel Inverted Index)
   → Fast retrieval of Top-50 candidates
        │
        ▼
[Stage 2] Cross-Encoder BERT Reranking
   → Deep semantic interaction scoring + reranking
        │
        ▼
Top-K Results + Streamlit Visualization Interface
```

**Main Modules**:
- `data_loader.py`: Data cleaning, Jieba tokenization, and parallel preprocessing
- `run_bm25.py`: Build and serialize BM25 inverted index
- `search_engine.py`: Core retrieval engine (BM25 + BERT inference)
- `app.py`: Streamlit interactive demo interface

## Tech Stack

- Python 3.11 + Jieba (Chinese word segmentation)
- rank_bm25 (BM25 implementation)
- sentence-transformers + MiniLM distilled model (lightweight Cross-Encoder)
- Streamlit (interactive UI)
- PyTorch + multiprocessing (parallel acceleration)

## Quick Start

```bash
git clone https://github.com/yanyintingyou/bm25-bert-chinese-qa.git
cd bm25-bert-chinese-qa
pip install -r requirements.txt

# Build index (first run may take time)
python work_code/run_bm25.py

# Launch visualization interface
streamlit run work_code/app.py
```

> **Note**: The full T2Ranking dataset (~2.3 million passages) is large and not included in this repository. Please download `collection.tsv` and related files yourself.

## Project Background

This is an archival repository of the author's graduation project for a **second bachelor's degree in Computer Science and Technology (self-taught program)**.

As a non-traditional, part-time self-taught student, the author has relatively limited systematic formal computer science training. The code may have shortcomings in engineering standards, robustness, and implementation details. This repository is uploaded primarily for personal learning outcome archiving and technical exchange. **Your understanding is greatly appreciated**.

If you are interested in practical "traditional retrieval + lightweight neural reranking" architectures, feel free to reference, critique, or improve upon it.

## References

- Graduation Thesis: 《基于BM25的中文问答系统检索方法研究与实现》
- T2Ranking: A Large-scale Chinese Benchmark for Passage Ranking (Xie et al., 2023)

## License

Apache-2.0 License

---

**Thank you for reading.** If this project is helpful to you, feel free to Star ⭐ or share your feedback!