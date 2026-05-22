[中文](README.md) | [English](README.en.md)

# bm25-bert-chinese-qa

**基于 "BM25 粗排 + Cross-Encoder BERT 重排" 的两阶段中文问答检索系统**

> ⚠️ **项目说明**：本仓库为本人**计算机科学与技术专业（自考助学本科二学历）**毕业设计项目的代码留档仓库。仅上传核心代码部分。非科班出身，水平有限，烦请谅解。

一个在消费级硬件上实现百万级中文文档实时语义检索的工程实践项目。

---

## 项目简介

针对大规模中文问答场景下“语义匹配精度”与“检索实时性”难以兼顾的问题，本项目实现了一种 **“BM25 粗排 + BERT 重排”** 的两阶段检索架构。

系统利用 BM25 算法快速从百万级文档中召回候选集，再通过轻量级 Cross-Encoder BERT 模型进行深层语义重排序，在保证低延迟的同时显著提升检索质量。

项目基于 **T2Ranking** 大规模中文基准数据集验证，并提供了 Streamlit 可视化交互界面。

## 核心成果

在 T2Ranking 验证集上的实验结果（相比纯 BM25 基线）：

| 指标          | 提升幅度   | 说明                     |
|---------------|------------|--------------------------|
| **MRR@10**    | **+28.68%** | 平均倒数排名显著改善     |
| **Recall@10** | **+20.77%** | 前10召回率明显提升       |
| 额外延迟      | **+46.2ms** | 语义重排带来的边际成本   |

在 RTX 5070 Ti 等消费级硬件上，系统能够以毫秒级响应处理百万级中文文档检索。

## 系统架构

```
用户自然语言查询
        │
        ▼
[第一阶段] BM25 粗排（并行倒排索引）
   → 快速召回 Top-50 候选文档
        │
        ▼
[第二阶段] Cross-Encoder BERT 重排
   → 深层语义交互打分 + 重排序
        │
        ▼
Top-K 结果 + Streamlit 可视化界面
```

**主要模块**：
- `data_loader.py`：数据清洗、Jieba 分词与并行预处理
- `run_bm25.py`：构建 BM25 倒排索引并序列化
- `search_engine.py`：核心检索引擎（BM25 + BERT 推理）
- `app.py`：Streamlit 可视化演示界面

## 技术栈

- Python 3.11 + Jieba（中文分词）
- rank_bm25（BM25 实现）
- sentence-transformers + MiniLM 蒸馏模型（轻量级 Cross-Encoder）
- Streamlit（交互界面）
- PyTorch + multiprocessing（并行加速）

## 快速开始

```bash
git clone https://github.com/yanyintingyou/bm25-bert-chinese-qa.git
cd bm25-bert-chinese-qa
pip install -r requirements.txt

# 构建索引（首次运行较慢）
python work_code/run_bm25.py

# 启动可视化界面
streamlit run work_code/app.py
```

> **注意**：完整 T2Ranking 数据集（约230万文档）体积较大，仓库中未包含原始数据文件。请自行下载 `collection.tsv` 等文件后使用。

## 项目性质说明

本项目是本人计算机科学与技术专业**自考助学本科二学历**的毕业设计留档版本。

由于是非全日制自考背景，正式系统性计算机训练相对有限，代码的工程规范性、健壮性和部分实现细节可能存在不足。上传此仓库主要用于个人学习成果存档与技术交流，**恳请理解与谅解**。

如果您对“传统检索模型 + 轻量神经重排序”这种实用架构感兴趣，欢迎参考、批评或改进。

## 参考资料

- 毕业设计论文：《基于BM25的中文问答系统检索方法研究与实现》
- T2Ranking: A Large-scale Chinese Benchmark for Passage Ranking (Xie et al., 2023)

## License

Apache-2.0 License

---

**感谢阅读**。如果这个项目对你有帮助，欢迎 Star ⭐ 或提出宝贵意见！