# bm25-bert-chinese-qa
基于"BM25 粗排 + Cross-Encoder BERT 重排"的两阶段中文问答检索系统。在消费级硬件上（RTX 5070 Ti）实现百万级文档的毫秒级响应，MRR@10 相对 BM25 基线提升 28.68%。包含完整的索引构建、语义重排引擎与 Streamlit 可视化交互界面，基于 T2Ranking 数据集验证。
