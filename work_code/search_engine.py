# -*- coding: utf-8 -*-
# search_engine.py
# 功能：封装核心检索逻辑（后端引擎），供 UI 调用
# 包含：数据加载、文本分词、BM25检索、BERT重排序

import os
import pickle
import time
import torch
import jieba
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

# ================= 配置区域 =================
CACHE_DIR = 'cache_data'
BM25_PATH = os.path.join(CACHE_DIR, 'bm25_model.pkl')
DATA_PATH = os.path.join(CACHE_DIR, 'raw_data_dfs.pkl')
BERT_MODEL_NAME = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'

# 停用词表 (必须与构建索引时保持完全一致！)
STOP_WORDS = set(['的', '了', '在', '是', '我', '有', '和', '就', 
                  '不', '人', '都', '一', '一个', '上', '也', '很', 
                  '到', '说', '要', '去', '你', '会', '着', '没有', 
                  '看', '好', '自己', '这'])

class SearchEngine:
    def __init__(self):
        """
        初始化搜索引擎：
        1. 加载BM25索引
        2. 加载文档原文库
        3. 加载BERT模型
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[Engine] 启动硬件设备: {self.device}")
        
        self.bm25 = None
        self.docs_df = None
        self.reranker = None
        self.is_loaded = False

    def load_resources(self):
        """
        加载所有静态资源
        """
        if self.is_loaded:
            return

        t0 = time.time()
        
        # 1. 加载基础数据 (Docs DF)
        print("[Engine] 正在加载文档库...")
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"找不到 {DATA_PATH}，请先运行run_bm25.py")
        
        # raw_data_dfs.pkl 包含 (docs_df, queries_df, qrels_dict)
        # 我们只需要 docs_df 来查原文
        with open(DATA_PATH, 'rb') as f:
            data_tuple = pickle.load(f)
            self.docs_df = data_tuple[0] # docs_df 是第一个
        
        # 2. 加载 BM25 索引
        print("[Engine]正在加载BM25索引")
        if not os.path.exists(BM25_PATH):
            raise FileNotFoundError(f"找不到 {BM25_PATH}")
        
        with open(BM25_PATH, 'rb') as f:
            self.bm25 = pickle.load(f)

        # 3. 加载 BERT 模型
        print(f"[Engine]正在加载BERT模型 ({self.device})...")
        # max_length=512 是物理极限
        self.reranker = CrossEncoder(BERT_MODEL_NAME, max_length=512, device=self.device)
        
        self.is_loaded = True
        print(f"[Engine] 系统就绪,总耗时: {time.time() - t0:.2f}s")
        print(f" 文档总数: {len(self.docs_df):,}")
        print(f" 词表大小: {len(self.bm25.idf):,}")

    def _tokenizer(self, text):
        """文本分词 """
        if not isinstance(text, str): return []
        tokens = jieba.lcut(text)
        return [t for t in tokens if t not in STOP_WORDS and len(t.strip()) > 0]

    def _format_result(self, pid, score, rank, tag="BM25"):
        """格式化单条结果"""
        # 从 docs_df 中查找原文
        # 假设 docs_df 有 'pid' 和 'text' 列
        # 为了速度，这里假设 docs_df 已经按 pid 排序或我们直接用 index 查找
        # 注意：bm25.get_scores 返回的索引对应 docs_df 的行号 (iloc)
        
        # 安全获取文档内容
        try:
            # 这里的 pid 其实是 dataframe 的 index (iloc)
            # 如果你的 bm25 是基于 list(docs_df['text']) 构建的，那么索引是一一对应的
            row = self.docs_df.iloc[pid] 
            doc_id = row['pid']
            doc_text = row['text']
        except IndexError:
            doc_id = "Unknown"
            doc_text = "[文档索引越界]"

        return {
            "rank": rank,
            "id": doc_id,
            "score": round(float(score), 4),
            "content": doc_text,
            "tag": tag
        }

    def search_bm25(self, query, top_k=10):
        """
        【模式 A】纯BM25检索
        """
        if not query.strip(): return []
        
        # 1. 分词
        tokenized_query = self._tokenizer(query)
        
        # 2. 打分
        scores = self.bm25.get_scores(tokenized_query)
        
        # 3. 排序并取 Top-K
        # argsort 返回的是从小到大的索引，所以要切片取最后 K 个并反转
        top_n_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for rank, idx in enumerate(top_n_indices):
            res = self._format_result(idx, scores[idx], rank + 1, tag="BM25")
            results.append(res)
            
        return results

    def search_bert(self, query, top_k=10, candidate_k=50):
        """
        【模式 B】BM25召回 + BERT重排序
        candidate_k: 给BERT多少个候选文档 (建议 50-100)
        """
        if not query.strip(): return []

        # 阶段1:BM25粗排
        tokenized_query = self._tokenizer(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # 获取前 candidate_k 个候选
        candidate_indices = np.argsort(bm25_scores)[-candidate_k:][::-1]
        
        #阶段2:构造BERT输入对
        candidates = []
        model_inputs = []
        
        for idx in candidate_indices:
            row = self.docs_df.iloc[idx]
            doc_text = row['text'][:300] # 截取前300字
            
            candidates.append({
                'original_idx': idx,  # 记录原始索引以便查原文
                'doc_text': doc_text,
                'pid': row['pid']
            })
            model_inputs.append([query, doc_text])
            
        #阶段3: GPU推理
        # predict 返回 numpy array
        bert_scores = self.reranker.predict(model_inputs, show_progress_bar=False)
        
        #阶段4:写入分数并重排
        for i, score in enumerate(bert_scores):
            candidates[i]['bert_score'] = score
            
        #按BERT分数降序排列
        candidates.sort(key=lambda x: x['bert_score'], reverse=True)
        
        #阶段5:格式化输出Top-K
        results = []
        for rank, cand in enumerate(candidates[:top_k]):
            # 直接调用 _format_result
            res = self._format_result(
                cand['original_idx'], 
                cand['bert_score'], 
                rank + 1, 
                tag="AI-Rerank"
            )
            results.append(res)
            
        return results

    def get_system_status(self):
        """供管理员面板使用的状态监控"""
        status = {
            "memory_docs": f"{len(self.docs_df):,} 条",
            "device": self.device,
            "gpu_mem": "N/A"
        }
        if self.device == 'cuda':
            # 获取显存占用 (MB)
            mem = torch.cuda.memory_allocated(0) / 1024 / 1024
            status["gpu_mem"] = f"{mem:.1f} MB"
        
        return status

# ================= 单元测试 =================
if __name__ == "__main__":
    # 这里的代码只有直接运行此文件时才会执行
    # 用来测试 Engine 是否写对了，不用启动网页
    print("正在进行单元测试...")
    
    engine = SearchEngine()
    engine.load_resources() # 加载数据
    
    test_query = "显卡花屏怎么办"
    print(f"\n测试查询: {test_query}")
    
    print("\n模式 A:BM25")
    res_a = engine.search_bm25(test_query, top_k=3)
    for r in res_a:
        print(f"[Rank {r['rank']}] Score:{r['score']} | {r['content'][:30]}...")
        
    print("\n模式 B: BERT")
    res_b = engine.search_bert(test_query, top_k=3)
    for r in res_b:
        print(f"[Rank {r['rank']}] Score:{r['score']} | {r['content'][:30]}...")