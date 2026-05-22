import os
import pickle
import numpy as np
from sentence_transformers import CrossEncoder

# ==================== 路径配置（推荐写法：基于脚本位置，更健壮） ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# cache_data 目录放在 work_code 的上级目录（即项目根目录）
CACHE_DIR = os.path.join(SCRIPT_DIR, '..', 'cache_data')
CACHE_DIR = os.path.abspath(CACHE_DIR)

BM25_PATH = os.path.join(CACHE_DIR, 'bm25_model.pkl')
DATA_PATH = os.path.join(CACHE_DIR, 'raw_data_dfs.pkl')

os.makedirs(CACHE_DIR, exist_ok=True)


class SearchEngine:
    def __init__(self, use_bert=True):
        self.use_bert = use_bert
        self.bm25 = None
        self.docs_df = None
        self.reranker = None
        self.load_resources()

    def load_resources(self):
        """加载 BM25 模型和文档数据"""
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                f"找不到 {DATA_PATH}，请先运行 work_code/run_bm25.py 生成索引"
            )

        if not os.path.exists(BM25_PATH):
            raise FileNotFoundError(f"找不到 {BM25_PATH}，请先运行 work_code/run_bm25.py 生成索引")

        print("正在加载 BM25 索引和文档数据...")
        with open(DATA_PATH, 'rb') as f:
            data_tuple = pickle.load(f)
            self.docs_df = data_tuple[0]

        with open(BM25_PATH, 'rb') as f:
            self.bm25 = pickle.load(f)

        if self.use_bert:
            print("正在加载 BERT 重排序模型...")
            self.reranker = CrossEncoder(
                'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', max_length=512
            )

        print("资源加载完成！")

    def search(self, query, top_k=10, candidate_k=50):
        """两阶段检索：BM25 粗排 + BERT 重排"""
        if self.bm25 is None or self.docs_df is None:
            raise RuntimeError("模型未加载，请检查索引文件")

        # 简单分词（实际项目中建议使用 jieba.lcut）
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(scores)[::-1][:candidate_k]

        candidates = []
        for idx in top_n_indices:
            doc = self.docs_df.iloc[idx]
            candidates.append({
                'pid': doc['pid'],
                'doc_text': doc['text'],
                'bm25_score': float(scores[idx])
            })

        if not self.use_bert or self.reranker is None:
            for i, cand in enumerate(candidates[:top_k]):
                cand['final_score'] = cand['bm25_score']
                cand['rank'] = i + 1
            return candidates[:top_k]

        # BERT 重排
        model_inputs = [[query, cand['doc_text']] for cand in candidates]
        bert_scores = self.reranker.predict(model_inputs)

        for i, cand in enumerate(candidates):
            cand['bert_score'] = float(bert_scores[i])

        candidates.sort(key=lambda x: x.get('bert_score', 0), reverse=True)

        for i, cand in enumerate(candidates[:top_k]):
            cand['final_score'] = cand['bert_score']
            cand['rank'] = i + 1

        return candidates[:top_k]


if __name__ == "__main__":
    engine = SearchEngine(use_bert=True)
    results = engine.search("什么是人工智能", top_k=5)
    for r in results:
        print(r['rank'], r['pid'], round(r.get('final_score', 0), 4))