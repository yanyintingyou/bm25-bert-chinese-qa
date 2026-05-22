import os
import pandas as pd

# ==================== 路径配置（基于脚本位置，更健壮） ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')   # data 目录放在项目根目录
DATA_DIR = os.path.abspath(DATA_DIR)

COLLECTION_PATH = os.path.join(DATA_DIR, 'collection.tsv')
QUERIES_PATH = os.path.join(DATA_DIR, 'queries.dev.tsv')
QRELS_PATH = os.path.join(DATA_DIR, 'qrels.retrieval.dev.tsv')


def load_data():
    """加载 T2Ranking 数据集"""
    if not os.path.exists(COLLECTION_PATH):
        raise FileNotFoundError(
            f"找不到 {COLLECTION_PATH}，请将 T2Ranking 的 collection.tsv 放在 data/ 目录下"
        )

    print("正在加载数据集...")
    docs_df = pd.read_csv(COLLECTION_PATH, sep='\t', header=None, names=['pid', 'text'])

    queries_df = None
    if os.path.exists(QUERIES_PATH):
        queries_df = pd.read_csv(QUERIES_PATH, sep='\t', header=None, names=['qid', 'query'])

    qrels_df = None
    if os.path.exists(QRELS_PATH):
        qrels_df = pd.read_csv(QRELS_PATH, sep='\t', header=None, names=['qid', 'pid', 'relevance'])

    print(f"加载完成：{len(docs_df)} 篇文档")
    return docs_df, queries_df, qrels_df


if __name__ == "__main__":
    docs, queries, qrels = load_data()
    print(docs.head(2))