# -*- coding: utf-8 -*-
#功能，数据的读取与清洗
import pandas as pd
import csv
import os
import re


DATA_DIR = 'data' 
COLLECTION_PATH = os.path.join(DATA_DIR, 'collection.tsv')
QUERIES_PATH = os.path.join(DATA_DIR, 'queries.dev.tsv')
QRELS_PATH = os.path.join(DATA_DIR, 'qrels.retrieval.dev.tsv')


def clean_text(text):
    if pd.isna(text): # 处理空值
        return ""
    if not isinstance(text, str):
        text = str(text)
        
    # 1. 去除 HTML 标签 (如 <br>, <img>, <a>...</a>)
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. 去除特殊的空白字符 (如 \t, \n, \r) 并替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    # 3. 去除首尾空格
    return text.strip()

def load_data():
    """
    加载并清洗 T2Ranking 数据集
    返回: (文档DataFrame, 问题DataFrame, 答案字典)
    """
    print("[系统启动] 正在加载数据，请稍候...")

    # ----------------------------------------------------
    # 1. 加载文档库 (Collection)
    # ----------------------------------------------------
    print(f"正在读取文档库: {COLLECTION_PATH}")
    if not os.path.exists(COLLECTION_PATH):
        raise FileNotFoundError(f"未找到文件: {COLLECTION_PATH}，请检查")

    try:
        df_collection = pd.read_csv(
            COLLECTION_PATH, 
            sep='\t', 
            header=0,               # 关键修改：第一行是表头，跳过它
            names=['pid', 'text'],  # 显式指定列名，防止表头不规范
            quoting=csv.QUOTE_NONE, # 核心：不处理引号，防止报错
            dtype={'pid': str, 'text': str}, # 强制 ID 为字符串
            on_bad_lines='skip'     # 跳过极其离谱的坏行（容错）
        )
        
        # 执行清洗
        print("正在清洗HTML标签")
        # 使用向量化操作加速清洗
        df_collection['text'] = df_collection['text'].apply(clean_text)
        
        print(f"文档库加载完成，共 {len(df_collection):,} 条文档。")
    except Exception as e:
        print(f"文档库加载失败: {e}，请检查")
        return None, None, None

    # ----------------------------------------------------
    # 2. 加载查询集 (Queries)
    # ----------------------------------------------------
    print(f"正在读取查询集: {QUERIES_PATH}")
    try:
        df_queries = pd.read_csv(
            QUERIES_PATH, 
            sep='\t', 
            header=0,               # 关键修改：跳过表头
            names=['qid', 'text'],
            quoting=csv.QUOTE_NONE,
            dtype={'qid': str, 'text': str}
        )
        print(f"查询集加载完成，共 {len(df_queries):,} 条问题。")
    except Exception as e:
        print(f"查询集加载失败: {e}，请检查")
        return None, None, None

    # ----------------------------------------------------
    # 3. 加载标准答案 (Qrels)
    # ----------------------------------------------------
    print(f"正在读取标准答案: {QRELS_PATH}")
    try:
        df_qrels = pd.read_csv(
            QRELS_PATH, 
            sep='\t', 
            header=0,               # 关键修改：跳过表头
            names=['qid', 'pid'],
            usecols=[0, 1],         # 只取前两列，忽略可能的第三列
            dtype={'qid': str, 'pid': str}
        )
        
        # 转换为字典: {'qid': ['pid1', 'pid2']}
        # 这里的逻辑是：把同一个问题的所有正确答案合并成一个列表
        qrels_dict = df_qrels.groupby('qid')['pid'].apply(list).to_dict()
        print(f"标准答案加载完成，涵盖 {len(qrels_dict):,} 个测试问题。")
        
    except Exception as e:
        print(f"标准答案加载失败: {e}，请检查")
        return None, None, None

    print("\n 所有数据加载清洗完毕！")
    return df_collection, df_queries, qrels_dict

# ================= 单元测试代码 =================
if __name__ == "__main__":
    # 这里的代码只有直接运行此文件时才会执行，被别人 import 时不会执行
    docs, queries, qrels = load_data()
    
    if docs is not None:
        print("\n数据抽样:")
        print("-" * 40)
        print("[Collection 样本]")
        print(docs.head(3)) 
        
        print("\n[Query 样本]")
        print(queries.head(3))
        
        print("\n[Qrels 样本]")

        first_qid = list(qrels.keys())[0]
        print(f"问题 ID: {first_qid} -> 正确答案文档 IDs: {qrels[first_qid]}")
