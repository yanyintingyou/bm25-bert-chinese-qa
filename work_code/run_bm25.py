# -*- coding: utf-8 -*-
#功能：基于bm25算法对于问进行排序，查验运行效果
import pandas as pd
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import pickle
import os
import random
import gc
import openpyxl

from data_loader import load_data

CACHE_DIR = 'cache_data' # 缓存文件夹
SAMPLE_SIZE = 500     # 采样规模

# 停用词表
STOP_WORDS = set(['的', '了', '在', '是', '我', '有', '和', '就', 
                  '不', '人', '都', '一', '一个', '上', '也', '很', 
                  '到', '说', '要', '去', '你', '会', '着', '没有', 
                  '看', '好', '自己', '这'])



def ensure_dir(directory):
    """确保缓存目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_cache(data, filename):
    """保存数据到缓存"""
    ensure_dir(CACHE_DIR)
    filepath = os.path.join(CACHE_DIR, filename)
    print(f"正在保存中间结果到 {filepath} ...")
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print("保存完成")

def load_cache(filename):
    """尝试从缓存读取数据"""
    filepath = os.path.join(CACHE_DIR, filename)
    if os.path.exists(filepath):
        print(f"发现缓存文件 {filepath}，正在加载...")
        start = time.time()
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"加载完成，耗时{time.time() - start:.2f}秒")
        return data
    return None



def text_tokenizer(text):
    if not isinstance(text, str): return []
    tokens = jieba.lcut(text)
    return [token for token in tokens if token not in STOP_WORDS and len(token.strip()) > 0]

def parallel_tokenize(text_series, num_processes=None):
    if num_processes is None:
        num_processes = max(1, cpu_count() - 2) 
    
    print(f"启动{num_processes}个进程进行并行分词...")
    with Pool(num_processes) as pool:
        tokens_list = list(tqdm(pool.imap(text_tokenizer, text_series), 
                                total=len(text_series), 
                                desc="分词进度"))
    return tokens_list

def calculate_metrics(qrels, results_dict, k=10):
    mrr_sum = 0.0
    recall_sum = 0.0
    checked_queries = 0

    for qid, pred_pids in results_dict.items():
        if qid not in qrels: continue
        target_pids = set(qrels[qid])
        checked_queries += 1
        
        # MRR
        mrr = 0.0
        for rank, pid in enumerate(pred_pids[:k]):
            if pid in target_pids:
                mrr = 1.0 / (rank + 1)
                break
        mrr_sum += mrr
        
        # Recall
        hits = sum(1 for pid in pred_pids[:k] if pid in target_pids)
        if len(target_pids) > 0:
            recall_sum += hits / len(target_pids)
    
    if checked_queries == 0: return 0.0, 0.0
    return mrr_sum / checked_queries, recall_sum / checked_queries

def main():
    random.seed(42)
    np.random.seed(42)
    ensure_dir(CACHE_DIR)


    # Step 1: 加载数据 (带缓存)

    print("\n[Step 1/4] 准备数据...")
    cached_data = load_cache('raw_data_dfs.pkl')
    
    if cached_data:
        docs_df, queries_df, qrels_dict = cached_data
    else:
        # 如果没有缓存，从头加载
        docs_df, queries_df, qrels_dict = load_data()
        if docs_df is None: return
        # 保存缓存
        save_cache((docs_df, queries_df, qrels_dict), 'raw_data_dfs.pkl')


    # Step 2: 分词 (带缓存 - 最耗时步骤)

    print("\n[Step 2/4] 准备分词数据...")
    
    bm25 = load_cache('bm25_model.pkl')
    
    if bm25 is not None:
        print("（调试）发现索引缓存，跳过分词。")
        # 这里的 corpus_tokens 根本不会被加载，内存占用极低
    else:
        # 只有在真的没有索引时，才去动用分词数据
        print("未找到BM25索引缓存，尝试加载分词数据...")
        
        corpus_tokens = load_cache('corpus_tokens.pkl')
        
        if corpus_tokens is None:
            print("未找到分词缓存，开始执行并行分词，请等待10分钟左右")
            start_time = time.time()
            corpus_tokens = parallel_tokenize(docs_df['text'])
            print(f"分词完成，耗时: {time.time() - start_time:.2f} 秒")
            # 保存缓存
            save_cache(corpus_tokens, 'corpus_tokens.pkl')


    # Step 3: 构建 BM25 索引 (带缓存)
    print("\n[Step 3/4] 准备 BM25 索引...")
    
    if bm25 is None:
        print("未找到索引缓存，开始构建索引，请稍等")
        bm25 = BM25Okapi(corpus_tokens)
        print("索引构建完成")
        # 保存缓存
        save_cache(bm25, 'bm25_model.pkl')
    else:
        print("（调试）索引缓存已存在，本次跳过")
    
    #强制释放内存
    if 'corpus_tokens' in locals():
        print("正在释放分词数据内存")
        del corpus_tokens  # 删除变量引用
        gc.collect()       # 强制运行垃圾回收
        print("内存释放完毕")


    # Step 4: 检索与评估
    print(f"\n[Step 4/4] 开始检索测试 (采样{SAMPLE_SIZE}条)...")
    
    # 筛选有效问题
    valid_qids = [qid for qid in queries_df['qid'] if qid in qrels_dict]
    
    # 随机采样
    if len(valid_qids) > SAMPLE_SIZE:
        sampled_qids = random.sample(valid_qids, SAMPLE_SIZE)
    else:
        sampled_qids = valid_qids
    
    # 构建采样问题的映射字典
    sampled_queries_map = queries_df.set_index('qid').loc[sampled_qids]['text'].to_dict()
    
    results = {}
    start_retrieval = time.time()
    detailed_results = []
    bert_candidates_data = []
    
    
    for qid, q_text in tqdm(sampled_queries_map.items(), desc="检索中......"):
        # 1. 基础检索逻辑
        tokenized_query = text_tokenizer(q_text)
        scores = bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(scores)[-50:][::-1]
        
        # 2. 构造 Top-10 PIDs
        top_n_pids = [docs_df.iloc[i]['pid'] for i in top_n_indices]
        results[qid] = top_n_pids
        
        current_candidates = []
        # 提前获取标准答案，用于给候选打标签
        target_pids = set(qrels_dict.get(qid, [])) 
        
        for idx in top_n_indices:
            row = docs_df.iloc[idx]
            current_candidates.append({
                'pid': row['pid'],
                'doc_text': row['text'][:300],  #由于BERT的token限制，截取前300字给BERT
                'is_relevant': 1 if row['pid'] in target_pids else 0
            })
            
        bert_candidates_data.append({
            'qid': qid,
            'query_text': q_text,
            'candidates': current_candidates
        })
        
        # 3. 提取详细分析数据
        # A. 获取Top-1的详细信息
        top1_idx = top_n_indices[0]
        top1_pid = top_n_pids[0]
        top1_score = scores[top1_idx]
        # 获取 Top-1 文档的内容
        top1_text = docs_df.iloc[top1_idx]['text'][:100]
        
        # B. 计算标准答案的排名
        target_pids = set(qrels_dict.get(qid, []))
        rank = -1  # 默认 -1 表示未召回
        is_hit = 0
        
        for r, pid in enumerate(top_n_pids):
            if pid in target_pids:
                rank = r + 1
                is_hit = 1
                break
        
        # C. 存入列表
        detailed_results.append({
            'QID': qid,
            'Query': q_text,
            'Gold_PIDs': ','.join([str(p) for p in target_pids]), # 标准答案ID
            'Top1_PID': top1_pid,
            'Top1_Score': round(top1_score, 4), # 保留4位小数
            'Top1_Text': top1_text, # 预测文档内容
            'Rank': rank,  # 命中排名
            'Is_Hit': is_hit  # 是否命中
        })

    total_time = time.time() - start_retrieval
    print(f"检索完成，耗时: {total_time:.2f} 秒")
    
    # 保存Top-50候选集给BERT使用
    candidates_path = os.path.join(CACHE_DIR, 'candidates_top50_for_bert.pkl')
    print(f"正在保存BERT候选数据到{candidates_path}")
    with open(candidates_path, 'wb') as f:
        pickle.dump(bert_candidates_data, f)
    print("BERT候选数据保存完成")

    # 计算指标
    mrr_10, recall_10 = calculate_metrics(qrels_dict, results, k=10)

    print("正在生成实验报告(Excel)")
    
    # 1. Sheet A: 详细案例 (Case Analysis)
    details_df = pd.DataFrame(detailed_results)
    
    # 2. Sheet B: 实验总览 (Summary)
    summary_data = {
        'Metric': ['MRR@10', 'Recall@10', 'Total Time(s)', 'Sample Size', 'Timestamp'],
        'Value': [mrr_10, recall_10, total_time, SAMPLE_SIZE, time.strftime("%Y-%m-%d %H:%M")]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # 3. Sheet C: 系统内核参数 (System_Meta)
    # 提取BM25对象的内部属性
    meta_data = {
        'Parameter': [
            'Algorithm', 
            'k1 (Saturation)', 
            'b (Length Norm)', 
            'Epsilon', 
            'Corpus Size (Docs)', 
            'Avg Doc Length (Words)', 
            'Vocab Size (Unique Terms)'
        ],
        'Value': [
            'BM25Okapi',
            getattr(bm25, 'k1', 1.5),         # 默认 k1
            getattr(bm25, 'b', 0.75),         # 默认 b
            getattr(bm25, 'epsilon', 0.25),   # 默认 epsilon
            getattr(bm25, 'corpus_size', 0),  # 文档总数
            round(getattr(bm25, 'avgdl', 0), 2),     # 平均文档长度
            len(getattr(bm25, 'idf', {}))     # 词汇表大小
        ],
        'Description': [
            'Ranking Model Name',
            'Controls term frequency saturation',
            'Controls document length normalization',
            'Floor value for negative IDF',
            'Total number of documents indexed',
            'Average number of tokens per document',
            'Total number of unique Chinese terms in index'
        ]
    }
    meta_df = pd.DataFrame(meta_data)

    # 4. Sheet D: 特征权重分析(Top_IDF_Words)
    # 提取IDF(逆文档频率) 最高的词
    if hasattr(bm25, 'idf'):
        # 排序：取出权重最高的50个词和最低的50个词
        sorted_idf = sorted(bm25.idf.items(), key=lambda x: x[1], reverse=True)
        top_50_words = sorted_idf[:50]
        
        idf_df = pd.DataFrame(top_50_words, columns=['Term', 'IDF_Score'])
        idf_df['Type'] = 'Top-50 (Rare/Discriminative)'
    else:
        idf_df = pd.DataFrame(columns=['Term', 'IDF_Score', 'Type'])

    # 5. 写入文件 (包含 4 个 Sheet)
    output_file = 'BM25_Experiment_Report.xlsx'
    try:
        with pd.ExcelWriter(output_file) as writer:
            summary_df.to_excel(writer, sheet_name='1_Summary', index=False)
            meta_df.to_excel(writer, sheet_name='2_System_Meta', index=False)
            idf_df.to_excel(writer, sheet_name='3_High_IDF_Terms', index=False)
            details_df.to_excel(writer, sheet_name='4_Case_Analysis', index=False)
            
        print(f"报表已生成: {output_file}")

    except Exception as e:
        print(f"Excel 保存失败: {e}，请检查")
    
    
    print("="*50)
    print(f"部分实验结果(Sampling N={SAMPLE_SIZE})")
    print(f"MRR@10    : {mrr_10:.4f}")
    print(f"Recall@10 : {recall_10:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()