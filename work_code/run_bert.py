# -*- coding: utf-8 -*-
# 功能：读取BM25的Top50候选集，利用BERT(Cross-Encoder) 进行重排，查验效果优化

import pickle
import os
import time
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import sys


CACHE_DIR = 'cache_data'
MODEL_NAME = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'

# 最终评估指标截断值 (和 BM25 保持一致，公平对比)
TOP_K_EVAL = 10 

# ================= 工具函数 =================

def load_pkl(filename):
    filepath = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(filepath):
        print(f"找不到文件{filepath}")
        print("请先运行run_bm25.py生成候选数据！")
        sys.exit(1)
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def calculate_metrics_bert(results, qrels_dict, k=10):
    """
    计算重排序后的 MRR 和 严格 Recall (修正版)
    results: BERT 处理后的结果列表
    qrels_dict: 标准答案字典 {'qid': ['pid1', 'pid2'...]}
    """
    mrr_sum = 0.0
    recall_sum = 0.0
    valid_queries = 0
    
    for item in results:
        qid = str(item['qid']) # 确保 QID 是字符串格式，防止匹配失败
        
        # 如果这个 query 不在标准答案里，跳过不计
        if qid not in qrels_dict:
            continue
            
        valid_queries += 1
        target_pids = set(qrels_dict[qid]) # 获取该问题的所有标准答案 ID
        total_relevant = len(target_pids)  # 分母：一共有几个正确答案
        
        # 取出 BERT 重排后的前 K 个候选
        top_k_candidates = item['candidates'][:k]
        
        # 统计命中情况
        hits_rank = [] 
        hits_count = 0
        
        for rank, cand in enumerate(top_k_candidates):
            if str(cand['pid']) in target_pids: # 确保 ID 类型一致
                hits_rank.append(rank + 1)
                hits_count += 1
        
        # 1. 计算 MRR (1 / 第一个命中位置)
        if hits_count > 0:
            mrr_sum += 1.0 / hits_rank[0]
            
        # 2. 计算 Recall (命中个数 / 总答案个数)
        if total_relevant > 0:
            recall_sum += hits_count / total_relevant

    if valid_queries == 0: return 0.0, 0.0
    return mrr_sum / valid_queries, recall_sum / valid_queries



def main():
    print("="*60)
    print("启动BERT重排序系统")
    print("="*60)

    # 1. 硬件检测与锁定
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        print(f"硬件加速已开启: {gpu_name}")
    else:
        device = 'cpu'
        print("未检测到GPU，正在使用CPU模式 ")
        print("请检查PyTorch版本是否支持")

    # 2. 加载候选数据
    print("\n [Step 1] 读取 BM25 候选集 (Top-50)")
    t0 = time.time()
    query_groups = load_pkl('candidates_top50_for_bert.pkl')
    print("   正在加载标准答案 (Qrels)...")
    raw_data = load_pkl('raw_data_dfs.pkl')
    if raw_data is None:
        print("错误：找不到文件，无法计算准确率。")
        return
    _, _, qrels_dict = raw_data
    print(f" 加载完成，共{len(query_groups)}个查询，耗时 {time.time()-t0:.2f} 秒")

    # 3. 加载 BERT 模型
    print(f"\n [Step 2] 加载Cross-Encoder模型:{MODEL_NAME}")
    # max_length=512 是 BERT 的物理极限，我们已经在 BM25 阶段截取了 300 字，这里安全
    reranker = CrossEncoder(MODEL_NAME, max_length=512, device=device)
    print("模型加载完毕")

    # 4. 执行重排序 (Core Loop)
    print("\n[Step 3] 开始推理")
    start_time = time.time()
    
    detailed_report = [] # 用于存 Excel 报表

    for q_data in tqdm(query_groups, desc="BERT Reranking"):
        query = q_data['query_text']
        
        # 【修改点1】强制将 QID 转为字符串，防止 int/str 不匹配
        qid = str(q_data['qid']) 
        
        candidates = q_data['candidates']
        
        # 【修改点2】提前获取该问题的标准答案集合，用于现场比对
        target_pids = set(qrels_dict.get(qid, []))
        
        # 4.1 构造模型输入 Pair
        model_inputs = [[query, cand['doc_text']] for cand in candidates]
        
        # 4.2 GPU 推理
        scores = reranker.predict(model_inputs, show_progress_bar=False)
        
        # 4.3 写入分数
        for i, score in enumerate(scores):
            candidates[i]['bert_score'] = float(score)
            candidates[i]['old_rank'] = i + 1 
            
        # 4.4 核心动作：按 BERT 分数重新排序 (降序)
        candidates.sort(key=lambda x: x['bert_score'], reverse=True)
        
       # 4.5 提取 Top-1 用于分析
        best_cand = candidates[0]

        # --- [修改开始] 获取用于绘图的精确数值排名 ---
        gold_old_rank = -1
        gold_new_rank = -1
        
        # 在排序后的【全量】候选集中寻找正确答案 (不再局限于 Top-10)
        # 目的：获取真实的排名数值 (如 12, 25)，以便画图
        for r, cand in enumerate(candidates):
            if str(cand['pid']) in target_pids:
                gold_new_rank = r + 1            # 1-based index (BERT排名)
                gold_old_rank = cand['old_rank'] # 原始 BM25 排名
                break
        
        # --- 计算指标逻辑 (Top-K截断) ---
        is_hit = 0
        if gold_new_rank != -1 and gold_new_rank <= TOP_K_EVAL:
            is_hit = 1
            
        # --- 构造原来的字符串用于肉眼看 ---
        gold_rank_change = "未召回"
        if gold_new_rank != -1:
            if gold_new_rank <= TOP_K_EVAL:
                gold_rank_change = f"{gold_old_rank} ➔ {gold_new_rank}"
            else:
                gold_rank_change = f"{gold_old_rank} ➔ >{TOP_K_EVAL}"

        # --- 写入增强版 Report ---
        detailed_report.append({
            'QID': qid, 
            'Query': query,
            # [关键修改] 新增两列纯数字，方便 Excel 画散点图
            'Old_Rank': gold_old_rank,  # X轴数据
            'New_Rank': gold_new_rank,  # Y轴数据
            # -------------------------------------
            'Is_Hit': is_hit,           # 0/1 用于计算召回率
            'Rank_Change': gold_rank_change, # 保留字符串格式方便查看
            'Top1_PID': best_cand['pid'],
            'Top1_Score': round(best_cand['bert_score'], 4),
            'Top1_Text': best_cand['doc_text'][:100]
        })
        # --- [修改结束] ---

    total_time = time.time() - start_time
    avg_time = (total_time / len(query_groups)) * 1000
    print(f"\n推理完成，总耗时:{total_time:.2f}秒")
    print(f"平均每题耗时:{avg_time:.1f}ms")

    # 5. 计算最终指标
    print(f"\n [Step 4] 计算评估指标 (对比 @{TOP_K_EVAL})...")
    mrr, recall = calculate_metrics_bert(query_groups, qrels_dict, k=TOP_K_EVAL)
    
    print("="*60)
    print("BM25 + BERT Rerank结果")
    print("="*60)
    print(f"MRR@{TOP_K_EVAL}    : {mrr:.4f} ") 
    print(f"Recall@{TOP_K_EVAL} : {recall:.4f} ")
    print("="*60)

    # 6. 保存详细报表
    output_file = 'BERT_Rerank_Report.xlsx'
    print(f"\n正在保存分析报表至{output_file}")
    df = pd.DataFrame(detailed_report)
    

    # 调整列顺序，把 Old_Rank 和 New_Rank 放在前面方便对比
    cols = ['QID', 'Query', 'Is_Hit', 'Old_Rank', 'New_Rank', 'Rank_Change', 'Top1_PID', 'Top1_Score', 'Top1_Text']
    df = df[cols]
    
    df.to_excel(output_file, index=False)
    print("全部完成。")

if __name__ == "__main__":
    main()