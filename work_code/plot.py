# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 加载数据
file_path = 'BERT_Rerank_Report.xlsx'
df = pd.read_excel(file_path)

# 过滤有效数据：只分析在两个阶段都出现在 Top-50 内的查询 (Rank > 0)
valid_df = df[(df['Old_Rank'] > 0) & (df['New_Rank'] > 0)].copy()

total_valid = len(valid_df)
    
# 1. 基础优化率
improved_count = len(valid_df[valid_df['New_Rank'] < valid_df['Old_Rank']])
improved_pct = (improved_count / total_valid) * 100
    
degraded_count = len(valid_df[valid_df['New_Rank'] > valid_df['Old_Rank']])
degraded_pct = (degraded_count / total_valid) * 100
    
# 2. 中位数跃迁 (Median Shift)
median_old = valid_df['Old_Rank'].median()
median_new = valid_df['New_Rank'].median()
    
# 3. 深层挽救 (Deep Rescue): Old > 10 -> New <= 10
rescue_count = len(valid_df[(valid_df['Old_Rank'] > 10) & (valid_df['New_Rank'] <= 10)])
rescue_pct = (rescue_count / total_valid) * 100
    
# 4. 精准定位 (Precision Hit): Old <= 10 -> New == 1
precision_count = len(valid_df[(valid_df['Old_Rank'] <= 10) & (valid_df['New_Rank'] == 1)])
precision_pct = (precision_count / total_valid) * 100

# 打印核验数据 (方便您写论文时再次确认)
print("-" * 30)
print(f"统计数据核验 总样本 N={total_valid}")
print(f"中位数排名: BM25={median_old} -> BERT={median_new}")
print(f"正向优化: {improved_count} ({improved_pct:.1f}%)")
print(f"反向退化: {degraded_count} ({degraded_pct:.1f}%)")
print(f"深层挽救 (>10->Top10): {rescue_count} ({rescue_pct:.1f}%)")
print(f"首位锁定 (Top10->Rank1): {precision_count} ({precision_pct:.1f}%)")
print("-" * 30)

# 设置绘图风格 (使用英文标签以确保在不同环境下显示正常)
plt.style.use('seaborn-v0_8-whitegrid')
# 定义学术风格的颜色
color_bm25 = '#A9A9A9'  # 灰色代表基线
color_bert = '#D62728'  # 红色代表BERT
color_improved = '#2CA02C' # 绿色代表优化
color_degraded = '#D62728' # 红色代表退化

# ==========================================
# 图表 1: 排名跃迁散点图 (Rank Migration Plot)
# ==========================================
plt.figure(figsize=(10, 10))

# 1. 绘制对角线 (y=x)
plt.plot([0, 55], [0, 55], color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='No Change (y=x)')

# 2. 区分优化组和退化组
improved = valid_df[valid_df['New_Rank'] < valid_df['Old_Rank']]
degraded = valid_df[valid_df['New_Rank'] > valid_df['Old_Rank']]
unchanged = valid_df[valid_df['New_Rank'] == valid_df['Old_Rank']]

# 3. 绘制散点 (添加少量随机抖动 Jitter 以防点重叠)
jitter_strength = 0.3
def add_jitter(arr):
    return arr + np.random.uniform(-jitter_strength, jitter_strength, len(arr))

plt.scatter(add_jitter(improved['Old_Rank']), add_jitter(improved['New_Rank']), 
            color=color_improved, alpha=0.6, s=40, label=f'Improved ({len(improved)})')
plt.scatter(add_jitter(degraded['Old_Rank']), add_jitter(degraded['New_Rank']), 
            color=color_degraded, alpha=0.6, s=40, marker='x', label=f'Degraded ({len(degraded)})')
plt.scatter(add_jitter(unchanged['Old_Rank']), add_jitter(unchanged['New_Rank']), 
            color='gray', alpha=0.3, s=20)

# 4. 标注区域
# 区域 A: 深层挽救 (Deep Rescue): Old > 10 & New <= 10
rect = plt.Rectangle((10.5, 0.5), 40, 10, linewidth=2, edgecolor='#FF7F0E', facecolor='none', linestyle='--')
plt.gca().add_patch(rect)
plt.text(48, 5, 'Deep Rescue\nRegion', fontsize=12, color='#FF7F0E', ha='right', fontweight='bold')

# 区域 B: 沉降区 (Top-1): New == 1
plt.axhline(y=1.5, color='blue', linestyle=':', alpha=0.5)
plt.text(52, 1, 'Top-1 Target', fontsize=10, color='blue', va='center')

# 5. 设置坐标轴
plt.xlim(0, 52)
plt.ylim(0, 52)
plt.gca().invert_yaxis() 
plt.gca().invert_xaxis() 
plt.gca().set_ylim(50, 0) # 0在上面，50在下面
plt.gca().set_xlim(0, 50) # 0在左边

plt.xlabel('BM25 Rank (Stage 1)', fontsize=12, fontweight='bold')
plt.ylabel('BERT Rank (Stage 2)', fontsize=12, fontweight='bold')
plt.title('Rank Migration Analysis', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)

# 保存
plt.savefig('rank_migration_scatter.png', dpi=300, bbox_inches='tight')
print("已生成: rank_migration_scatter.png")

# ==========================================
# 图表 2: 重力沉降效应分布图 (Sedimentation Histogram)
# ==========================================
plt.figure(figsize=(12, 6))

# 定义分桶
bins = [1, 2, 4, 11, 51] # 对应区间: 1, 2-3, 4-10, 11-50
bin_labels = ['Top-1', 'Top 2-3', 'Top 4-10', '> Top 10']

# 统计每个区间的数量
def get_counts(ranks):
    c1 = len(ranks[ranks == 1])
    c2_3 = len(ranks[(ranks >= 2) & (ranks <= 3)])
    c4_10 = len(ranks[(ranks >= 4) & (ranks <= 10)])
    c_tail = len(ranks[ranks > 10])
    return [c1, c2_3, c4_10, c_tail]

bm25_counts = get_counts(valid_df['Old_Rank'])
bert_counts = get_counts(valid_df['New_Rank'])

x = np.arange(len(bin_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, bm25_counts, width, label='BM25 (Stage 1)', color='#A9A9A9', alpha=0.8)
rects2 = ax.bar(x + width/2, bert_counts, width, label='BERT (Stage 2)', color='#1F77B4', alpha=0.9)

# 添加数值标签
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

# 设置标签和标题
ax.set_ylabel('Number of Queries', fontsize=12)
ax.set_title('Rank Distribution Shift', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(bin_labels, fontsize=11)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('rank_distribution_sedimentation.png', dpi=300)
print("已生成: rank_distribution_sedimentation.png")