# -*- coding: utf-8 -*-
# app.py
# 功能：Streamlit 前端界面 (GUI)，调用 search_engine.py
# 特性：双模切换、卡片式UI、实时性能监控

import streamlit as st
import time
from search_engine import SearchEngine

#1. 页面基础配置
st.set_page_config(
    page_title="智能问答检索系统 (BM25 + BERT)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

#2. CSS 样式注入
st.markdown("""
<style>
    /* 全局字体优化 */
    .main {
        font-family: "Microsoft YaHei", sans-serif;
    }
    
    /* 结果卡片容器 */
    .result-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
        border: 1px solid #e0e0e0;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.1);
    }
    
    /* 两种模式的侧边高亮条 */
    .border-bm25 {
        border-left: 6px solid #4F8BF9; 
    }
    .border-bert {
        border-left: 6px solid #00CC96; 
        background-color: #fafffd; 
    }

    /* 排名徽章 (红色背景) */
    .rank-badge {
        background-color: #FF4B4B;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.9em;
        margin-right: 10px;
        vertical-align: middle;
    }

    /* 标题样式 */
    .doc-content {
        color: #1f2937;
        font-size: 1.1em;
        line-height: 1.6;
        margin-top: 10px;
    }

    /* 底部元数据栏 */
    .meta-info {
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px dashed #eee;
        font-size: 0.85em;
        color: #666;
        display: flex;
        justify-content: space-between;
    }
    
    /* 分数高亮 */
    .score-highlight {
        font-family: monospace;
        font-weight: bold;
        color: #0E1117;
    }
</style>
""", unsafe_allow_html=True)

#3. 核心引擎加载 (带缓存)
@st.cache_resource(show_spinner=False)
def get_engine():
    engine = SearchEngine()
    engine.load_resources()
    return engine

#4. 侧边栏：控制台
with st.sidebar:
    st.title("系统控制台")
    
    st.markdown("1. 核心模式选择")
    # 模式选择器
    search_mode = st.radio(
        "检索算法:",
        ("模式A(仅BM25)", "模式B(BM25 + BERT)"),
        index=1, # 默认选中第二个(智能模式)
        help="模式A仅基于BM25匹配；模式B包含BERT重排序。"
    )

    st.markdown("---")
    st.markdown("2. 结果参数")
    top_k = st.slider("最终展示数量 (Top-K)", 3, 20, 10)
    
    # 只有在 BERT 模式下才显示这个参数
    candidate_n = 50
    if "BERT" in search_mode:
        candidate_n = st.slider("AI重排候选数 (N)", 20, 100, 50, 
                                help="增加此数值可提升召回率，但会增加耗时")

    st.markdown("---")
    st.markdown("3. 硬件状态")
    # 占位符，稍后更新数据
    status_container = st.empty()

#5. 主界面逻辑

st.title("中文智能问答检索系统")
st.caption("基于 **BM25索引** 与 **BERT语义重排序** 的两阶段检索架构 | CS毕业设计 | 基于BM25的中文问答系统检索方法研究与实现")

# 5.1 加载引擎
with st.spinner("正在加载后端引擎 (这需要一定的时间)"):
    engine = get_engine()

# 5.2 实时更新侧边栏状态
status = engine.get_system_status()
status_container.markdown(f"""
- 计算设备: `{status['device']}`
- 显存占用: `{status['gpu_mem']}`
- 索引规模: `{status['memory_docs']}`
""")

# 5.3 搜索交互区 (表单)
with st.form("search_form"):
    col1, col2 = st.columns([5, 1])
    with col1:
        query_text = st.text_input("请输入查询问题:", placeholder="例如：显示器花屏怎么办？")
    with col2:
        st.write("") 
        st.write("") 
        submitted = st.form_submit_button("开始搜索", use_container_width=True)

#6. 搜索执行与响应时间计算

if submitted and query_text:
    if not query_text.strip():
        st.warning("输入内容不能为空！")
    else:
        #响应时间计算
        t0 = time.time()  # 记录开始时间
        
        # 根据模式调用不同函数
        if "BERT" in search_mode:
            # 模式B
            results = engine.search_bert(query_text, top_k=top_k, candidate_k=candidate_n)
            style_class = "border-bert"
            score_label = "Semantic Score"
            badge_text = "AI-Rerank"
        else:
            # 模式A
            results = engine.search_bm25(query_text, top_k=top_k)
            style_class = "border-bm25"
            score_label = "BM25 Score"
            badge_text = "Keyword-Match"
            
        t1 = time.time()  # 记录结束时间
        elapsed_time = t1 - t0  # 计算耗时 (秒)

        # 显示顶部统计条
        st.success(f"""
        检索完成!
        耗时: {elapsed_time:.4f} 秒. 展示: **{len(results)}** 条相关结果
        """)

        # 遍历并渲染结果卡片
        for item in results:
            # 将所有 HTML 挤在一行，彻底杜绝空格干扰
            st.markdown(f'<div class="result-card {style_class}"><div style="display: flex; align-items: center; margin-bottom: 8px;"><span class="rank-badge">Rank {item["rank"]}</span><span style="font-weight: bold; font-size: 1.05em;">{item["content"][:40]}...</span></div><div class="doc-content">{item["content"][:250]}...</div><div class="meta-info"><span> 文档ID: <b>{item["id"]}</b> &nbsp;|&nbsp;  模式: {badge_text}</span><span class="score-highlight">{score_label}: {item["score"]}</span></div></div>', unsafe_allow_html=True)
            
# 页脚
st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>© 2026 Graduation Project | CS毕业设计 </div>", unsafe_allow_html=True)