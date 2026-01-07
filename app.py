# -*- coding: utf-8 -*-
"""
基于机器学习的心脏病数据分析与预测
Streamlit 入口文件 - 数据挖掘全生命周期
"""

import os
import sys

# 设置环境变量
os.environ['ARROW_LIBHDFS_DIR'] = ''
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'

import streamlit as st
from pathlib import Path

# 添加项目目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="心脏病数据分析与预测",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 隐藏默认UI元素 ====================
st.markdown("""
<style>
    /* 隐藏左上角页面导航 */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    /* 隐藏Deploy按钮 */
    .stDeployButton {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* 主内容区域 */
    .main {
        background-color: #f8f9fa;
    }
    
    /* 侧边栏 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e8e8e8 !important;
    }
    
    /* 侧边栏标题 */
    .sidebar-title {
        color: #ffffff !important;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.8rem 1rem;
        background: rgba(52, 152, 219, 0.2);
        border-left: 3px solid #3498db;
        margin-bottom: 0.5rem;
        border-radius: 0 8px 8px 0;
    }
    
    /* 流程步骤指示器 */
    .step-indicator {
        display: flex;
        align-items: center;
        padding: 0.6rem 1rem;
        margin: 2px 0;
        border-radius: 4px;
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }
    .step-number {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: rgba(52, 152, 219, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .step-active .step-number {
        background: #3498db;
        color: white !important;
    }
    
    /* 导航Radio样式 */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 0;
    }
    [data-testid="stSidebar"] .stRadio > div > label {
        background-color: transparent !important;
        padding: 0.7rem 1rem;
        margin: 2px 0;
        border-radius: 4px;
        border-left: 3px solid transparent;
        transition: all 0.2s ease;
        font-size: 0.95rem;
    }
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background-color: rgba(52, 152, 219, 0.15) !important;
        border-left-color: rgba(52, 152, 219, 0.5);
    }
    [data-testid="stSidebar"] .stRadio > div > label[data-baseweb="radio"][aria-checked="true"] {
        background-color: rgba(52, 152, 219, 0.25) !important;
        border-left-color: #3498db;
        font-weight: 500;
    }
    
    /* 流程线 */
    .flow-line {
        border-left: 2px dashed rgba(52, 152, 219, 0.3);
        margin-left: 1.5rem;
        padding-left: 1rem;
    }
    
    /* 主标题样式 */
    h1 {
        color: #2c3e50;
        font-size: 1.8rem !important;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.8rem;
        margin-bottom: 1.5rem;
    }
    h2 {
        color: #34495e;
        font-size: 1.3rem !important;
        margin-top: 1.5rem;
        padding-bottom: 0.5rem;
    }
    
    /* 信息卡片 */
    .info-card {
        background: white;
        border-radius: 8px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
    }
    
    /* 标签页样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: #f1f3f4;
        border-radius: 8px 8px 0 0;
        padding: 0.3rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.7rem 1.2rem;
        font-weight: 500;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* 进度说明 */
    .progress-hint {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.6);
        padding: 0.3rem 1rem;
        margin-top: -0.3rem;
    }
</style>
""", unsafe_allow_html=True)


# ==================== 数据科学叙事线导航 ====================
# ==================== 导航配置 ====================
# 数据挖掘流程模块（用户操作流程）
WORKFLOW_MODULES = {
    "1. 数据探索": {
        "desc": "EDA与数据清洗",
        "page": "eda"
    },
    "2. 特征工程": {
        "desc": "特征处理与选择",
        "page": "feature"
    },
    "3. 模型实验室": {
        "desc": "算法对比与调优",
        "page": "model"
    },
    "4. 智能预测": {
        "desc": "风险预测与解释",
        "page": "predict"
    }
}


def main():
    """主函数"""
    # 初始化状态
    if 'current_module' not in st.session_state:
        st.session_state.current_module = "项目概览"
    
    # 渲染侧边栏
    render_sidebar()
    
    # 页面路由
    route_page(st.session_state.current_module)


def render_sidebar():
    """渲染侧边栏 - 分离项目概览和数据挖掘流程"""
    
    # 标题
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 1.2rem; font-weight: 600; color: #3498db;">
            心脏病数据分析与预测
        </div>
        <div style="font-size: 0.8rem; opacity: 0.7; margin-top: 0.3rem;">
            基于机器学习的数据挖掘
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # ========== 项目概览（独立区域） ==========
    st.sidebar.markdown("""
    <div style="font-size: 0.75rem; color: rgba(255,255,255,0.5); padding: 0 0.5rem; margin-bottom: 0.3rem;">
        静态展示
    </div>
    """, unsafe_allow_html=True)
    
    # 项目概览按钮
    overview_selected = st.session_state.current_module == "项目概览"
    if st.sidebar.button(
        "项目概览",
        use_container_width=True,
        type="primary" if overview_selected else "secondary",
        key="nav_overview"
    ):
        st.session_state.current_module = "项目概览"
        st.rerun()
    
    st.sidebar.caption("默认数据集的基本信息")
    
    st.sidebar.markdown("---")
    
    # ========== 数据挖掘流程（用户操作区域） ==========
    st.sidebar.markdown('<div class="sidebar-title">数据挖掘流程</div>', unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="font-size: 0.75rem; color: rgba(255,255,255,0.5); padding: 0 0.5rem 0.5rem 0.5rem;">
        用户操作区域
    </div>
    """, unsafe_allow_html=True)
    
    # 获取当前索引
    workflow_keys = list(WORKFLOW_MODULES.keys())
    current_workflow = st.session_state.current_module if st.session_state.current_module in workflow_keys else None
    current_index = workflow_keys.index(current_workflow) if current_workflow else 0
    
    # 流程导航选择
    selected = st.sidebar.radio(
        "选择模块",
        workflow_keys,
        index=current_index if current_workflow else None,
        format_func=lambda x: x,
        label_visibility="collapsed"
    )
    
    # 显示当前模块描述
    if selected in WORKFLOW_MODULES:
        st.sidebar.markdown(f"""
        <div class="progress-hint">
            当前: {WORKFLOW_MODULES[selected]['desc']}
        </div>
        """, unsafe_allow_html=True)
    
    # 更新状态
    if selected and selected != st.session_state.current_module:
        st.session_state.current_module = selected
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # 流程进度指示
    st.sidebar.markdown("""
    <div style="padding: 0.5rem 1rem; font-size: 0.8rem;">
        <div style="opacity: 0.7; margin-bottom: 0.5rem;">工作流程</div>
        <div style="display: flex; align-items: center; gap: 4px; flex-wrap: wrap;">
            <span style="background: #3498db; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem;">数据</span>
            <span style="opacity: 0.5;">→</span>
            <span style="background: #2ecc71; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem;">特征</span>
            <span style="opacity: 0.5;">→</span>
            <span style="background: #e74c3c; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem;">模型</span>
            <span style="opacity: 0.5;">→</span>
            <span style="background: #9b59b6; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem;">预测</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 模型状态反馈
    st.sidebar.markdown("---")
    if 'model_results' in st.session_state and st.session_state['model_results']:
        results = st.session_state['model_results']
        best_result = max(results, key=lambda x: x.get('准确率', 0))
        best_model = best_result.get('模型', 'RF')
        best_acc = best_result.get('准确率', 0)
        st.sidebar.markdown(f"""
        <div style="padding: 0.5rem 1rem; background: rgba(46, 204, 113, 0.2); border-radius: 6px; margin: 0.5rem;">
            <div style="font-size: 0.75rem; opacity: 0.8;">当前最优模型</div>
            <div style="font-size: 0.9rem; font-weight: 600; color: #2ecc71;">{best_model}</div>
            <div style="font-size: 0.8rem;">准确率: {best_acc*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div style="padding: 0.5rem 1rem; background: rgba(231, 76, 60, 0.2); border-radius: 6px; margin: 0.5rem;">
            <div style="font-size: 0.75rem; opacity: 0.8;">模型状态</div>
            <div style="font-size: 0.85rem;">待训练</div>
            <div style="font-size: 0.75rem; opacity: 0.7;">请前往模型实验室</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 技术栈信息
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="padding: 0.5rem 1rem; font-size: 0.75rem; opacity: 0.7;">
        <div style="margin-bottom: 0.3rem; font-weight: 500;">技术栈</div>
        <div>Python · Scikit-learn · SHAP</div>
        <div>Streamlit · Plotly · Pandas</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.caption("本科毕业设计作品")


def route_page(module: str):
    """页面路由"""
    if module == "项目概览":
        from pages.overview import render_overview
        render_overview()
    
    elif module == "1. 数据探索":
        from pages.eda import render_eda
        render_eda()
    
    elif module == "2. 特征工程":
        from pages.feature_engineering import render_feature_engineering
        render_feature_engineering()
    
    elif module == "3. 模型实验室":
        from pages.model_lab import render_model_lab
        render_model_lab()
    
    elif module == "4. 智能预测":
        from pages.prediction import render_prediction
        render_prediction()


# ==================== 程序入口 ====================
# 直接运行主函数（兼容 Hugging Face Spaces 和本地运行）
main()
