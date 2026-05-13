from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import AppConfig, PathConfig
from pages.eda import render_eda
from pages.feature_engineering import render_feature_engineering
from pages.model_lab import render_model_lab
from pages.overview import render_overview
from pages.prediction import render_prediction


st.set_page_config(
    page_title=AppConfig.PAGE_TITLE,
    page_icon=AppConfig.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        /* ── 全局字体与背景 ── */
        html, body, [class*="css"] {
            font-family: "Inter", "PingFang SC", "Microsoft YaHei", sans-serif;
        }
        .stApp {
            background: #f8f9fa;
        }

        /* ── 隐藏 Streamlit 自带导航与工具栏 ── */
        [data-testid="stSidebarNav"],
        [data-testid="stToolbarActions"],
        [data-testid="stAppDeployButton"],
        [data-testid="stMainMenu"],
        [data-testid="stStatusWidget"],
        footer,
        #stDecoration {
            display: none !important;
        }

        /* ── 侧边栏：深色、干净 ── */
        [data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid #1e293b;
        }
        [data-testid="stSidebar"] * { color: #94a3b8; }

        /* ── 侧边栏导航 radio 样式（步骤编号） ── */
        [data-testid="stSidebar"] [role="radiogroup"] {
            display: flex;
            flex-direction: column;
            gap: 2px;
            padding: 4px 8px;
        }
        [data-testid="stSidebar"] .stRadio label {
            display: flex;
            align-items: center;
            color: #64748b;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
            border: none;
            background: transparent;
            margin: 0;
            letter-spacing: 0.3px;
        }
        [data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(255,255,255,0.04);
            color: #cbd5e1;
        }
        [data-testid="stSidebar"] .stRadio label[data-checked="true"] {
            background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(139,92,246,0.08));
            color: #93c5fd !important;
            font-weight: 600;
        }

        /* ── 主内容区 ── */
        .block-container {
            padding: 2rem 2.5rem 2rem 2.5rem;
            max-width: 1200px;
        }

        /* ── 按钮：扁平蓝 ── */
        .stButton > button {
            background: #3b82f6;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 20px;
            font-weight: 500;
            font-size: 0.9rem;
            transition: background 0.15s ease;
            box-shadow: none;
        }
        .stButton > button:hover {
            background: #2563eb;
            box-shadow: none;
            transform: none;
        }
        .stButton > button:active {
            background: #1d4ed8;
        }

        /* ── 标签页：下划线风格 ── */
        [data-testid="stTabs"] [role="tablist"] {
            border-bottom: 1px solid #e2e8f0;
            gap: 0;
        }
        [data-testid="stTabs"] [role="tab"] {
            background: transparent;
            border: none;
            border-bottom: 2px solid transparent;
            border-radius: 0;
            padding: 8px 16px;
            color: #718096;
            font-weight: 500;
            font-size: 0.9rem;
            box-shadow: none;
            margin-bottom: -1px;
        }
        [data-testid="stTabs"] [role="tab"]:hover {
            color: #2d3748;
            background: transparent;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            background: transparent !important;
            color: #3b82f6 !important;
            border-bottom: 2px solid #3b82f6;
            box-shadow: none;
            transform: none;
        }
        [data-testid="stTabs"] [role="tab"] span {
            color: inherit !important;
        }

        /* ── 文件上传器：隐藏英文提示文字 ── */
        [data-testid="stFileUploader"] section small {
            display: none !important;
        }
        [data-testid="stFileUploader"] section div:first-child {
            padding: 0.5rem !important;
        }

        /* ── 全局卡片样式（替代大量 inline HTML） ── */
        .app-card {
            background: white;
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
            margin-bottom: 1rem;
        }
        .app-card-header {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 1rem;
            color: #1e293b;
            font-weight: 600;
            font-size: 0.95rem;
        }
        .app-metric {
            background: linear-gradient(135deg, #f8fafc, #f1f5f9);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            text-align: center;
        }
        .app-metric-value {
            font-size: 1.3rem;
            font-weight: 700;
            color: #1e293b;
        }
        .app-metric-label {
            font-size: 0.78rem;
            color: #64748b;
            margin-top: 2px;
        }
        .app-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        .app-badge-blue  { background: #dbeafe; color: #1e40af; }
        .app-badge-green { background: #dcfce7; color: #166534; }
        .app-badge-gray  { background: #f1f5f9; color: #475569; }
        .app-alert-warning {
            background: linear-gradient(135deg, #fef3c7, #fde68a);
            border: 1px solid #fcd34d;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: #92400e;
            font-size: 0.85rem;
        }

        /* ── 数据框 ── */
        .stDataFrame {
            border: 1px solid #e2e8f0;
            border-radius: 6px;
        }

        /* ── 输入控件统一描边 ── */
        .stSelectbox > div:first-child,
        .stNumberInput > div:first-child,
        .stTextInput > div:first-child {
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            background: white;
        }
        .stSelectbox > div:first-child:focus-within,
        .stNumberInput > div:first-child:focus-within,
        .stTextInput > div:first-child:focus-within {
            border-color: #3b82f6;
            box-shadow: 0 0 0 2px rgba(59,130,246,0.15);
        }

        /* ── 标题层级 ── */
        h1 { font-size: 1.5rem; font-weight: 700; color: #1a202c; margin-bottom: 0.25rem; }
        h2 { font-size: 1.15rem; font-weight: 600; color: #2d3748; margin-bottom: 0.5rem; }
        h3 { font-size: 1rem; font-weight: 600; color: #4a5568; }
        p, li { color: #4a5568; font-size: 0.9rem; }

        /* ── 分隔线 ── */
        hr { border: none; border-top: 1px solid #e2e8f0; margin: 1.25rem 0; }

        /* ── metric 卡片 ── */
        div[data-testid="metric-container"] {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
        }
        div[data-testid="metric-container"]:hover {
            border-color: #cbd5e0;
        }

        /* ── 告警提示 ── */
        .stSuccess, .stInfo, .stWarning, .stError {
            border-radius: 6px;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("current_page", "项目概览")
    st.session_state.setdefault("raw_data", None)
    st.session_state.setdefault("clean_data", None)
    st.session_state.setdefault("data_summary", None)
    st.session_state.setdefault("feature_scores", None)
    st.session_state.setdefault("selected_features", None)
    st.session_state.setdefault("model_comparison", None)
    st.session_state.setdefault("best_model", None)


def _on_nav_change() -> None:
    """侧边栏导航切换时立即更新当前页面"""
    st.session_state["current_page"] = st.session_state["nav_radio"]


def render_sidebar() -> str:
    with st.sidebar:
        # 顶部品牌区
        st.markdown(
            """
            <div style="padding: 1.5rem 1rem 1rem 1rem; text-align: center;">
                <div style="font-size: 1.6rem; font-weight: 700; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">心脏病分析系统</div>
                <div style="color: #64748b; font-size: 0.75rem; margin-top: 4px;">数据分析与风险预测</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page_order = ["项目概览", "数据探索", "特征工程", "模型实验", "风险预测"]

        st.radio(
            "导航",
            page_order,
            index=page_order.index(st.session_state["current_page"]),
            key="nav_radio",
            on_change=_on_nav_change,
            label_visibility="collapsed",
        )

        selected = st.session_state["nav_radio"]
        st.session_state["current_page"] = selected

        # 状态面板
        has_data = st.session_state.get("clean_data") is not None
        cmp = st.session_state.get("model_comparison")

        st.markdown(
            "<div style='padding: 0 1rem 1rem 1rem;'>",
            unsafe_allow_html=True,
        )

        # 数据状态
        if has_data:
            st.markdown(
                "<div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>"
                "<div style='width: 6px; height: 6px; border-radius: 50%; background: #22c55e;'></div>"
                "<div style='color: #86efac; font-size: 0.8rem;'>数据已加载</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>"
                "<div style='width: 6px; height: 6px; border-radius: 50%; background: #64748b;'></div>"
                "<div style='color: #64748b; font-size: 0.8rem;'>等待数据</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        # 模型状态
        if cmp is not None:
            best_name = cmp["best_name"]
            best_f1 = cmp["results"].iloc[0]["f1"]
            st.markdown(
                f"<div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>"
                f"<div style='width: 6px; height: 6px; border-radius: 50%; background: #22c55e;'></div>"
                f"<div style='color: #86efac; font-size: 0.8rem;'>模型已训练</div>"
                f"</div>"
                f"<div style='margin-left: 1.1rem; color: #64748b; font-size: 0.7rem; margin-bottom: 0.5rem;'>{best_name} | F1 {best_f1:.3f}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>"
                "<div style='width: 6px; height: 6px; border-radius: 50%; background: #64748b;'></div>"
                "<div style='color: #64748b; font-size: 0.8rem;'>等待训练</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    return selected


def main() -> None:
    PathConfig.ensure_dirs()
    inject_style()
    init_state()
    render_sidebar()
    st.session_state["current_page"] = st.session_state.get("nav_radio", "项目概览")

    route = {
        "项目概览": render_overview,
        "数据探索": render_eda,
        "特征工程": render_feature_engineering,
        "模型实验": render_model_lab,
        "风险预测": render_prediction,
    }
    route[st.session_state["current_page"]]()


if __name__ == "__main__":
    main()
