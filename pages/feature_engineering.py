import plotly.graph_objects as go
import streamlit as st

from src.heart_pipeline import (
    BASE_NUMERIC_FEATURES,
    DERIVED_NUMERIC_FEATURES,
    MODE_LABELS,
    MODE_HINTS,
    SCREENING_FEATURES,
    compute_feature_scores,
    prepare_xy,
)


def _card(html_content: str) -> str:
    return f'<div style="background:white;border-radius:10px;padding:1.2rem;box-shadow:0 1px 3px rgba(0,0,0,0.06);margin-bottom:1rem;">{html_content}</div>'


def render_feature_engineering() -> None:
    st.markdown(
        "<div style='margin-bottom:0.5rem;'>"
        "<div style='font-size:1.5rem;font-weight:700;color:#0f172a;'>特征工程</div>"
        "<div style='font-size:0.85rem;color:#64748b;margin-top:2px;'>特征评分与选择，构建最佳特征子集</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    data = st.session_state.get("clean_data")
    if data is None:
        st.warning("请先在「数据探索」页面加载并清洗数据。")
        return

    # ================================================================
    # 模式选择
    # ================================================================
    st.markdown(
        _card(
            "<div style='margin-bottom:0.75rem;font-size:0.85rem;font-weight:600;color:#334155;'>选择分析模式</div>"
        ),
        unsafe_allow_html=True,
    )
    mode = st.radio(
        "模式",
        ["clinical", "screening"],
        format_func=lambda x: MODE_LABELS[x],
        horizontal=True,
        index=0 if st.session_state.get("feature_mode", "clinical") == "clinical" else 1,
        label_visibility="collapsed",
        key="feature_mode",
    )

    st.info(MODE_HINTS[mode])

    st.write("")  # 间距

    # ================================================================
    # 计算特征评分
    # ================================================================
    feature_scores = compute_feature_scores(data)
    if mode == "screening":
        screening_set = set(SCREENING_FEATURES)
        feature_scores = feature_scores[feature_scores["feature"].isin(screening_set)]
    st.session_state["feature_scores"] = feature_scores

    # ================================================================
    # 特征选择区
    # ================================================================
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown(
            _card(
                "<div style='margin-bottom:0.75rem;font-size:0.85rem;font-weight:600;color:#334155;'>特征数量</div>"
            ),
            unsafe_allow_html=True,
        )
        top_k = st.slider(
            "特征数",
            min_value=3,
            max_value=len(feature_scores),
            value=min(6, len(feature_scores)),
            label_visibility="collapsed",
        )
        selected = feature_scores.head(top_k)["feature"].tolist()
        st.session_state["selected_features"] = selected

        # 选中标签
        st.markdown(
            f"<div style='display:flex;flex-wrap:wrap;gap:5px;margin-top:0.75rem;'>"
            + "".join(
                f"<span style='background:#eff6ff;color:#1d4ed8;font-size:0.75rem;padding:2px 10px;border-radius:12px;border:1px solid #bfdbfe;'>{c}</span>"
                for c in selected
            )
            + f"<span style='color:#94a3b8;font-size:0.7rem;padding:2px 6px;'>共{len(selected)}个</span>"
            + "</div>",
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            _card(
                "<div style='margin-bottom:0.75rem;font-size:0.85rem;font-weight:600;color:#334155;'>综合评分排名</div>"
            ),
            unsafe_allow_html=True,
        )
        chart_data = feature_scores.head(10).copy()
        bar_colors = ["#3b82f6" if f in selected else "#e2e8f0" for f in chart_data["feature"]]

        fig = go.Figure(go.Bar(
            x=chart_data["combined_score"],
            y=chart_data["feature"],
            orientation="h",
            marker_color=bar_colors,
            marker_line_width=0,
            text=[f"{s:.3f}" for s in chart_data["combined_score"]],
            textposition="outside",
            textfont=dict(size=10, color="#64748b"),
        ))
        fig.update_layout(
            yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="#334155")),
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False, visible=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=5, r=40, t=5, b=5),
            height=260,
            font=dict(family="sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "<div style='display:flex;gap:1rem;font-size:0.7rem;color:#94a3b8;margin-top:-0.5rem;'>"
            "<span><span style='display:inline-block;width:10px;height:10px;background:#3b82f6;border-radius:2px;margin-right:4px;'></span>已选中</span>"
            "<span><span style='display:inline-block;width:10px;height:10px;background:#e2e8f0;border-radius:2px;margin-right:4px;'></span>其他</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ================================================================
    # 特征评分表格
    # ================================================================
    st.markdown(
        _card(
            "<div style='margin-bottom:0.75rem;font-size:0.85rem;font-weight:600;color:#334155;'>特征评分详情</div>"
        ),
        unsafe_allow_html=True,
    )
    display_df = feature_scores.copy()
    display_df.insert(0, "排名", range(1, len(display_df) + 1))
    st.dataframe(
        display_df.round(4),
        use_container_width=True,
        hide_index=True,
        column_config={
            "排名": st.column_config.NumberColumn(format="%d"),
            "feature": "特征名称",
            "mutual_info": st.column_config.NumberColumn("互信息", format="%.4f"),
            "rf_importance": st.column_config.NumberColumn("随机森林", format="%.4f"),
            "combined_score": st.column_config.NumberColumn("综合得分", format="%.4f"),
        },
    )

    # ================================================================
    # 数据指标卡片
    # ================================================================
    st.write("")  # 间距
    X, y = prepare_xy(data)

    metric_cards = [
        ("总样本数", X.shape[0], "#2563eb"),
        ("候选特征", len(feature_scores), "#7c3aed"),
        ("选中特征", len(selected), "#0891b2"),
        ("阳性样本", int(y.sum()), "#d97706"),
    ]

    cols = st.columns(4)
    for i, (label, value, color) in enumerate(metric_cards):
        with cols[i]:
            st.markdown(
                f"<div style='text-align:center;padding:0.8rem;background:#f8fafc;border-radius:8px;border:1px solid #f1f5f9;'>"
                f"<div style='font-size:0.7rem;color:#94a3b8;margin-bottom:4px;'>{label}</div>"
                f"<div style='font-size:1.4rem;font-weight:700;color:{color};'>{value}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ================================================================
    # 字段总览
    # ================================================================
    st.markdown(
        _card(
            "<div style='margin-bottom:0.75rem;font-size:0.85rem;font-weight:600;color:#334155;'>字段总览</div>"
        ),
        unsafe_allow_html=True,
    )

    t_cols = st.columns(3)
    field_groups = [
        ("原始数值特征", BASE_NUMERIC_FEATURES, "bg-gray-50", "slate-600"),
        ("派生年龄特征", DERIVED_NUMERIC_FEATURES, "bg-indigo-50", "indigo-600"),
        ("当前选中特征", selected, "bg-blue-50", "blue-600"),
    ]

    for i, (title, items, _, _) in enumerate(field_groups):
        with t_cols[i]:
            st.markdown(
                f"<div style='font-size:0.75rem;font-weight:600;color:#64748b;margin-bottom:0.5rem;'>{title}</div>"
                + "".join(
                    f"<span style='display:inline-block;background:#f8fafc;color:#475569;font-size:0.75rem;padding:2px 10px;border-radius:4px;margin:2px;border:1px solid #f1f5f9;'>{c}</span>"
                    for c in items
                )
                + f"<div style='font-size:0.7rem;color:#94a3b8;margin-top:4px;'>共 {len(items)} 个</div>",
                unsafe_allow_html=True,
            )

    # ================================================================
    # 操作说明（折叠）
    # ================================================================
    with st.expander("操作说明"):
        st.markdown("""
**第 1 步：选模式**
- 专业版：用全部指标（肌钙蛋白、CK-MB），准确率更高
- 筛查版：只用常规指标（血压、心率、血糖），适合居家自测

**第 2 步：拖动滑块选数量**
- 往右拖 = 用更多特征
- 往左拖 = 用更少特征
- 默认 6 个兼顾效果和速度

**第 3 步：看图表**
- 条形越长 = 特征越重要
- 蓝色 = 已选中，灰色 = 未选中
        """)
