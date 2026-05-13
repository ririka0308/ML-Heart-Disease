from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.config import PathConfig
from src.heart_pipeline import (
    ALL_NUMERIC_FEATURES,
    auto_map_columns,
    build_correlation_frame,
    clean_dataset,
    load_dataset,
    save_processed_dataset,
)

# ── 统一 Plotly 主题 ───────────────────────────────────────────────────────────
_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Inter, PingFang SC, sans-serif", size=12, color="#4a5568"),
    title_font=dict(size=14, color="#1a202c"),
    margin=dict(l=10, r=10, t=40, b=10),
    height=320,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
_AXIS = dict(gridcolor="#edf2f7", linecolor="#e2e8f0", tickfont=dict(size=11))
_COLORS = {"positive": "#e53e3e", "negative": "#3182ce"}


def _store_dataset(clean_df, raw_df, summary, source_type, file_name):
    st.session_state["raw_data"] = raw_df
    st.session_state["clean_data"] = clean_df
    st.session_state["data_summary"] = summary
    st.session_state["data_source_type"] = source_type
    st.session_state["data_file_name"] = file_name
    st.session_state["data_source_selected"] = True


def render_eda() -> None:
    # ── 标题 ──────────────────────────────────────────────────────────────────
    st.title("数据探索与清洗")
    st.caption("数据加载 - 质量检查 - 分布分析 - 相关性分析")
    st.divider()

    # ── 数据来源 ────────────────────────────────────────────────────────────────
    st.session_state.setdefault("data_source_selected", False)
    st.session_state.setdefault("data_source_type", None)
    st.session_state.setdefault("data_file_name", None)

    cur_type = st.session_state.get("data_source_type")
    cur_file = st.session_state.get("data_file_name")

    # 已加载状态
    if cur_type and cur_file:
        label_map = {"default": "内置数据集", "uploaded": "用户上传"}
        st.success(f"当前：{label_map.get(cur_type, cur_type)} — {cur_file}")

    # 两个选项：内置 / 上传
    col_opt, col_btn = st.columns([3, 1])
    with col_opt:
        opt = st.radio(
            "选择数据来源",
            ["使用内置数据集", "上传 CSV 文件"],
            horizontal=True,
            label_visibility="collapsed",
            key="data_source_option",
        )

    uploaded = None
    if opt == "上传 CSV 文件":
        uploaded = st.file_uploader(
            "选择 CSV 文件",
            type=["csv"],
            label_visibility="collapsed",
        )

    with col_btn:
        st.markdown("<div style='height:2px;'></div>", unsafe_allow_html=True)
        load_clicked = st.button("加载数据", use_container_width=True, type="primary")

    # ── 加载逻辑 ───────────────────────────────────────────────────────────────
    should_load = False
    source_type = None
    fname = None

    if load_clicked:
        if opt == "使用内置数据集":
            should_load = True
            source_type = "default"
            fname = Path(PathConfig.DEFAULT_DATA_FILE).name
        elif uploaded is not None:
            should_load = True
            source_type = "uploaded"
            fname = uploaded.name
        else:
            st.warning("请先选择或上传数据文件。")

    if should_load:
        with st.spinner("加载并清洗数据中..."):
            file_path = PathConfig.DEFAULT_DATA_FILE if source_type == "default" else uploaded
            raw_df = load_dataset(file_path)
            col_map = auto_map_columns(raw_df)
            st.session_state["column_mapping"] = col_map
            clean_df, summary = clean_dataset(raw_df)
            _store_dataset(clean_df, raw_df, summary, source_type, fname)
        st.success(f"数据加载完成！共 {summary.rows_after} 条样本，{summary.positive_rate*100:.1f}% 阳性率")

    # ── 目标列选择（数据集没有 Result 列时让用户手动指定）────────────────────────
    clean_df = st.session_state.get("clean_data")
    if clean_df is not None and "Result" not in clean_df.columns:
        raw_df = st.session_state.get("raw_data")
        if raw_df is not None:
            raw_cols = [c for c in raw_df.columns if str(c).strip() not in ("id", "ID", "Id")]
            target_col = st.selectbox(
                "选择诊断结果列",
                [""] + sorted(raw_cols),
                help="数据中没有自动识别出诊断结果列，请手动选择哪一列代表阳性/阴性。",
            )
            if target_col:
                raw_df2 = raw_df.copy()
                raw_df2.rename(columns={target_col: "Result"}, inplace=True)
                col_map2 = auto_map_columns(raw_df2)
                st.session_state["column_mapping"] = col_map2
                clean_df2, summary2 = clean_dataset(raw_df2)
                _store_dataset(
                    clean_df2, raw_df2, summary2,
                    source_type or st.session_state.get("data_source_type", "uploaded"),
                    fname or st.session_state.get("data_file_name", target_col),
                )
                st.success(f"已使用 '{target_col}' 作为诊断结果列。")
                st.rerun()
            else:
                st.warning("请选择诊断结果列以启用模型训练，或跳过仅查看数据分布。")

    # ── 没有数据时提前返回 ─────────────────────────────────────────────────────
    if st.session_state.get("clean_data") is None:
        st.info("请选择数据来源开始分析")
        return

    clean_df = st.session_state["clean_data"]
    raw_df   = st.session_state["raw_data"]
    summary  = st.session_state["data_summary"]
    file_name = st.session_state["data_file_name"]

    # ── 数据概况 ───────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background: white; border-radius: 12px; padding: 1.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                <span style="font-size: 1.2rem;"> </span>
                <span style="color: #1e293b; font-weight: 600; font-size: 1rem;">数据概况</span>
            </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("原始行数",   summary.rows_before)
    c2.metric("重复样本",   summary.duplicate_rows,  delta=f"-{summary.duplicate_rows}" if summary.duplicate_rows else None)
    c3.metric("异常值截断", summary.clipped_cells)
    c4.metric("清洗后阳性率", f"{summary.positive_rate*100:.1f}%")

    if st.button("导出清洗后数据", type="primary"):
        saved_path = save_processed_dataset(clean_df)
        st.success(f"已导出：{saved_path}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 列名映射提示（仅用户上传时显示）────────────────────────────────────────
    col_map = st.session_state.get("column_mapping", {})
    if col_map:
        mapped_items = [(v, k) for k, v in col_map.items()]
        if mapped_items:
            map_df = pd.DataFrame(mapped_items, columns=["原始列名", "映射为标准列名"])
            st.dataframe(map_df, use_container_width=True, hide_index=True)

    # ── 标签页 ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["数据预览", "清洗详情", "分布分析", "相关性分析"])

    # ── Tab 1：数据预览 ──────────────────────────────────────────────────────
    with tab1:
        st.caption(f"文件：{file_name}　共 {len(clean_df)} 行 × {len(clean_df.columns)} 列")
        st.dataframe(clean_df, use_container_width=True)

    # ── Tab 2：清洗详情 ──────────────────────────────────────────────────────
    with tab2:
        left, right = st.columns(2)
        before_missing = (
            pd.DataFrame({
                "特征": list(summary.missing_before.keys()),
                "清洗前缺失数": list(summary.missing_before.values()),
            }).sort_values("清洗前缺失数", ascending=False)
        )
        after_missing = (
            pd.DataFrame({
                "特征": list(summary.missing_after.keys()),
                "清洗后缺失数": list(summary.missing_after.values()),
            }).sort_values("清洗后缺失数", ascending=False)
        )
        with left:
            st.subheader("清洗前缺失统计")
            st.dataframe(before_missing, hide_index=True, use_container_width=True)
        with right:
            st.subheader("清洗后缺失统计")
            st.dataframe(after_missing, hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("清洗策略说明")
        st.markdown("""
- **数值型缺失**：用中位数填补
- **类别型缺失**：用众数填补
- **重复样本**：整行删除
- **异常值**：IQR 截断（上下各 1.5 倍四分位距）
        """)

    # ── Tab 3：分布分析 ──────────────────────────────────────────────────────
    with tab3:
        has_result = "Result" in clean_df.columns
        available_features = sorted(
            [f for f in ALL_NUMERIC_FEATURES if f in clean_df.columns],
            key=lambda x: ALL_NUMERIC_FEATURES.index(x) if x in ALL_NUMERIC_FEATURES else 99
        )
        feature = st.selectbox("选择特征", available_features)

        col_hist, col_box = st.columns(2)

        with col_hist:
            hist_kwargs = dict(nbins=30, barmode="overlay", opacity=0.75)
            if has_result:
                hist_kwargs["color"] = "Result"
                hist_kwargs["color_discrete_map"] = _COLORS
            fig = px.histogram(clean_df, x=feature, **hist_kwargs,
                title=f"{feature} 频率分布", template="plotly_white")
            fig.update_layout(**_LAYOUT)
            fig.update_xaxes(**_AXIS, title=feature)
            fig.update_yaxes(**_AXIS, title="频数")
            st.plotly_chart(fig, use_container_width=True)

        with col_box:
            if has_result:
                fig2 = px.box(
                    clean_df, x="Result", y=feature, color="Result",
                    color_discrete_map=_COLORS, boxmode="group",
                    title=f"{feature} — 分组对比", template="plotly_white")
                fig2.update_layout(**{**_LAYOUT, "showlegend": False})
                fig2.update_xaxes(**_AXIS, title="诊断结果")
            else:
                fig2 = px.box(
                    clean_df, y=feature,
                    title=f"{feature} 分布", template="plotly_white")
                fig2.update_layout(**{**_LAYOUT, "showlegend": False})
            fig2.update_yaxes(**_AXIS, title=feature)
            st.plotly_chart(fig2, use_container_width=True)

        # 描述统计对比
        if has_result:
            st.subheader("描述统计对比")
            desc = clean_df.groupby("Result")[feature].describe().T
            st.dataframe(desc.style.format("{:.3f}"), use_container_width=True)

        # ── 各特征分组对比总览 ──────────────────────────────────────────────────
        if has_result:
            st.subheader("各特征分组对比总览")
            st.caption("所有数值特征在阳性/阴性分组下的分布对比，一目了然哪些特征对诊断最有区分力。")

            numeric_for_box = [f for f in ALL_NUMERIC_FEATURES if f in clean_df.columns]
            if len(numeric_for_box) > 1:
                melt_df = clean_df[numeric_for_box + ["Result"]].melt(
                    id_vars=["Result"], var_name="特征", value_name="数值"
                )
                n_features = len(numeric_for_box)
                n_cols = 4
                n_rows = (n_features + n_cols - 1) // n_cols

                fig_box_all = px.box(
                    melt_df, x="Result", y="数值", color="Result",
                    facet_col="特征", facet_col_wrap=n_cols,
                    color_discrete_map=_COLORS,
                    template="plotly_white",
                    height=280 * n_rows,
                )
                fig_box_all.update_layout(
                    showlegend=False,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(size=11, color="#4a5568"),
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                fig_box_all.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                st.plotly_chart(fig_box_all, use_container_width=True)

                # 各特征均值差异表
                st.caption("各特征在两组间的均值差异（绝对值越大表示区分能力越强）")
                group_means = clean_df.groupby("Result")[numeric_for_box].mean().T
                group_means["差值"] = (group_means.iloc[:, 1] - group_means.iloc[:, 0]).abs()
                group_means = group_means.sort_values("差值", ascending=False)
                st.dataframe(group_means.style.format("{:.3f}"), use_container_width=True)

    # ── Tab 4：相关性分析 ─────────────────────────────────────────────────────
    with tab4:
        corr = build_correlation_frame(clean_df)

        if corr.empty or corr.shape[0] < 2:
            st.info("需要至少两个数值特征才能计算相关性矩阵。")
        else:
            fig3 = go.Figure(data=go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale="RdBu", zmin=-1, zmax=1,
                texttemplate="%{z:.2f}", textfont=dict(size=11),
                hoverongaps=False, colorbar=dict(title="r", thickness=14),
            ))
            fig3.update_layout(
                title="特征相关性矩阵", height=400,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="white", plot_bgcolor="white",
                font=dict(size=11, color="#4a5568"),
            )
            st.plotly_chart(fig3, use_container_width=True)

            st.subheader("与诊断结果的相关性（排序）")
            if "ResultBinary" in corr.columns:
                strong = (
                    corr["ResultBinary"].drop("ResultBinary")
                    .sort_values(key=lambda s: s.abs(), ascending=False)
                    .reset_index()
                    .rename(columns={"index": "特征", "ResultBinary": "相关系数 r"})
                )
                strong["强度"] = strong["相关系数 r"].apply(
                    lambda x: "强" if abs(x) > 0.5 else "中" if abs(x) > 0.3 else "弱"
                )
                st.dataframe(
                    strong,
                    hide_index=True,
                    use_container_width=True,
                    column_config={"相关系数 r": st.column_config.NumberColumn(format="%.3f")},
                )
                st.caption("绝对值 > 0.5 为强相关，0.3-0.5 为中等相关，< 0.3 为弱相关。")
            else:
                st.info("需要选择诊断结果列后才能查看相关性排序。")
