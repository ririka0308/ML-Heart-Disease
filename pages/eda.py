# -*- coding: utf-8 -*-
"""
数据探索 (EDA) 页面
核心功能：数据预览、清洗日志、特征分布、相关性热力图
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PathConfig


def render_eda():
    """渲染数据探索页面"""
    st.title("数据探索 (EDA)")
    
    st.markdown("""
    <div style="background: #e8f4fd; border-left: 4px solid #3498db; padding: 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;">
        <strong>探索性数据分析 (Exploratory Data Analysis)</strong><br/>
        对原始数据进行初步分析，了解数据特征、发现数据质量问题、探索变量间关系。
    </div>
    """, unsafe_allow_html=True)
    
    # 标签页
    tab1, tab2, tab3, tab4 = st.tabs([
        "数据预览", "数据清洗", "分布分析", "相关性分析"
    ])
    
    with tab1:
        render_data_preview_tab()
    
    with tab2:
        data = st.session_state.get('data')
        if data is not None:
            render_data_cleaning(data)
        else:
            st.info("请先在「数据预览」页面选择数据源")
    
    with tab3:
        data = st.session_state.get('data')
        if data is not None:
            render_distribution_analysis(data)
        else:
            st.info("请先在「数据预览」页面选择数据源")
    
    with tab4:
        data = st.session_state.get('data')
        if data is not None:
            render_correlation_analysis(data)
        else:
            st.info("请先在「数据预览」页面选择数据源")


def render_data_preview_tab():
    """数据预览标签页 - 包含数据源选择"""
    st.subheader("数据预览")
    
    # 1. 选择数据源
    st.markdown("##### 选择数据源")
    
    uploaded_file = st.file_uploader(
        "上传数据集 (CSV格式)",
        type="csv",
        help="支持CSV格式的心脏病数据集"
    )
    
    if st.button("使用默认数据集", type="primary"):
        try:
            data = pd.read_csv(PathConfig.DEFAULT_DATA_FILE)
            st.session_state['raw_data'] = data.copy()
            st.session_state['data'] = data
            st.success(f"已加载默认数据集: {data.shape[0]} 条记录, {data.shape[1]} 个字段")
            st.rerun()
        except Exception as e:
            st.error(f"加载数据失败: {e}")
            return
    
    # 处理上传文件
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state['raw_data'] = data.copy()
        st.session_state['data'] = data
        st.success(f"已加载上传数据: {data.shape[0]} 条记录, {data.shape[1]} 个字段")
    
    # 2. 显示数据预览
    data = st.session_state.get('data')
    
    if data is None:
        st.info("请上传数据集或点击「使用默认数据集」开始分析")
        return
    
    st.markdown("---")
    render_data_preview(data)


def render_data_preview(data: pd.DataFrame):
    """数据预览"""
    st.subheader("数据预览")
    
    # 基本统计卡片
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("样本数量", f"{data.shape[0]:,}")
    with col2:
        st.metric("特征数量", data.shape[1])
    with col3:
        missing = data.isnull().sum().sum()
        st.metric("缺失值", missing)
    with col4:
        duplicates = data.duplicated().sum()
        st.metric("重复行", duplicates)
    with col5:
        if 'Result' in data.columns:
            pos_rate = (data['Result'] == 'positive').mean() * 100
            st.metric("阳性率", f"{pos_rate:.1f}%")
    
    st.markdown("---")
    
    # 数据表格
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("##### 数据样本")
        n_rows = st.slider("显示行数", 5, 50, 10, key="preview_rows")
        st.dataframe(data.head(n_rows), use_container_width=True)
    
    with col2:
        st.markdown("##### 字段信息")
        info_df = pd.DataFrame({
            '字段': data.columns,
            '类型': [str(t).replace('object', '文本').replace('int64', '整数').replace('float64', '小数') 
                    for t in data.dtypes],
            '非空': [f"{data[c].notna().sum()}/{len(data)}" for c in data.columns]
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)
    
    # 统计描述
    st.markdown("---")
    st.markdown("##### 统计描述")
    desc = data.describe(include='all').round(2)
    st.dataframe(desc, use_container_width=True)


def render_data_cleaning(data: pd.DataFrame):
    """数据清洗"""
    st.subheader("数据清洗")
    
    # 初始化清洗日志
    if 'cleaning_log' not in st.session_state:
        st.session_state['cleaning_log'] = []
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 缺失值分析
        st.markdown("##### 缺失值检测")
        missing = data.isnull().sum()
        missing_df = pd.DataFrame({
            '字段': missing.index,
            '缺失数': missing.values,
            '缺失率': (missing.values / len(data) * 100).round(2)
        })
        missing_df = missing_df[missing_df['缺失数'] > 0]
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
            
            fig = px.bar(missing_df, x='字段', y='缺失数', 
                        color='缺失率', color_continuous_scale='Reds',
                        title='缺失值分布')
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("数据完整，无缺失值")
    
    with col2:
        # 异常值检测
        st.markdown("##### 异常值检测")
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_col = st.selectbox("选择字段检测异常值", numeric_cols)
            
            Q1 = data[selected_col].quantile(0.25)
            Q3 = data[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = data[(data[selected_col] < lower) | (data[selected_col] > upper)]
            
            st.write(f"IQR范围: [{lower:.2f}, {upper:.2f}]")
            st.write(f"异常值: {len(outliers)} 条 ({len(outliers)/len(data)*100:.1f}%)")
            
            fig = px.box(data, y=selected_col, title=f'{selected_col} 箱线图')
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
    
    # 清洗操作
    st.markdown("---")
    st.markdown("##### 数据清洗操作")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        missing_method = st.selectbox(
            "缺失值处理",
            ["不处理", "删除缺失行", "均值填充", "中位数填充", "众数填充"]
        )
    
    with col2:
        dup_method = st.selectbox(
            "重复值处理",
            ["不处理", "删除重复行"]
        )
    
    with col3:
        outlier_method = st.selectbox(
            "异常值处理",
            ["不处理", "删除异常值", "边界截断"]
        )
    
    if st.button("执行清洗", type="primary"):
        cleaned_data = data.copy()
        log = []
        original_len = len(cleaned_data)
        
        # 缺失值处理
        if missing_method == "删除缺失行":
            cleaned_data = cleaned_data.dropna()
            log.append(f"删除缺失行: {original_len - len(cleaned_data)} 条")
        elif missing_method == "均值填充":
            for col in cleaned_data.select_dtypes(include=[np.number]).columns:
                cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mean())
            log.append("数值列均值填充完成")
        elif missing_method == "中位数填充":
            for col in cleaned_data.select_dtypes(include=[np.number]).columns:
                cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
            log.append("数值列中位数填充完成")
        elif missing_method == "众数填充":
            for col in cleaned_data.columns:
                mode_val = cleaned_data[col].mode()
                if len(mode_val) > 0:
                    cleaned_data[col] = cleaned_data[col].fillna(mode_val.iloc[0])
            log.append("众数填充完成")
        
        # 重复值处理
        if dup_method == "删除重复行":
            before = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates()
            log.append(f"删除重复行: {before - len(cleaned_data)} 条")
        
        # 更新数据
        st.session_state['data'] = cleaned_data
        st.session_state['cleaning_log'].extend(log)
        st.success(f"清洗完成！当前数据: {len(cleaned_data)} 条记录")
        
        for item in log:
            st.write(f"- {item}")
    
    # 清洗日志
    if st.session_state['cleaning_log']:
        st.markdown("---")
        st.markdown("##### 清洗日志")
        for i, log_item in enumerate(st.session_state['cleaning_log'], 1):
            st.write(f"{i}. {log_item}")


def render_distribution_analysis(data: pd.DataFrame):
    """分布分析"""
    st.subheader("特征分布分析")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("没有数值型特征")
        return
    
    # 单特征分布
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_feature = st.selectbox("选择特征", numeric_cols)
        show_by_result = st.checkbox("按诊断结果分组", value=True)
    
    with col2:
        if show_by_result and 'Result' in data.columns:
            fig = px.histogram(data, x=selected_feature, color='Result',
                              marginal='box', barmode='overlay',
                              color_discrete_map={'positive': '#e74c3c', 'negative': '#3498db'},
                              title=f'{selected_feature} 分布 (按诊断结果)')
        else:
            fig = px.histogram(data, x=selected_feature, marginal='box',
                              title=f'{selected_feature} 分布')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # 统计信息
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 统计信息")
        stats = data[selected_feature].describe()
        stats_df = pd.DataFrame({'统计量': stats.index, '值': stats.values.round(4)})
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        if 'Result' in data.columns:
            st.markdown("##### 分组统计")
            group_stats = data.groupby('Result')[selected_feature].agg(['mean', 'std', 'min', 'max']).round(3)
            group_stats.columns = ['均值', '标准差', '最小值', '最大值']
            st.dataframe(group_stats, use_container_width=True)
    
    # 所有特征分布
    st.markdown("---")
    st.markdown("##### 全部数值特征分布")
    
    n_cols = 4
    n_features = len(numeric_cols)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = make_subplots(rows=n_rows, cols=n_cols, 
                        subplot_titles=numeric_cols[:n_rows*n_cols])
    
    for i, col in enumerate(numeric_cols[:n_rows*n_cols]):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1
        fig.add_trace(
            go.Histogram(x=data[col], name=col, showlegend=False, 
                        marker_color='#3498db'),
            row=row, col=col_idx
        )
    
    fig.update_layout(height=200*n_rows, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_correlation_analysis(data: pd.DataFrame):
    """相关性分析"""
    st.subheader("相关性分析")
    
    # 准备数据
    df = data.copy()
    
    # 编码分类变量
    if 'Result' in df.columns:
        df['Result_num'] = df['Result'].map({'negative': 0, 'positive': 1})
    
    if 'Gender' in df.columns:
        df['Gender_num'] = df['Gender'].map({'Male': 1, 'Female': 0})
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("数值特征不足")
        return
    
    # 相关系数方法选择
    corr_method = st.selectbox("相关系数方法", ["pearson", "spearman", "kendall"])
    
    # 计算相关矩阵
    corr_matrix = df[numeric_cols].corr(method=corr_method)
    
    # 竖向排列布局
    
    # 1. 相关系数矩阵
    st.markdown("##### 相关系数矩阵")
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="相关系数"),
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        aspect='auto',
        text_auto='.2f'
    )
    fig.update_layout(
        title=f'{corr_method.capitalize()}',
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. 与目标变量相关性
    st.markdown("##### 与目标变量相关性")
    
    if 'Result_num' in corr_matrix.columns:
        target_corr = corr_matrix['Result_num'].drop(['Result_num']).sort_values(key=abs, ascending=False)
        
        corr_df = pd.DataFrame({
            '特征': target_corr.index,
            '相关系数': target_corr.values.round(4)
        })
        corr_df['强度'] = corr_df['相关系数'].apply(
            lambda x: '强' if abs(x) > 0.5 else ('中' if abs(x) > 0.3 else '弱')
        )
        
        # 表格
        st.dataframe(corr_df, use_container_width=True, hide_index=True)
        
        # 条形图
        fig = px.bar(corr_df, x='相关系数', y='特征', orientation='h',
                    color='相关系数', color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("需要目标变量进行相关性分析")
    
    # 3. 分析结论
    st.markdown("##### 分析结论")
    st.info("""
**相关性分析要点:**
- 相关系数 > 0.5: **强相关**
- 0.3 < 相关系数 < 0.5: **中等相关**
- 相关系数 < 0.3: **弱相关**

高相关特征对模型预测更重要
""")
