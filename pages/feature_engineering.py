# -*- coding: utf-8 -*-
"""
特征工程页面
核心功能：标准化处理、特征降维、特征重要性排名、特征选择
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PathConfig


def render_feature_engineering():
    """渲染特征工程页面"""
    st.title("特征工程")
    
    st.markdown("""
    <div style="background: #e8f8f5; border-left: 4px solid #2ecc71; padding: 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;">
        <strong>特征工程 (Feature Engineering)</strong><br/>
        对原始特征进行处理和转换，包括标准化、降维、特征选择等，提升模型性能。
    </div>
    """, unsafe_allow_html=True)
    
    # 加载数据
    data = get_data()
    if data is None:
        st.warning("请先在「数据探索」页面加载数据")
        return
    
    # 标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "标准化处理", "特征降维", "特征重要性", "特征选择", "相关性分析"
    ])
    
    with tab1:
        render_standardization(data)
    
    with tab2:
        render_dimensionality_reduction(data)
    
    with tab3:
        render_feature_importance(data)
    
    with tab4:
        render_feature_selection(data)
    
    with tab5:
        render_correlation_analysis(data)


def get_data():
    """获取数据"""
    if 'data' in st.session_state:
        return st.session_state['data']
    
    try:
        data = pd.read_csv(PathConfig.DEFAULT_DATA_FILE)
        st.session_state['data'] = data
        return data
    except:
        return None


def prepare_data(data):
    """准备数据，编码分类变量"""
    df = data.copy()
    
    # 编码目标变量
    if 'Result' in df.columns:
        le = LabelEncoder()
        df['target'] = le.fit_transform(df['Result'])
        df = df.drop('Result', axis=1)
    
    # 编码其他分类变量
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df


def render_standardization(data):
    """标准化处理"""
    st.subheader("标准化处理")
    
    st.markdown("""
    标准化是将特征转换为统一尺度的过程，常用方法包括：
    - **Z-score 标准化**: 均值为0，标准差为1
    - **Min-Max 归一化**: 将值缩放到[0,1]范围
    """)
    
    df = data.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("没有数值特征")
        return
    
    # 竖向布局
    
    # 1. 选择特征预览
    st.markdown("##### 标准化前后对比")
    selected_col = st.selectbox("选择特征预览", numeric_cols, key="std_preview")
    
    # 2. 标准化对比图
    original = df[selected_col].dropna()
    scaler = StandardScaler()
    standardized = scaler.fit_transform(original.values.reshape(-1, 1)).flatten()
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=original, name='原始分布', opacity=0.7, marker_color='#3498db'))
    fig.add_trace(go.Histogram(x=standardized, name='标准化后', opacity=0.7, marker_color='#e74c3c'))
    fig.update_layout(barmode='overlay', title=f'{selected_col} 标准化对比', height=350)
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. 统计信息
    st.markdown("##### 统计信息")
    stats_df = pd.DataFrame({
        '指标': ['均值', '标准差', '最小值', '最大值'],
        '原始': [original.mean(), original.std(), original.min(), original.max()],
        '标准化后': [standardized.mean(), standardized.std(), standardized.min(), standardized.max()]
    }).round(4)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # 4. 应用标准化
    st.markdown("##### 应用标准化")
    
    transform_method = st.selectbox("转换方法", [
        "Z-score 标准化",
        "Min-Max 归一化",
        "对数转换 (log1p)"
    ])
    
    cols_to_transform = st.multiselect(
        "选择要转换的特征", 
        numeric_cols, 
        default=numeric_cols[:min(3, len(numeric_cols))]
    )
    
    # 5. 方法说明
    st.markdown("##### 方法说明")
    if transform_method == "Z-score 标准化":
        st.info("公式: z = (x - μ) / σ\n适用于: 大多数机器学习算法，尤其是SVM、KNN")
    elif transform_method == "Min-Max 归一化":
        st.info("公式: x' = (x - min) / (max - min)\n适用于: 神经网络、图像处理")
    else:
        st.info("公式: x' = log(1 + x)\n适用于: 偏斜分布数据、长尾分布")
    
    # 6. 应用按钮
    if cols_to_transform and st.button("应用转换", type="primary", key="apply_std"):
        transformed_data = data.copy()
        
        for col in cols_to_transform:
            if transform_method == "Z-score 标准化":
                scaler = StandardScaler()
                transformed_data[col] = scaler.fit_transform(transformed_data[[col]])
            elif transform_method == "Min-Max 归一化":
                min_val = transformed_data[col].min()
                max_val = transformed_data[col].max()
                if max_val > min_val:
                    transformed_data[col] = (transformed_data[col] - min_val) / (max_val - min_val)
            else:
                transformed_data[col] = np.log1p(transformed_data[col].clip(lower=0))
        
        st.session_state['data'] = transformed_data
        st.success(f"已对 {len(cols_to_transform)} 个特征应用 {transform_method}")
        st.rerun()


def render_dimensionality_reduction(data):
    """特征降维 (PCA)"""
    st.subheader("特征降维 (PCA)")
    
    st.markdown("""
    **主成分分析 (PCA)** 是一种无监督降维方法，通过线性变换将高维数据投影到低维空间，
    同时尽可能保留原始数据的方差信息。
    """)
    
    df = prepare_data(data)
    feature_cols = [c for c in df.columns if c != 'target']
    X = df[feature_cols].fillna(df[feature_cols].median())
    
    if len(feature_cols) < 2:
        st.warning("特征数量不足，无法进行降维")
        return
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("##### PCA 参数设置")
        n_components = st.slider("主成分数量", 2, min(len(feature_cols), 10), min(3, len(feature_cols)))
        
        # 执行PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # 方差解释比
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        st.markdown("##### 方差解释率")
        var_df = pd.DataFrame({
            '主成分': [f'PC{i+1}' for i in range(n_components)],
            '方差解释率': explained_var,
            '累计方差': cumulative_var
        }).round(4)
        st.dataframe(var_df, use_container_width=True, hide_index=True)
        
        st.metric("总方差解释率", f"{cumulative_var[-1]*100:.1f}%")
    
    with col2:
        st.markdown("##### 方差解释可视化")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f'PC{i+1}' for i in range(n_components)],
            y=explained_var,
            name='单独方差',
            marker_color='#3498db'
        ))
        fig.add_trace(go.Scatter(
            x=[f'PC{i+1}' for i in range(n_components)],
            y=cumulative_var,
            name='累计方差',
            mode='lines+markers',
            marker_color='#e74c3c'
        ))
        fig.update_layout(
            title='主成分方差解释',
            yaxis_title='方差解释率',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # PCA可视化
    st.markdown("---")
    st.markdown("##### PCA 降维可视化")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 2D散点图
        pca_df = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
        if 'target' in df.columns:
            pca_df['Result'] = df['target'].map({0: '阴性', 1: '阳性'})
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='Result',
                           color_discrete_map={'阳性': '#e74c3c', '阴性': '#3498db'},
                           title='PCA 2D投影')
        else:
            fig = px.scatter(pca_df, x='PC1', y='PC2', title='PCA 2D投影')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if n_components >= 3:
            # 3D散点图
            pca_df_3d = pd.DataFrame(X_pca[:, :3], columns=['PC1', 'PC2', 'PC3'])
            if 'target' in df.columns:
                pca_df_3d['Result'] = df['target'].map({0: '阴性', 1: '阳性'})
                fig = px.scatter_3d(pca_df_3d, x='PC1', y='PC2', z='PC3', color='Result',
                                   color_discrete_map={'阳性': '#e74c3c', '阴性': '#3498db'},
                                   title='PCA 3D投影')
            else:
                fig = px.scatter_3d(pca_df_3d, x='PC1', y='PC2', z='PC3', title='PCA 3D投影')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("需要3个以上主成分才能显示3D图")
    
    # 主成分载荷
    st.markdown("---")
    st.markdown("##### 主成分载荷 (特征贡献)")
    
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=feature_cols
    ).round(4)
    st.dataframe(loadings, use_container_width=True)


def render_correlation_analysis(data):
    """相关性分析"""
    st.subheader("相关性分析")
    
    # 准备数据
    df = prepare_data(data)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("数值特征不足")
        return
    
    # 计算相关矩阵
    corr_method = st.selectbox("相关系数方法", ["pearson", "spearman", "kendall"])
    corr_matrix = df[numeric_cols].corr(method=corr_method)
    
    # 竖向布局
    
    # 1. 热力图
    st.markdown("##### 相关系数矩阵")
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="相关系数"),
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        aspect='auto'
    )
    fig.update_layout(
        title=f'{corr_method.capitalize()}',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. 与目标变量的相关性
    st.markdown("##### 与目标变量的相关性")
    
    if 'target' in corr_matrix.columns:
        target_corr = corr_matrix['target'].drop('target').sort_values(key=abs, ascending=False)
        
        corr_df = pd.DataFrame({
            '特征': target_corr.index,
            '相关系数': target_corr.values.round(4)
        })
        corr_df['强度'] = corr_df['相关系数'].apply(
            lambda x: '强' if abs(x) > 0.5 else ('中' if abs(x) > 0.3 else '弱')
        )
        st.dataframe(corr_df, use_container_width=True, hide_index=True)
        
        # 相关性条形图
        fig = px.bar(corr_df, x='相关系数', y='特征', orientation='h',
                    color='相关系数', color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("需要目标变量来计算相关性")
    
    # 3. 高相关特征对
    st.markdown("##### 高相关特征对")
    threshold = st.slider("相关系数阈值", 0.5, 0.95, 0.7)
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append({
                    '特征1': corr_matrix.columns[i],
                    '特征2': corr_matrix.columns[j],
                    '相关系数': round(corr_matrix.iloc[i, j], 4)
                })
    
    if high_corr_pairs:
        st.dataframe(pd.DataFrame(high_corr_pairs), hide_index=True)
    else:
        st.info(f"无高于 {threshold} 的相关特征对")


def render_feature_importance(data):
    """特征重要性分析"""
    st.subheader("特征重要性分析")
    
    df = prepare_data(data)
    
    if 'target' not in df.columns:
        st.warning("需要目标变量进行特征重要性分析")
        return
    
    # 准备特征和目标
    feature_cols = [c for c in df.columns if c != 'target']
    X = df[feature_cols]
    y = df['target']
    
    # 处理缺失值
    X = X.fillna(X.median())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 基于随机森林的特征重要性")
        
        with st.spinner("计算中..."):
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            importance_df = pd.DataFrame({
                '特征': feature_cols,
                '重要性': rf.feature_importances_
            }).sort_values('重要性', ascending=True)
            
            fig = px.bar(importance_df, x='重要性', y='特征', orientation='h',
                        color='重要性', color_continuous_scale='Blues')
            fig.update_layout(height=450, title='随机森林特征重要性')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### 互信息量")
        
        with st.spinner("计算中..."):
            mi_scores = mutual_info_classif(X, y, random_state=42)
            
            mi_df = pd.DataFrame({
                '特征': feature_cols,
                '互信息量': mi_scores
            }).sort_values('互信息量', ascending=True)
            
            fig = px.bar(mi_df, x='互信息量', y='特征', orientation='h',
                        color='互信息量', color_continuous_scale='Greens')
            fig.update_layout(height=450, title='互信息量')
            st.plotly_chart(fig, use_container_width=True)
    
    # 综合排名
    st.markdown("---")
    st.markdown("##### 特征重要性综合排名")
    
    combined_df = importance_df.merge(mi_df, on='特征')
    combined_df['RF排名'] = combined_df['重要性'].rank(ascending=False).astype(int)
    combined_df['MI排名'] = combined_df['互信息量'].rank(ascending=False).astype(int)
    combined_df['综合排名'] = (combined_df['RF排名'] + combined_df['MI排名']) / 2
    combined_df = combined_df.sort_values('综合排名')
    
    st.dataframe(
        combined_df[['特征', '重要性', '互信息量', 'RF排名', 'MI排名', '综合排名']].round(4),
        use_container_width=True, hide_index=True
    )


def render_feature_selection(data):
    """特征选择"""
    st.subheader("特征选择")
    
    df = prepare_data(data)
    
    if 'target' not in df.columns:
        st.warning("需要目标变量进行特征选择")
        return
    
    feature_cols = [c for c in df.columns if c != 'target']
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['target']
    
    # 竖向布局
    
    # 1. 选择特征数量
    st.markdown("##### 选择特征数量")
    k_features = st.slider("保留的特征数量", 1, len(feature_cols), min(5, len(feature_cols)))
    
    # 2. 选择方法
    st.markdown("##### 选择方法")
    method = st.selectbox("特征选择方法", [
        "随机森林", "互信息", "F-score (ANOVA)", "卡方检验"
    ])
    
    # 3. 选择结果
    st.markdown("##### 选择结果")
    
    with st.spinner("计算中..."):
        if method == "随机森林":
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            scores = rf.feature_importances_
        elif method == "互信息":
            scores = mutual_info_classif(X, y, random_state=42)
        elif method == "F-score (ANOVA)":
            selector = SelectKBest(score_func=f_classif, k=k_features)
            selector.fit(X, y)
            scores = selector.scores_
        else:  # 卡方检验
            X_pos = X - X.min() + 1  # 确保非负
            selector = SelectKBest(score_func=chi2, k=k_features)
            selector.fit(X_pos, y)
            scores = selector.scores_
        
        score_df = pd.DataFrame({
            '特征': feature_cols,
            '得分': scores
        }).sort_values('得分', ascending=False)
        
        selected_features = score_df.head(k_features)['特征'].tolist()
        
        st.dataframe(score_df.head(k_features).round(4), hide_index=True, use_container_width=True)
    
    # 4. 可视化对比
    st.markdown("##### 选中特征与未选中特征对比")
    
    score_df['选中'] = score_df['特征'].isin(selected_features)
    fig = px.bar(score_df, x='得分', y='特征', orientation='h',
                color='选中', color_discrete_map={True: '#27ae60', False: '#bdc3c7'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # 保存选择
    if st.button("应用特征选择", type="primary"):
        original_data = data.copy()
        cols_to_keep = selected_features.copy()
        if 'Result' in original_data.columns:
            cols_to_keep.append('Result')
        
        filtered_data = original_data[cols_to_keep]
        st.session_state['data'] = filtered_data
        st.session_state['selected_features'] = selected_features
        st.success(f"已选择 {len(selected_features)} 个特征: {', '.join(selected_features)}")
