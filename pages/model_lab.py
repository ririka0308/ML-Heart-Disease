# -*- coding: utf-8 -*-
"""
模型实验室页面
核心功能：多算法对比训练、超参数调优、模型评估、ROC曲线、混淆矩阵
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report, roc_auc_score
)
import sys
from pathlib import Path
import time
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PathConfig

# 可用模型
AVAILABLE_MODELS = {
    'Logistic Regression': LogisticRegression,
    'Random Forest': RandomForestClassifier,
    'SVM': SVC,
    'KNN': KNeighborsClassifier,
    'Decision Tree': DecisionTreeClassifier,
    'Gradient Boosting': GradientBoostingClassifier
}

# 模型默认参数
MODEL_PARAMS = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1.0, 10.0],
        'max_iter': [100, 200, 500]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    },
    'SVM': {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['rbf', 'linear', 'poly'],
        'gamma': ['scale', 'auto']
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'Decision Tree': {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}


def render_model_lab():
    """渲染模型实验室页面"""
    st.title("模型实验室")
    
    st.markdown("""
    <div style="background: #fdedec; border-left: 4px solid #e74c3c; padding: 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;">
        <strong>模型训练与评估 (Model Training & Evaluation)</strong><br/>
        对比多种机器学习算法，进行超参数调优，评估模型性能，选择最优模型。
    </div>
    """, unsafe_allow_html=True)
    
    # 数据准备
    X, y, feature_names = prepare_model_data()
    if X is None:
        return
    
    # 标签页
    tab1, tab2, tab3, tab4 = st.tabs([
        "多模型对比", "超参数调优", "模型评估", "模型导出"
    ])
    
    with tab1:
        render_model_comparison(X, y, feature_names)
    
    with tab2:
        render_hyperparameter_tuning(X, y)
    
    with tab3:
        render_model_evaluation(X, y, feature_names)
    
    with tab4:
        render_model_export(X, y)


def prepare_model_data():
    """准备模型训练数据"""
    if 'data' not in st.session_state:
        try:
            data = pd.read_csv(PathConfig.DEFAULT_DATA_FILE)
            st.session_state['data'] = data
        except:
            st.warning("请先在「数据分析」页面加载数据")
            return None, None, None
    
    data = st.session_state['data']
    
    # 检查目标变量
    if 'Result' not in data.columns:
        st.warning("数据集缺少目标变量 'Result'")
        return None, None, None
    
    # 准备特征和目标
    df = data.copy()
    le = LabelEncoder()
    y = le.fit_transform(df['Result'])
    
    feature_cols = [c for c in df.columns if c != 'Result']
    X = df[feature_cols].copy()
    
    # 编码分类变量
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    # 处理缺失值
    X = X.fillna(X.median())
    
    # 显示数据信息
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("样本数量", len(y))
    with col2:
        st.metric("特征数量", X.shape[1])
    with col3:
        st.metric("阳性样本", sum(y))
    with col4:
        st.metric("阴性样本", len(y) - sum(y))
    
    return X, y, feature_cols


def render_model_comparison(X, y, feature_names):
    """多模型对比"""
    st.subheader("多模型对比训练")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("##### 选择模型")
        selected_models = st.multiselect(
            "选择要对比的模型",
            list(AVAILABLE_MODELS.keys()),
            default=list(AVAILABLE_MODELS.keys())[:4]
        )
        
        test_size = st.slider("测试集比例", 0.1, 0.4, 0.2)
        cv_folds = st.slider("交叉验证折数", 3, 10, 5)
        
        run_comparison = st.button("开始训练对比", type="primary", use_container_width=True)
    
    with col2:
        if run_comparison and selected_models:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, model_name in enumerate(selected_models):
                status_text.text(f"训练模型: {model_name}...")
                
                model_class = AVAILABLE_MODELS[model_name]
                
                if model_name == 'SVM':
                    model = model_class(probability=True, random_state=42)
                elif model_name in ['Logistic Regression']:
                    model = model_class(random_state=42, max_iter=500)
                elif model_name == 'KNN':
                    model = model_class()
                else:
                    model = model_class(random_state=42)
                
                start_time = time.time()
                
                # 交叉验证
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds)
                
                # 训练并预测
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                
                train_time = time.time() - start_time
                
                results.append({
                    '模型': model_name,
                    '准确率': accuracy_score(y_test, y_pred),
                    '精确率': precision_score(y_test, y_pred),
                    '召回率': recall_score(y_test, y_pred),
                    'F1分数': f1_score(y_test, y_pred),
                    'AUC': roc_auc_score(y_test, y_prob) if y_prob is not None else None,
                    'CV均值': cv_scores.mean(),
                    'CV标准差': cv_scores.std(),
                    '训练时间(s)': round(train_time, 3)
                })
                
                progress_bar.progress((i + 1) / len(selected_models))
            
            status_text.text("训练完成!")
            
            # 保存结果
            st.session_state['model_results'] = results
            st.session_state['X_train'] = X_train_scaled
            st.session_state['X_test'] = X_test_scaled
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['scaler'] = scaler
    
    # 显示结果
    if 'model_results' in st.session_state:
        results = st.session_state['model_results']
        results_df = pd.DataFrame(results)
        
        st.markdown("---")
        st.markdown("##### 模型性能对比")
        
        # 表格
        st.dataframe(
            results_df.style.format({
                '准确率': '{:.4f}', '精确率': '{:.4f}', '召回率': '{:.4f}',
                'F1分数': '{:.4f}', 'AUC': '{:.4f}', 'CV均值': '{:.4f}',
                'CV标准差': '{:.4f}'
            }).background_gradient(subset=['准确率', 'F1分数', 'AUC'], cmap='Greens'),
            use_container_width=True, hide_index=True
        )
        
        # 雷达图
        col1, col2 = st.columns(2)
        
        with col1:
            metrics = ['准确率', '精确率', '召回率', 'F1分数']
            fig = go.Figure()
            
            for _, row in results_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row[m] for m in metrics],
                    theta=metrics,
                    fill='toself',
                    name=row['模型']
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title='模型性能雷达图',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 柱状图对比
            fig = px.bar(results_df, x='模型', y=['准确率', 'F1分数', 'AUC'],
                        barmode='group', title='关键指标对比')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def render_hyperparameter_tuning(X, y):
    """超参数调优"""
    st.subheader("超参数调优")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        model_name = st.selectbox("选择模型", list(MODEL_PARAMS.keys()))
        
        st.markdown("##### 参数搜索空间")
        params = MODEL_PARAMS[model_name]
        
        selected_params = {}
        for param, values in params.items():
            if isinstance(values[0], (int, float)):
                selected_params[param] = st.multiselect(
                    param, values, default=values
                )
            else:
                selected_params[param] = st.multiselect(
                    param, values, default=values
                )
        
        cv_folds = st.slider("交叉验证折数", 3, 10, 5, key="tune_cv")
        
        run_tuning = st.button("开始调优", type="primary", use_container_width=True)
    
    with col2:
        if run_tuning:
            # 数据划分
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model_class = AVAILABLE_MODELS[model_name]
            
            if model_name == 'SVM':
                base_model = model_class(probability=True, random_state=42)
            elif model_name == 'Logistic Regression':
                base_model = model_class(random_state=42, max_iter=500)
            elif model_name == 'KNN':
                base_model = model_class()
            else:
                base_model = model_class(random_state=42)
            
            with st.spinner("网格搜索中，请稍候..."):
                grid_search = GridSearchCV(
                    base_model, selected_params, cv=cv_folds,
                    scoring='f1', n_jobs=-1, verbose=0
                )
                grid_search.fit(X_train_scaled, y_train)
            
            st.success("调优完成!")
            
            st.markdown("##### 最优参数")
            best_params_df = pd.DataFrame([grid_search.best_params_])
            st.dataframe(best_params_df, hide_index=True)
            
            st.markdown(f"##### 最优 F1 分数: {grid_search.best_score_:.4f}")
            
            # 测试集评估
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test_scaled)
            
            st.markdown("##### 测试集性能")
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("准确率", f"{accuracy_score(y_test, y_pred):.4f}")
            with col_b:
                st.metric("精确率", f"{precision_score(y_test, y_pred):.4f}")
            with col_c:
                st.metric("召回率", f"{recall_score(y_test, y_pred):.4f}")
            with col_d:
                st.metric("F1分数", f"{f1_score(y_test, y_pred):.4f}")
            
            # 保存最优模型
            st.session_state['best_model'] = best_model
            st.session_state['best_model_name'] = model_name
            st.session_state['best_scaler'] = scaler
            
            # 搜索结果可视化
            st.markdown("---")
            st.markdown("##### 参数搜索结果")
            
            cv_results = pd.DataFrame(grid_search.cv_results_)
            top_results = cv_results.nsmallest(10, 'rank_test_score')[
                ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
            ]
            top_results.columns = ['参数', '平均分数', '标准差', '排名']
            st.dataframe(top_results.round(4), hide_index=True)


def render_model_evaluation(X, y, feature_names):
    """模型评估"""
    st.subheader("模型评估")
    
    if 'y_test' not in st.session_state:
        st.info("请先在「多模型对比」标签页训练模型")
        return
    
    model_name = st.selectbox("选择评估模型", list(AVAILABLE_MODELS.keys()), key="eval_model")
    
    # 训练选定模型
    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train = st.session_state['y_train']
    y_test = st.session_state['y_test']
    
    model_class = AVAILABLE_MODELS[model_name]
    if model_name == 'SVM':
        model = model_class(probability=True, random_state=42)
    elif model_name == 'Logistic Regression':
        model = model_class(random_state=42, max_iter=500)
    elif model_name == 'KNN':
        model = model_class()
    else:
        model = model_class(random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 混淆矩阵
        st.markdown("##### 混淆矩阵")
        cm = confusion_matrix(y_test, y_pred)
        
        fig = px.imshow(cm, labels=dict(x="预测", y="实际", color="数量"),
                       x=['阴性', '阳性'], y=['阴性', '阳性'],
                       text_auto=True, color_continuous_scale='Blues')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        # 分类报告
        st.markdown("##### 分类报告")
        report = classification_report(y_test, y_pred, target_names=['阴性', '阳性'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4), use_container_width=True)
    
    with col2:
        # ROC曲线
        st.markdown("##### ROC 曲线")
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                    name=f'{model_name} (AUC={roc_auc:.4f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                    name='随机猜测', line=dict(dash='dash')))
            fig.update_layout(
                xaxis_title='假阳性率 (FPR)',
                yaxis_title='真阳性率 (TPR)',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("该模型不支持概率预测")
        
        # 预测分布
        st.markdown("##### 预测概率分布")
        if y_prob is not None:
            prob_df = pd.DataFrame({
                '预测概率': y_prob,
                '实际标签': ['阳性' if i == 1 else '阴性' for i in y_test]
            })
            fig = px.histogram(prob_df, x='预测概率', color='实际标签',
                              barmode='overlay', nbins=30,
                              color_discrete_map={'阳性': '#e74c3c', '阴性': '#3498db'})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)


def render_model_export(X, y):
    """模型导出"""
    st.subheader("模型导出")
    
    if 'best_model' not in st.session_state:
        st.info("请先在「超参数调优」标签页训练最优模型")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 当前最优模型")
        st.write(f"模型类型: {st.session_state.get('best_model_name', 'N/A')}")
        st.write(f"模型对象: {type(st.session_state['best_model']).__name__}")
        
        if hasattr(st.session_state['best_model'], 'get_params'):
            st.markdown("##### 模型参数")
            params = st.session_state['best_model'].get_params()
            params_df = pd.DataFrame([params])
            st.dataframe(params_df.T.rename(columns={0: '值'}))
    
    with col2:
        st.markdown("##### 导出模型")
        
        model_filename = st.text_input("模型文件名", value="best_model.pkl")
        
        if st.button("导出模型", type="primary"):
            model_path = PathConfig.MODEL_DIR / model_filename
            scaler_path = PathConfig.MODEL_DIR / "scaler.pkl"
            
            PathConfig.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(st.session_state['best_model'], model_path)
            joblib.dump(st.session_state['best_scaler'], scaler_path)
            
            st.success(f"模型已导出到: {model_path}")
            st.success(f"标准化器已导出到: {scaler_path}")
