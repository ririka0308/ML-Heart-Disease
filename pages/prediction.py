# -*- coding: utf-8 -*-
"""
预测演示页面
核心功能：输入特征预测、SHAP可解释性分析
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PathConfig

# 特征配置
FEATURE_CONFIG = {
    'Age': {'label': '年龄', 'min': 20, 'max': 90, 'default': 50, 'unit': '岁'},
    'Gender': {'label': '性别', 'options': ['Male', 'Female']},
    'Heart rate': {'label': '心率', 'min': 40, 'max': 200, 'default': 75, 'unit': 'bpm'},
    'Systolic blood pressure': {'label': '收缩压', 'min': 80, 'max': 220, 'default': 120, 'unit': 'mmHg'},
    'Diastolic blood pressure': {'label': '舒张压', 'min': 40, 'max': 140, 'default': 80, 'unit': 'mmHg'},
    'Blood sugar': {'label': '血糖', 'min': 50, 'max': 400, 'default': 100, 'unit': 'mg/dL'},
    'CK-MB': {'label': 'CK-MB', 'min': 0.0, 'max': 100.0, 'default': 2.0, 'unit': 'ng/mL'},
    'Troponin': {'label': '肌钙蛋白', 'min': 0.0, 'max': 10.0, 'default': 0.01, 'unit': 'ng/mL'}
}


def render_prediction():
    """渲染预测演示页面"""
    st.title("智能预测")
    
    st.markdown("""
    <div style="background: #f5eef8; border-left: 4px solid #9b59b6; padding: 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;">
        <strong>风险预测与解释 (Risk Prediction & Interpretation)</strong><br/>
        输入患者特征进行心脏病风险预测，并通过SHAP分析解释预测结果。
    </div>
    """, unsafe_allow_html=True)
    
    # 加载/训练模型
    model, scaler, feature_names = get_model()
    if model is None:
        st.error("模型加载失败")
        return
    
    # 标签页
    tab1, tab2 = st.tabs(["单例预测", "批量预测"])
    
    with tab1:
        render_single_prediction(model, scaler, feature_names)
    
    with tab2:
        render_batch_prediction(model, scaler, feature_names)


def get_model():
    """获取或训练模型"""
    if 'prediction_model' in st.session_state:
        return (
            st.session_state['prediction_model'],
            st.session_state['prediction_scaler'],
            st.session_state['prediction_features']
        )
    
    # 加载数据训练模型
    try:
        data = pd.read_csv(PathConfig.DEFAULT_DATA_FILE)
    except:
        st.error("无法加载数据集")
        return None, None, None
    
    # 准备数据
    df = data.copy()
    le = LabelEncoder()
    y = le.fit_transform(df['Result'])
    
    feature_cols = [c for c in df.columns if c != 'Result']
    X = df[feature_cols].copy()
    
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    X = X.fillna(X.median())
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    st.session_state['prediction_model'] = model
    st.session_state['prediction_scaler'] = scaler
    st.session_state['prediction_features'] = feature_cols
    st.session_state['training_data'] = X
    
    return model, scaler, feature_cols


def render_single_prediction(model, scaler, feature_names):
    """单例预测"""
    st.subheader("输入患者特征")
    
    input_data = {}
    
    # 输入表单
    col1, col2, col3 = st.columns(3)
    
    for i, feature in enumerate(feature_names):
        config = FEATURE_CONFIG.get(feature, {})
        label = config.get('label', feature)
        
        col = [col1, col2, col3][i % 3]
        
        with col:
            if 'options' in config:
                # 分类变量
                value = st.selectbox(label, config['options'], key=f"input_{feature}")
                input_data[feature] = 1 if value == config['options'][0] else 0
            else:
                # 数值变量
                min_val = config.get('min', 0)
                max_val = config.get('max', 100)
                default_val = config.get('default', (min_val + max_val) // 2)
                unit = config.get('unit', '')
                
                input_data[feature] = st.number_input(
                    f"{label} ({unit})" if unit else label,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    key=f"input_{feature}"
                )
    
    st.markdown("---")
    
    # 预测按钮
    if st.button("执行预测", type="primary", use_container_width=True):
        # 准备输入
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_names]
        input_scaled = scaler.transform(input_df)
        
        # 预测
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        st.markdown("---")
        st.subheader("预测结果")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if prediction == 1:
                st.error("### 高风险")
                st.metric("心脏病风险", f"{probability[1]*100:.1f}%", delta="需关注")
            else:
                st.success("### 低风险")
                st.metric("心脏病风险", f"{probability[1]*100:.1f}%", delta="正常")
            
            # 概率图
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability[1] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "风险概率 (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#e74c3c" if prediction == 1 else "#27ae60"},
                    'steps': [
                        {'range': [0, 30], 'color': "#d5f5e3"},
                        {'range': [30, 60], 'color': "#fdebd0"},
                        {'range': [60, 100], 'color': "#fadbd8"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # SHAP 解释
            render_shap_explanation(model, input_scaled, feature_names)


def render_shap_explanation(model, input_data, feature_names):
    """SHAP 可解释性分析"""
    st.markdown("##### 特征贡献分析")
    
    try:
        import shap
        
        # 获取训练数据用于SHAP
        if 'training_data' in st.session_state:
            background_data = st.session_state['training_data'].values[:100]
            scaler = st.session_state['prediction_scaler']
            background_scaled = scaler.transform(background_data)
        else:
            st.warning("需要训练数据进行SHAP分析")
            return
        
        # 计算SHAP值
        with st.spinner("计算特征贡献..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)
            
            # 获取正类的SHAP值
            if isinstance(shap_values, list):
                shap_vals = np.array(shap_values[1]).flatten()
            else:
                shap_vals = np.array(shap_values).flatten()
            
            # 确保特征值是1维数组
            feature_values = np.array(input_data).flatten()
            
            # 创建SHAP贡献图
            contribution_df = pd.DataFrame({
                '特征': feature_names,
                '贡献值': shap_vals[:len(feature_names)],
                '特征值': feature_values[:len(feature_names)]
            })
            contribution_df['贡献方向'] = contribution_df['贡献值'].apply(
                lambda x: '增加风险' if x > 0 else '降低风险'
            )
            contribution_df = contribution_df.sort_values('贡献值', key=abs, ascending=True)
            
            # 水平条形图
            fig = px.bar(
                contribution_df, x='贡献值', y='特征', orientation='h',
                color='贡献方向',
                color_discrete_map={'增加风险': '#e74c3c', '降低风险': '#27ae60'}
            )
            fig.update_layout(height=350, xaxis_title="SHAP贡献值", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示关键特征
            st.markdown("##### 关键影响因素")
            top_features = contribution_df.nlargest(3, '贡献值', keep='all')
            for _, row in top_features.iterrows():
                feature_label = FEATURE_CONFIG.get(row['特征'], {}).get('label', row['特征'])
                direction = "增加" if row['贡献值'] > 0 else "降低"
                st.write(f"- **{feature_label}**: {direction}心脏病风险 (贡献值: {row['贡献值']:.4f})")
    
    except ImportError:
        st.warning("SHAP库未安装，使用特征重要性替代")
        render_feature_importance_explanation(model, feature_names)
    except Exception as e:
        st.warning(f"SHAP分析失败: {e}")
        render_feature_importance_explanation(model, feature_names)


def render_feature_importance_explanation(model, feature_names):
    """基于特征重要性的解释（SHAP不可用时）"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            '特征': feature_names,
            '重要性': model.feature_importances_
        }).sort_values('重要性', ascending=True)
        
        fig = px.bar(importance_df, x='重要性', y='特征', orientation='h',
                    color='重要性', color_continuous_scale='Blues')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


def render_batch_prediction(model, scaler, feature_names):
    """批量预测"""
    st.subheader("批量预测")
    
    st.markdown("""
    上传CSV文件进行批量预测。文件应包含以下特征列：
    - Age, Gender, Heart rate, Systolic blood pressure
    - Diastolic blood pressure, Blood sugar, CK-MB, Troponin
    """)
    
    uploaded_file = st.file_uploader("上传预测数据 (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write(f"已加载 {len(batch_data)} 条记录")
            st.dataframe(batch_data.head(), use_container_width=True)
            
            if st.button("执行批量预测", type="primary"):
                # 检查特征
                missing_cols = set(feature_names) - set(batch_data.columns)
                if missing_cols:
                    st.error(f"缺少特征列: {missing_cols}")
                    return
                
                # 准备数据
                X_batch = batch_data[feature_names].copy()
                
                # 编码分类变量
                for col in X_batch.select_dtypes(include=['object']).columns:
                    X_batch[col] = LabelEncoder().fit_transform(X_batch[col].astype(str))
                
                X_batch = X_batch.fillna(X_batch.median())
                X_batch_scaled = scaler.transform(X_batch)
                
                # 预测
                predictions = model.predict(X_batch_scaled)
                probabilities = model.predict_proba(X_batch_scaled)[:, 1]
                
                # 结果
                result_df = batch_data.copy()
                result_df['预测结果'] = ['高风险' if p == 1 else '低风险' for p in predictions]
                result_df['风险概率'] = [f"{p*100:.1f}%" for p in probabilities]
                
                st.markdown("---")
                st.subheader("预测结果")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总样本数", len(predictions))
                with col2:
                    st.metric("高风险人数", sum(predictions))
                with col3:
                    st.metric("低风险人数", len(predictions) - sum(predictions))
                
                st.dataframe(result_df, use_container_width=True)
                
                # 风险分布
                fig = px.pie(result_df, names='预测结果', title='风险分布',
                            color='预测结果',
                            color_discrete_map={'高风险': '#e74c3c', '低风险': '#27ae60'})
                st.plotly_chart(fig, use_container_width=True)
                
                # 下载结果
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "下载预测结果",
                    csv,
                    "prediction_results.csv",
                    "text/csv"
                )
        
        except Exception as e:
            st.error(f"处理失败: {e}")
