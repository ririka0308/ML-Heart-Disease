# -*- coding: utf-8 -*-
"""
可视化绘图模块
封装 Plotly/Matplotlib/SHAP 所有绘图函数
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MedicalReference, UIConfig

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class PlotManager:
    """可视化管理器"""
    
    # 颜色方案
    COLORS = {
        'primary': '#3498db',
        'success': '#2ecc71',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'info': '#17a2b8',
        'secondary': '#6c757d',
        'light': '#f8f9fa',
        'dark': '#343a40'
    }
    
    COLOR_SCALE = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    # ==================== 数据概览图表 ====================
    
    @staticmethod
    def plot_gender_distribution(df: pd.DataFrame) -> go.Figure:
        """性别分布饼图"""
        gender_counts = df['Gender'].value_counts()
        gender_labels = ['男性' if g == 1 else '女性' for g in gender_counts.index]
        
        fig = go.Figure(data=[go.Pie(
            labels=gender_labels,
            values=gender_counts.values,
            hole=0.4,
            marker=dict(colors=[PlotManager.COLORS['primary'], PlotManager.COLORS['danger']]),
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title='性别分布',
            height=350,
            showlegend=True
        )
        return fig
    
    @staticmethod
    def plot_age_distribution(df: pd.DataFrame) -> go.Figure:
        """年龄分布箱线图"""
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=df['Age'],
            name='年龄分布',
            marker_color=PlotManager.COLORS['primary'],
            boxpoints='outliers'
        ))
        
        fig.update_layout(
            title='年龄分布',
            yaxis_title='年龄（岁）',
            height=350
        )
        return fig
    
    @staticmethod
    def plot_target_distribution(df: pd.DataFrame, target_col: str = 'Result') -> go.Figure:
        """目标变量分布"""
        if df[target_col].dtype == 'object':
            counts = df[target_col].value_counts()
            labels = ['阳性' if v == 'positive' else '阴性' for v in counts.index]
        else:
            counts = df[target_col].value_counts()
            labels = ['阳性' if v == 1 else '阴性' for v in counts.index]
        
        colors = [PlotManager.COLORS['danger'], PlotManager.COLORS['success']]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=counts.values,
            hole=0.5,
            marker=dict(colors=colors),
            textinfo='label+percent+value',
            textposition='inside'
        )])
        
        fig.update_layout(
            title='诊断结果分布',
            height=400,
            annotations=[dict(text='诊断', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        return fig
    
    @staticmethod
    def plot_feature_histogram(df: pd.DataFrame, feature: str, 
                               by_result: bool = True) -> go.Figure:
        """特征分布直方图"""
        if by_result and 'Result' in df.columns:
            fig = px.histogram(
                df, x=feature, color='Result',
                color_discrete_map={'positive': PlotManager.COLORS['danger'],
                                   'negative': PlotManager.COLORS['primary']},
                barmode='overlay',
                opacity=0.7,
                marginal='rug'
            )
        else:
            fig = px.histogram(df, x=feature, color_discrete_sequence=[PlotManager.COLORS['primary']])
        
        feature_cn = MedicalReference.FEATURE_NAMES_CN.get(feature, feature)
        fig.update_layout(
            title=f'{feature_cn} 分布',
            xaxis_title=feature_cn,
            yaxis_title='频数',
            height=400
        )
        return fig
    
    @staticmethod
    def plot_feature_boxplot(df: pd.DataFrame, feature: str) -> go.Figure:
        """特征箱线图（按诊断结果分组）"""
        fig = px.box(
            df, x='Result', y=feature,
            color='Result',
            color_discrete_map={'positive': PlotManager.COLORS['danger'],
                               'negative': PlotManager.COLORS['primary']}
        )
        
        feature_cn = MedicalReference.FEATURE_NAMES_CN.get(feature, feature)
        fig.update_layout(
            title=f'{feature_cn} 按诊断结果分布',
            xaxis_title='诊断结果',
            yaxis_title=feature_cn,
            height=400
        )
        return fig
    
    # ==================== 相关性分析图表 ====================
    
    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame, 
                                 numeric_cols: List[str] = None) -> go.Figure:
        """相关性热力图"""
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={'size': 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='特征相关性热力图',
            height=600,
            xaxis_nticks=len(corr_matrix.columns),
            yaxis_nticks=len(corr_matrix.index)
        )
        return fig
    
    @staticmethod
    def plot_feature_importance(feature_names: List[str], 
                               importance_values: np.ndarray,
                               title: str = '特征重要性') -> go.Figure:
        """特征重要性条形图"""
        # 排序
        sorted_idx = np.argsort(importance_values)
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_values = importance_values[sorted_idx]
        
        # 中文名称转换
        sorted_names_cn = [MedicalReference.FEATURE_NAMES_CN.get(n, n) for n in sorted_names]
        
        fig = go.Figure(go.Bar(
            x=sorted_values,
            y=sorted_names_cn,
            orientation='h',
            marker=dict(
                color=sorted_values,
                colorscale='Viridis'
            ),
            text=[f'{v:.4f}' for v in sorted_values],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='重要性分数',
            yaxis_title='特征',
            height=400
        )
        return fig
    
    # ==================== 模型评估图表 ====================
    
    @staticmethod
    def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, 
                       auc_score: float, model_name: str = 'Model') -> go.Figure:
        """ROC曲线"""
        fig = go.Figure()
        
        # ROC曲线
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {auc_score:.4f})',
            line=dict(color=PlotManager.COLORS['primary'], width=2)
        ))
        
        # 对角线
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='随机分类器',
            line=dict(color=PlotManager.COLORS['secondary'], width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='ROC曲线',
            xaxis_title='假阳性率 (FPR)',
            yaxis_title='真阳性率 (TPR)',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
            height=500,
            legend=dict(x=0.6, y=0.1)
        )
        return fig
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, 
                             labels: List[str] = None) -> go.Figure:
        """混淆矩阵热力图"""
        if labels is None:
            labels = ['阴性', '阳性']
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f'预测{l}' for l in labels],
            y=[f'实际{l}' for l in labels],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={'size': 20}
        ))
        
        fig.update_layout(
            title='混淆矩阵',
            height=400
        )
        return fig
    
    @staticmethod
    def plot_model_comparison(results_df: pd.DataFrame) -> go.Figure:
        """多模型性能对比图"""
        metrics = ['准确率', '精确率', '召回率', 'F1分数', 'AUC-ROC']
        
        fig = go.Figure()
        
        for metric in metrics:
            if metric in results_df.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=results_df.index if 'model' not in results_df.columns else results_df['model'],
                    y=results_df[metric],
                    text=results_df[metric].round(4),
                    textposition='outside'
                ))
        
        fig.update_layout(
            title='模型性能对比',
            barmode='group',
            xaxis_title='模型',
            yaxis_title='得分',
            yaxis_range=[0, 1.1],
            height=500,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        return fig
    
    # ==================== SHAP 可解释性图表 ====================
    
    @staticmethod
    def plot_shap_waterfall(shap_values: np.ndarray, 
                            feature_names: List[str],
                            feature_values: np.ndarray = None,
                            base_value: float = 0.5) -> go.Figure:
        """
        SHAP瀑布图 - 解释单个样本的预测
        
        Args:
            shap_values: SHAP值数组
            feature_names: 特征名称列表
            feature_values: 特征实际值（可选）
            base_value: 基准值
        """
        # 排序
        sorted_idx = np.argsort(np.abs(shap_values))[::-1]
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_values = shap_values[sorted_idx]
        
        # 转换为中文
        sorted_names_cn = [MedicalReference.FEATURE_NAMES_CN.get(n, n) for n in sorted_names]
        
        # 如果有特征值，添加到标签
        if feature_values is not None:
            sorted_feature_vals = [feature_values[i] for i in sorted_idx]
            labels = [f'{n} = {v:.2f}' for n, v in zip(sorted_names_cn, sorted_feature_vals)]
        else:
            labels = sorted_names_cn
        
        # 颜色
        colors = [PlotManager.COLORS['danger'] if v > 0 else PlotManager.COLORS['primary'] 
                  for v in sorted_values]
        
        fig = go.Figure(go.Bar(
            x=sorted_values,
            y=labels,
            orientation='h',
            marker_color=colors,
            text=[f'{v:+.4f}' for v in sorted_values],
            textposition='outside'
        ))
        
        # 添加基准线
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title='SHAP特征贡献分析（瀑布图）',
            xaxis_title='SHAP值（对预测概率的影响）',
            yaxis_title='特征',
            height=350,
            annotations=[dict(
                text='红色: 增加患病风险 | 蓝色: 降低患病风险',
                xref='paper', yref='paper',
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=12, color='gray')
            )]
        )
        return fig
    
    @staticmethod
    def plot_shap_summary(shap_values: np.ndarray,
                          feature_names: List[str],
                          X: pd.DataFrame = None) -> go.Figure:
        """
        SHAP全局摘要图 - 显示所有特征的整体重要性
        
        Args:
            shap_values: 所有样本的SHAP值矩阵 [n_samples, n_features]
            feature_names: 特征名称列表
            X: 特征数据（用于着色）
        """
        # 计算平均绝对SHAP值
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # 排序
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_importance = mean_abs_shap[sorted_idx]
        
        # 转换为中文
        sorted_names_cn = [MedicalReference.FEATURE_NAMES_CN.get(n, n) for n in sorted_names]
        
        fig = go.Figure(go.Bar(
            x=sorted_importance,
            y=sorted_names_cn,
            orientation='h',
            marker=dict(
                color=sorted_importance,
                colorscale='Viridis'
            ),
            text=[f'{v:.4f}' for v in sorted_importance],
            textposition='outside'
        ))
        
        fig.update_layout(
            title='SHAP特征重要性（全局）',
            xaxis_title='平均|SHAP值|',
            yaxis_title='特征',
            height=400
        )
        return fig
    
    @staticmethod
    def plot_shap_force(shap_values: np.ndarray,
                        feature_names: List[str],
                        feature_values: np.ndarray,
                        base_value: float,
                        prediction: float) -> go.Figure:
        """
        SHAP力图 - 显示特征如何推动预测
        """
        # 创建子图
        fig = make_subplots(rows=1, cols=1)
        
        # 排序
        sorted_idx = np.argsort(np.abs(shap_values))[::-1][:6]  # Top 6
        
        # 基准值
        cumulative = base_value
        
        # 添加累积条形图
        for i, idx in enumerate(sorted_idx):
            val = shap_values[idx]
            name = MedicalReference.FEATURE_NAMES_CN.get(feature_names[idx], feature_names[idx])
            color = PlotManager.COLORS['danger'] if val > 0 else PlotManager.COLORS['primary']
            
            fig.add_trace(go.Bar(
                x=[name],
                y=[abs(val)],
                name=f'{name}: {val:+.3f}',
                marker_color=color,
                text=[f'{val:+.3f}'],
                textposition='outside'
            ))
        
        fig.update_layout(
            title=f'预测解释 (预测概率: {prediction:.2%})',
            barmode='stack',
            height=300,
            showlegend=True
        )
        return fig
    
    # ==================== 预测结果图表 ====================
    
    @staticmethod
    def plot_prediction_gauge(probability: float) -> go.Figure:
        """预测结果仪表盘"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "心脏病风险指数", 'font': {'size': 20}},
            number={'suffix': '%', 'font': {'size': 40}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': PlotManager.COLORS['success']},
                    {'range': [30, 60], 'color': PlotManager.COLORS['warning']},
                    {'range': [60, 100], 'color': PlotManager.COLORS['danger']}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    @staticmethod
    def plot_prediction_pie(probability: float) -> go.Figure:
        """预测概率饼图"""
        fig = go.Figure(data=[go.Pie(
            labels=['阴性', '阳性'],
            values=[1 - probability, probability],
            hole=0.5,
            marker=dict(colors=[PlotManager.COLORS['primary'], PlotManager.COLORS['danger']]),
            textinfo='label+percent',
            textposition='inside'
        )])
        
        risk_level = "高风险" if probability > 0.5 else "低风险"
        color = PlotManager.COLORS['danger'] if probability > 0.5 else PlotManager.COLORS['success']
        
        fig.update_layout(
            title='预测概率分布',
            height=350,
            annotations=[dict(
                text=risk_level,
                x=0.5, y=0.5,
                font_size=18,
                font_color=color,
                showarrow=False
            )]
        )
        return fig
    
    # ==================== Dashboard 统计图表 ====================
    
    @staticmethod
    def plot_daily_predictions(df: pd.DataFrame) -> go.Figure:
        """每日预测数量趋势图"""
        df['date'] = pd.to_datetime(df['created_at']).dt.date
        daily_counts = df.groupby('date').size().reset_index(name='count')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_counts['date'],
            y=daily_counts['count'],
            mode='lines+markers',
            name='预测数量',
            line=dict(color=PlotManager.COLORS['primary'], width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='每日预测数量趋势',
            xaxis_title='日期',
            yaxis_title='预测数量',
            height=350
        )
        return fig
    
    @staticmethod
    def plot_risk_ratio_trend(df: pd.DataFrame) -> go.Figure:
        """高风险比例趋势图"""
        df['date'] = pd.to_datetime(df['created_at']).dt.date
        daily_stats = df.groupby('date').agg({
            'predict_result': ['sum', 'count']
        }).reset_index()
        daily_stats.columns = ['date', 'high_risk', 'total']
        daily_stats['ratio'] = daily_stats['high_risk'] / daily_stats['total'] * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['ratio'],
            mode='lines+markers',
            name='高风险比例',
            line=dict(color=PlotManager.COLORS['danger'], width=2),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.2)'
        ))
        
        fig.update_layout(
            title='高风险患者比例趋势',
            xaxis_title='日期',
            yaxis_title='高风险比例 (%)',
            height=350
        )
        return fig


# 创建全局绘图管理器实例
plot_manager = PlotManager()
