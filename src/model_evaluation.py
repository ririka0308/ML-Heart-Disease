"""
模型评估模块
包含交叉验证、ROC曲线、混淆矩阵等评估工具
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)


class ModelEvaluator:
    """模型评估类"""

    def __init__(self):
        self.evaluation_results = {}

    def calculate_metrics(self, y_true, y_pred, y_prob):
        """
        计算评估指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率

        Returns:
            指标字典
        """
        metrics = {
            '准确率': accuracy_score(y_true, y_pred),
            '精确率': precision_score(y_true, y_pred),
            '召回率': recall_score(y_true, y_pred),
            'F1分数': f1_score(y_true, y_pred),
            'AUC-ROC': roc_auc_score(y_true, y_prob)
        }
        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, title='混淆矩阵'):
        """
        绘制混淆矩阵

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            title: 图表标题

        Returns:
            matplotlib图对象
        """
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='g',
            cmap='Blues',
            xticklabels=['预测为阴性', '预测为阳性'],
            yticklabels=['实际为阴性', '实际为阳性'],
            ax=ax
        )
        ax.set_title(title)
        ax.set_xlabel('预测结果')
        ax.set_ylabel('实际结果')

        return fig

    def plot_roc_curve(self, y_true, y_prob, title='ROC曲线'):
        """
        绘制ROC曲线

        Args:
            y_true: 真实标签
            y_prob: 预测概率
            title: 图表标题

        Returns:
            matplotlib图对象和AUC值
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            fpr,
            tpr,
            color='darkorange',
            lw=2,
            label=f'ROC曲线 (AUC = {roc_auc:.4f})'
        )
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('假阳性率 (FPR)')
        ax.set_ylabel('真阳性率 (TPR)')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        return fig, roc_auc

    def plot_precision_recall_curve(self, y_true, y_prob, title='精确率-召回率曲线'):
        """
        绘制精确率-召回率曲线

        Args:
            y_true: 真实标签
            y_prob: 预测概率
            title: 图表标题

        Returns:
            matplotlib图对象和AUC值
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            recall,
            precision,
            color='blue',
            lw=2,
            label=f'PR曲线 (AUC = {pr_auc:.4f})'
        )
        ax.set_xlabel('召回率 (Recall)')
        ax.set_ylabel('精确率 (Precision)')
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        return fig, pr_auc

    def cross_validate_model(self, model, X, y, cv=5, scoring='roc_auc'):
        """
        交叉验证

        Args:
            model: 模型对象
            X: 特征数据
            y: 目标变量
            cv: 交叉验证折数
            scoring: 评分指标

        Returns:
            交叉验证结果
        """
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        results = {
            '均值': scores.mean(),
            '标准差': scores.std(),
            '分数': scores.tolist(),
            '最小值': scores.min(),
            '最大值': scores.max()
        }

        return results

    def compare_models(self, models_dict, X_test, y_test):
        """
        比较多个模型的性能

        Args:
            models_dict: 模型字典 {name: model}
            X_test: 测试特征
            y_test: 测试标签

        Returns:
            比较结果DataFrame
        """
        results = {}

        for name, model in models_dict.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = self.calculate_metrics(y_test, y_pred, y_prob)
            results[name] = metrics

        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values('AUC-ROC', ascending=False)

        return results_df

    def generate_classification_report(self, y_true, y_pred):
        """
        生成分类报告

        Args:
            y_true: 真实标签
            y_pred: 预测标签

        Returns:
            分类报告DataFrame
        """
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        return report_df

    def plot_feature_importance(self, feature_names, importance_scores, top_n=15):
        """
        绘制特征重要性

        Args:
            feature_names: 特征名称列表
            importance_scores: 重要性分数列表
            top_n: 显示前N个特征

        Returns:
            matplotlib图对象
        """
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(
            x='Importance',
            y='Feature',
            data=importance_df.head(top_n),
            ax=ax,
            palette='viridis'
        )
        ax.set_title(f'特征重要性 (Top {top_n})')
        ax.set_xlabel('重要性分数')
        ax.set_ylabel('特征')

        return fig


def evaluate_single_model(model, X_test, y_test):
    """评估单个模型"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob)
    }

    return metrics


def plot_model_comparison(results_df):
    """绘制模型对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    results_df.plot(kind='bar', ax=ax, colormap='viridis')
    ax.set_title('各模型性能指标对比')
    ax.set_ylabel('分数')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig
