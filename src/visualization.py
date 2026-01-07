"""
可视化模块
包含数据探索、模型结果等各种可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    """可视化类"""

    def __init__(self):
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

    def plot_distribution(self, data, column, hue=None):
        """
        绘制特征分布图

        Args:
            data: 数据
            column: 列名
            hue: 分组列名

        Returns:
            matplotlib图对象
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 直方图
        sns.histplot(data=data, x=column, hue=hue, kde=True, ax=axes[0])
        axes[0].set_title(f'{column} 分布直方图')
        axes[0].set_xlabel(column)
        axes[0].set_ylabel('数量')

        # 箱线图
        if hue:
            sns.boxplot(data=data, x=hue, y=column, ax=axes[1])
            axes[1].set_title(f'{column} 分布箱线图')
            axes[1].set_xlabel(hue)
            axes[1].set_ylabel(column)
        else:
            sns.boxplot(data=data, y=column, ax=axes[1])
            axes[1].set_title(f'{column} 箱线图')
            axes[1].set_ylabel(column)

        return fig

    def plot_target_distribution(self, data, target_column='Result'):
        """
        绘制目标变量分布

        Args:
            data: 数据
            target_column: 目标列名

        Returns:
            matplotlib图对象
        """
        result_counts = data[target_column].value_counts()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 柱状图
        result_counts.plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'])
        axes[0].set_title('目标变量分布（柱状图）')
        axes[0].set_xlabel('结果')
        axes[0].set_ylabel('数量')
        axes[0].tick_params(axis='x', rotation=0)

        # 饼图
        result_counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
                          colors=['#3498db', '#e74c3c'])
        axes[1].set_title('目标变量分布（饼图）')
        axes[1].set_ylabel('')

        return fig

    def plot_correlation_heatmap(self, data):
        """
        绘制相关性热力图

        Args:
            data: 数据

        Returns:
            matplotlib图对象
        """
        # 转换目标变量为数值
        data_corr = data.copy()
        if 'Result' in data_corr.columns:
            data_corr['Result'] = data_corr['Result'].map({'negative': 0, 'positive': 1})

        # 计算相关性矩阵
        corr_matrix = data_corr.select_dtypes(include=[np.number]).corr()

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            linewidths=0.5,
            ax=ax,
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title('特征相关性热力图')

        return fig

    def plot_correlation_with_target(self, data, target_column='Result'):
        """
        绘制特征与目标变量的相关性

        Args:
            data: 数据
            target_column: 目标列名

        Returns:
            matplotlib图对象
        """
        data_corr = data.copy()
        data_corr[target_column] = data_corr[target_column].map({'negative': 0, 'positive': 1})

        corr_matrix = data_corr.select_dtypes(include=[np.number]).corr()
        corr_with_target = corr_matrix[target_column].sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        corr_with_target.drop(target_column).plot(kind='barh', ax=ax,
                                                  color=['#3498db', '#e74c3c'])
        ax.set_title('特征与目标变量相关性')
        ax.set_xlabel('相关性系数')
        ax.set_ylabel('特征')

        return fig

    def plot_scatter_matrix(self, data, features, hue=None):
        """
        绘制散点图矩阵

        Args:
            data: 数据
            features: 特征列表
            hue: 分组列名

        Returns:
            matplotlib图对象
        """
        if hue:
            colors = data[hue].map({'negative': 'blue', 'positive': 'red'})
        else:
            colors = None

        fig, ax = plt.subplots(figsize=(12, 12))
        pd.plotting.scatter_matrix(
            data[features],
            c=colors,
            figsize=(12, 12),
            ax=ax,
            alpha=0.6
        )
        plt.tight_layout()

        return fig

    def plot_model_comparison(self, results_df):
        """
        绘制模型性能对比

        Args:
            results_df: 结果DataFrame

        Returns:
            matplotlib图对象
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        results_df.plot(kind='bar', ax=ax, colormap='viridis')
        ax.set_title('各模型性能指标对比')
        ax.set_ylabel('分数')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def plot_prediction_probability(self, probability, prediction):
        """
        绘制预测概率

        Args:
            probability: 阳性概率
            prediction: 预测结果

        Returns:
            matplotlib图对象
        """
        probabilities = [1 - probability, probability]
        labels = ['阴性', '阳性']
        colors = ['#3498db', '#e74c3c']

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, probabilities, color=colors, alpha=0.8)
        ax.set_title('心脏病发病风险预测概率')
        ax.set_ylabel('概率')
        ax.set_ylim([0, 1])

        for i, v in enumerate(probabilities):
            ax.text(i, v + 0.02, f'{v:.2%}', ha='center',
                   fontweight='bold', fontsize=12)

        return fig

    def plot_batch_predictions(self, predictions_df):
        """
        绘制批量预测结果

        Args:
            predictions_df: 预测结果DataFrame

        Returns:
            matplotlib图对象
        """
        pos_count = (predictions_df['Prediction'] == 'Positive').sum()
        neg_count = (predictions_df['Prediction'] == 'Negative').sum()

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 饼图
        axes[0].pie([neg_count, pos_count],
                   labels=['阴性', '阳性'],
                   autopct='%1.1f%%',
                   colors=['#3498db', '#e74c3c'],
                   startangle=90)
        axes[0].set_title('预测结果分布')

        # 柱状图
        axes[1].bar(['阴性', '阳性'], [neg_count, pos_count],
                   color=['#3498db', '#e74c3c'])
        axes[1].set_title('预测结果数量')
        axes[1].set_ylabel('数量')

        for i, v in enumerate([neg_count, pos_count]):
            axes[1].text(i, v + max(neg_count, pos_count) * 0.02,
                        str(v), ha='center', fontweight='bold')

        return fig


def plot_feature_distributions(data, features, target_column='Result'):
    """绘制多个特征的分布"""
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]

    for i, feature in enumerate(features):
        if i < len(axes):
            ax = axes[i]
            for label in data[target_column].unique():
                subset = data[data[target_column] == label]
                sns.histplot(data=subset, x=feature, ax=ax,
                           label=label, alpha=0.6, kde=True)
            ax.set_title(f'{feature} 分布')
            ax.legend()

    plt.tight_layout()
    return fig


def plot_pairplot(data, features, target_column='Result'):
    """绘制特征配对图"""
    plot_data = data[features + [target_column]].copy()
    if target_column in plot_data.columns:
        plot_data[target_column] = plot_data[target_column].map(
            {'negative': '阴性', 'positive': '阳性'}
        )

    fig = sns.pairplot(plot_data, hue=target_column, diag_kind='hist',
                       plot_kws={'alpha': 0.6}, diag_kws={'alpha': 0.6})

    return fig
