"""
数据预处理模块
处理缺失值、异常值、重复值等
"""

import pandas as pd
import numpy as np


class DataPreprocessor:
    """数据预处理类"""

    def __init__(self):
        self.missing_values = {}
        self.outliers = {}
        self.duplicates = 0

    def handle_missing_values(self, data):
        """
        处理缺失值

        Args:
            data: 原始数据

        Returns:
            处理后的数据
        """
        df = data.copy()

        # 统计缺失值
        missing = df.isnull().sum()
        self.missing_values = {col: count for col, count in missing.items() if count > 0}

        # 填充缺失值
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # 数值型用中位数填充
                df[col].fillna(df[col].median(), inplace=True)
            else:
                # 类别型用众数填充
                df[col].fillna(df[col].mode()[0], inplace=True)

        return df

    def detect_outliers(self, data, columns=None):
        """
        检测异常值 (使用IQR方法)

        Args:
            data: 原始数据
            columns: 要检测的列名列表

        Returns:
            包含异常值信息的字典
        """
        df = data.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns

        outliers_dict = {}

        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outliers_dict[col] = {
                'count': len(outliers),
                'indices': outliers.index.tolist(),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }

        self.outliers = outliers_dict
        return outliers_dict

    def handle_outliers(self, data, columns=None, method='remove'):
        """
        处理异常值

        Args:
            data: 原始数据
            columns: 要处理的列名列表
            method: 处理方法 ('remove' | 'clip' | 'median')

        Returns:
            处理后的数据
        """
        df = data.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns

        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            if method == 'remove':
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            elif method == 'clip':
                df[col] = df[col].clip(lower_bound, upper_bound)
            elif method == 'median':
                median = df[col].median()
                df.loc[df[col] < lower_bound, col] = median
                df.loc[df[col] > upper_bound, col] = median

        return df

    def handle_duplicates(self, data):
        """
        处理重复值

        Args:
            data: 原始数据

        Returns:
            处理后的数据
        """
        self.duplicates = data.duplicated().sum()
        return data.drop_duplicates()

    def normalize_data(self, data, columns=None, method='standard'):
        """
        数据标准化

        Args:
            data: 原始数据
            columns: 要标准化的列名列表
            method: 标准化方法 ('standard' | 'minmax' | 'robust')

        Returns:
            标准化后的数据
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

        df = data.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()

        df[columns] = scaler.fit_transform(df[columns])

        return df, scaler

    def preprocess(self, data, handle_missing=True, handle_outliers=True,
                   handle_duplicates=True, normalize=True):
        """
        完整的数据预处理流程

        Args:
            data: 原始数据
            handle_missing: 是否处理缺失值
            handle_outliers: 是否处理异常值
            handle_duplicates: 是否处理重复值
            normalize: 是否标准化

        Returns:
            处理后的数据
        """
        df = data.copy()

        if handle_missing:
            df = self.handle_missing_values(df)

        if handle_duplicates:
            df = self.handle_duplicates(df)

        if handle_outliers:
            df = self.handle_outliers(df)

        if normalize:
            df, scaler = self.normalize_data(df)
            return df, scaler

        return df

    def get_preprocessing_report(self):
        """获取预处理报告"""
        report = {
            'missing_values': self.missing_values,
            'outliers': {k: v['count'] for k, v in self.outliers.items()},
            'duplicates': self.duplicates
        }
        return report


def load_and_preprocess_data(filepath):
    """
    加载并预处理数据

    Args:
        filepath: 数据文件路径

    Returns:
        处理后的数据和预处理器对象
    """
    data = pd.read_csv(filepath)
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess(data)

    return processed_data, preprocessor
