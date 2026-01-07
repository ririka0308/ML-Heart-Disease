"""
特征工程模块
创建新特征、特征选择、特征编码
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE


class FeatureEngineer:
    """特征工程类"""

    def __init__(self):
        self.created_features = []
        self.selected_features = []

    def create_derived_features(self, data):
        """
        创建衍生特征

        Args:
            data: 原始数据

        Returns:
            包含新特征的数据
        """
        df = data.copy()

        # 1. 血压比 (Diastolic/Systolic)
        df['Blood_Pressure_Ratio'] = (
            df['Diastolic blood pressure'] / df['Systolic blood pressure']
        )
        self.created_features.append('Blood_Pressure_Ratio')

        # 2. 脉压差 (Systolic - Diastolic)
        df['Pulse_Pressure'] = (
            df['Systolic blood pressure'] - df['Diastolic blood pressure']
        )
        self.created_features.append('Pulse_Pressure')

        # 3. 年龄分组
        df['Age_Group'] = pd.cut(
            df['Age'],
            bins=[0, 40, 60, 80, 120],
            labels=['青年', '中年', '老年', '高龄']
        )
        self.created_features.append('Age_Group')

        # 4. 心率分组
        df['Heart_Rate_Group'] = pd.cut(
            df['Heart rate'],
            bins=[0, 60, 100, 200],
            labels=['过缓', '正常', '过快']
        )
        self.created_features.append('Heart_Rate_Group')

        # 5. 血糖分类
        df['Blood_Sugar_Category'] = pd.cut(
            df['Blood sugar'],
            bins=[0, 100, 126, 500],
            labels=['正常', '偏高', '糖尿病']
        )
        self.created_features.append('Blood_Sugar_Category')

        # 6. 心脏酶联合得分 (CK-MB + Troponin归一化后的平均)
        ck_mb_rank = df['CK-MB'].rank(pct=True)
        troponin_rank = df['Troponin'].rank(pct=True)
        df['Cardiac_Enzyme_Score'] = (ck_mb_rank + troponin_rank) / 2
        self.created_features.append('Cardiac_Enzyme_Score')

        # 7. BMI近似值 (基于年龄、血压、血糖的相关性)
        df['Risk_Score_Composite'] = (
            (df['Age'].rank(pct=True) * 0.3) +
            (df['CK-MB'].rank(pct=True) * 0.4) +
            (df['Troponin'].rank(pct=True) * 0.3)
        )
        self.created_features.append('Risk_Score_Composite')

        return df

    def encode_categorical_features(self, data):
        """
        编码分类特征

        Args:
            data: 原始数据

        Returns:
            编码后的数据
        """
        df = data.copy()

        # 检测分类列
        categorical_cols = df.select_dtypes(include=['category', 'object']).columns

        # 使用One-Hot编码
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        return df

    def select_features_kbest(self, X, y, k=8):
        """
        使用K-Best方法选择特征

        Args:
            X: 特征数据
            y: 目标变量
            k: 选择的特征数量

        Returns:
            选择后的特征和特征名称
        """
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X, y)

        selected_features = X.columns[selector.get_support()].tolist()
        self.selected_features = selected_features

        return X_selected, selected_features

    def select_features_rfe(self, X, y, model, n_features=8):
        """
        使用RFE方法选择特征

        Args:
            X: 特征数据
            y: 目标变量
            model: 评估模型
            n_features: 选择的特征数量

        Returns:
            选择后的特征和特征名称
        """
        rfe = RFE(model, n_features_to_select=n_features)
        X_selected = rfe.fit_transform(X, y)

        selected_features = X.columns[rfe.support_].tolist()
        self.selected_features = selected_features

        return X_selected, selected_features

    def scale_features(self, data, columns=None, method='standard'):
        """
        特征缩放

        Args:
            data: 原始数据
            columns: 要缩放的列名列表
            method: 缩放方法 ('standard' | 'minmax')

        Returns:
            缩放后的数据和scaler对象
        """
        df = data.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()

        df[columns] = scaler.fit_transform(df[columns])

        return df, scaler

    def engineer_features(self, data, encode=True, scale=True, select_method='all'):
        """
        完整的特征工程流程

        Args:
            data: 原始数据
            encode: 是否编码分类特征
            scale: 是否缩放特征
            select_method: 特征选择方法 ('all' | 'kbest' | 'rfe')

        Returns:
            处理后的数据
        """
        df = data.copy()

        # 创建衍生特征
        df = self.create_derived_features(df)

        # 编码分类特征
        if encode:
            df = self.encode_categorical_features(df)

        # 分离特征和目标
        if 'Result' in df.columns:
            X = df.drop('Result', axis=1)
            y = df['Result'].map({'negative': 0, 'positive': 1})
        else:
            X = df
            y = None

        # 缩放特征
        if scale:
            X, scaler = self.scale_features(X)

        # 特征选择
        if select_method == 'kbest' and y is not None:
            X, selected_features = self.select_features_kbest(X, y)
        elif select_method == 'rfe' and y is not None:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            X, selected_features = self.select_features_rfe(X, y, model)

        if 'Result' in data.columns:
            df['Result'] = data['Result'].map({'negative': 0, 'positive': 1})

        return df, scaler if scale else None

    def get_feature_importance(self, X, y, model):
        """
        获取特征重要性

        Args:
            X: 特征数据
            y: 目标变量
            model: 训练好的模型

        Returns:
            特征重要性DataFrame
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = X.columns

            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)

            return importance_df
        else:
            return None


def create_age_group_feature(data):
    """创建年龄分组特征"""
    bins = [0, 40, 60, 80, 120]
    labels = ['青年', '中年', '老年', '高龄']
    data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels)
    return data


def create_blood_pressure_features(data):
    """创建血压相关特征"""
    # 血压比
    data['Blood_Pressure_Ratio'] = (
        data['Diastolic blood pressure'] / data['Systolic blood pressure']
    )
    # 脉压差
    data['Pulse_Pressure'] = (
        data['Systolic blood pressure'] - data['Diastolic blood pressure']
    )
    return data


def create_cardiac_enzyme_features(data):
    """创建心脏酶相关特征"""
    # CK-MB和Troponin的联合得分
    data['Cardiac_Enzyme_Score'] = (
        data['CK-MB'].rank(pct=True) + data['Troponin'].rank(pct=True)
    ) / 2
    return data
