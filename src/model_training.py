"""
模型训练模块
包含多种机器学习模型的训练接口
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
import joblib
import os
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer


class ModelTrainer:
    """模型训练类"""

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.model_results = {}

    def get_all_models(self):
        """
        获取所有可用模型

        Returns:
            模型字典
        """
        models = {
            '逻辑回归': LogisticRegression(
                random_state=42, max_iter=1000, class_weight='balanced'
            ),
            '决策树': DecisionTreeClassifier(
                random_state=42, class_weight='balanced'
            ),
            '随机森林': RandomForestClassifier(
                n_estimators=200, random_state=42, class_weight='balanced'
            ),
            '梯度提升': GradientBoostingClassifier(
                random_state=42
            ),
            '支持向量机': SVC(
                probability=True, random_state=42, class_weight='balanced'
            ),
            'K近邻': KNeighborsClassifier(),
            '朴素贝叶斯': GaussianNB(),
            '神经网络': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42
            )
        }

        return models

    def train_single_model(self, model, X_train, y_train, model_name):
        """
        训练单个模型

        Args:
            model: 模型对象
            X_train: 训练特征
            y_train: 训练标签
            model_name: 模型名称

        Returns:
            训练好的模型
        """
        model.fit(X_train, y_train)
        self.models[model_name] = model
        return model

    def train_all_models(self, X_train, y_train):
        """
        训练所有模型

        Args:
            X_train: 训练特征
            y_train: 训练标签

        Returns:
            训练好的模型字典
        """
        models = self.get_all_models()

        for name, model in models.items():
            print(f"正在训练 {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
            print(f"{name} 训练完成!")

        return self.models

    def evaluate_model(self, model, X_test, y_test):
        """
        评估单个模型

        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签

        Returns:
            评估指标字典
        """
        # 预测
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # 计算指标
        metrics = {
            '准确率': accuracy_score(y_test, y_pred),
            '精确率': precision_score(y_test, y_pred),
            '召回率': recall_score(y_test, y_pred),
            'F1分数': f1_score(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_prob)
        }

        return metrics

    def evaluate_all_models(self, X_test, y_test):
        """
        评估所有模型

        Args:
            X_test: 测试特征
            y_test: 测试标签

        Returns:
            所有模型的评估结果DataFrame
        """
        results = {}

        for name, model in self.models.items():
            metrics = self.evaluate_model(model, X_test, y_test)
            results[name] = metrics
            self.model_results[name] = metrics

        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values('AUC-ROC', ascending=False)

        return results_df

    def get_best_model(self, metric='AUC-ROC'):
        """
        获取最佳模型

        Args:
            metric: 评估指标

        Returns:
            最佳模型和模型名称
        """
        if not self.model_results:
            return None, None

        best_model_name = max(
            self.model_results.keys(),
            key=lambda k: self.model_results[k][metric]
        )
        best_model = self.models[best_model_name]

        self.best_model = best_model

        return best_model, best_model_name

    def hyperparameter_tuning(self, model, param_grid, X_train, y_train,
                             cv=5, scoring='roc_auc'):
        """
        超参数调优

        Args:
            model: 模型对象
            param_grid: 参数网格
            X_train: 训练特征
            y_train: 训练标签
            cv: 交叉验证折数
            scoring: 评分指标

        Returns:
            最佳模型
        """
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        return grid_search.best_estimator_, grid_search.best_params_

    def explain_model_shap(self, model, X_train, X_test, feature_names):
        """
        使用SHAP解释模型

        Args:
            model: 训练好的模型
            X_train: 训练特征
            X_test: 测试特征
            feature_names: 特征名称列表

        Returns:
            SHAP解释器和SHAP值
        """
        # 创建SHAP解释器
        if hasattr(model, 'predict_proba'):
            # 树模型和集成模型
            if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
                explainer = shap.TreeExplainer(model)
            # 线性模型
            elif isinstance(model, LogisticRegression):
                explainer = shap.LinearExplainer(model, X_train)
            # 其他模型使用KernelExplainer
            else:
                explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
        else:
            return None, None

        # 计算SHAP值
        shap_values = explainer.shap_values(X_test)

        return explainer, shap_values

    def explain_model_lime(self, model, X_train, X_test, feature_names, class_names=['无心脏病', '有心脏病']):
        """
        使用LIME解释模型

        Args:
            model: 训练好的模型
            X_train: 训练特征
            X_test: 测试特征
            feature_names: 特征名称列表
            class_names: 类别名称列表

        Returns:
            LIME解释器
        """
        # 创建LIME解释器
        explainer = LimeTabularExplainer(
            X_train.values if hasattr(X_train, 'values') else X_train,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification'
        )

        return explainer

    def create_stacking_model(self, base_models, meta_model, X_train, y_train):
        """
        创建并训练Stacking集成模型

        Args:
            base_models: 基础模型列表
            meta_model: 元模型
            X_train: 训练特征
            y_train: 训练标签

        Returns:
            训练好的Stacking模型
        """
        # 创建Stacking分类器
        stacking_clf = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,  # 交叉验证折数
            stack_method='predict_proba',  # 使用概率作为元特征
            n_jobs=-1
        )

        # 训练模型
        stacking_clf.fit(X_train, y_train)

        return stacking_clf

    def perform_feature_selection(self, X, y, method='SelectKBest', k=5, model=None):
        """
        执行特征选择

        Args:
            X: 特征数据
            y: 目标变量
            method: 特征选择方法 (SelectKBest, RFE, SelectFromModel)
            k: 选择的特征数量
            model: 用于RFE或SelectFromModel的模型

        Returns:
            选择后的特征和特征选择器
        """
        if method == 'SelectKBest':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'RFE':
            if model is None:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator=model, n_features_to_select=k, step=1)
        elif method == 'SelectFromModel':
            if model is None:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = SelectFromModel(estimator=model, max_features=k)
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")

        # 拟合特征选择器
        X_selected = selector.fit_transform(X, y)

        return X_selected, selector

    def create_polynomial_features(self, X, degree=2, interaction_only=False):
        """
        创建多项式特征和交互特征

        Args:
            X: 特征数据
            degree: 多项式次数
            interaction_only: 是否只保留交互特征

        Returns:
            转换后的特征和多项式转换器
        """
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        X_poly = poly.fit_transform(X)
        return X_poly, poly

    def perform_pca(self, X, n_components=2):
        """
        执行主成分分析降维

        Args:
            X: 特征数据
            n_components: 保留的主成分数量

        Returns:
            降维后的特征和PCA转换器
        """
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        return X_pca, pca

    def scale_features(self, X_train, X_test=None):
        """
        标准化特征

        Args:
            X_train: 训练特征
            X_test: 测试特征（可选）

        Returns:
            标准化后的特征和缩放器
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, scaler
        
        return X_train_scaled, scaler

    def engineer_features(self, X, y, feature_selection=False, polynomial_features=False,
                         pca=False, scaling=False, feature_selection_k=5, polynomial_degree=2):
        """
        完整的特征工程流程

        Args:
            X: 特征数据
            y: 目标变量
            feature_selection: 是否执行特征选择
            polynomial_features: 是否创建多项式特征
            pca: 是否执行PCA降维
            scaling: 是否标准化特征
            feature_selection_k: 特征选择保留的特征数量
            polynomial_degree: 多项式特征的次数

        Returns:
            处理后的特征和相关转换器
        """
        X_processed = X.copy()
        transformers = {}

        # 标准化特征
        if scaling:
            if y is not None:
                # 划分训练集和测试集进行缩放
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y, test_size=0.2, random_state=42, stratify=y
                )
                X_train_scaled, X_test_scaled, scaler = self.scale_features(X_train, X_test)
                transformers['scaler'] = scaler
                return {
                    'X_train': X_train_scaled,
                    'X_test': X_test_scaled,
                    'y_train': y_train,
                    'y_test': y_test,
                    'transformers': transformers
                }
            else:
                X_scaled, scaler = self.scale_features(X_processed)
                X_processed = X_scaled
                transformers['scaler'] = scaler

        # 创建多项式特征
        if polynomial_features:
            X_poly, poly = self.create_polynomial_features(X_processed, degree=polynomial_degree)
            X_processed = X_poly
            transformers['polynomial'] = poly

        # 特征选择
        if feature_selection and y is not None:
            X_selected, selector = self.perform_feature_selection(X_processed, y, k=feature_selection_k)
            X_processed = X_selected
            transformers['feature_selector'] = selector

        # PCA降维
        if pca:
            X_pca, pca_transformer = self.perform_pca(X_processed, n_components=min(feature_selection_k, 5))
            X_processed = X_pca
            transformers['pca'] = pca_transformer

        return {
            'X_processed': X_processed,
            'transformers': transformers
        }

    def save_model(self, model, filepath):
        """
        保存模型

        Args:
            model: 模型对象
            filepath: 保存路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        print(f"模型已保存到 {filepath}")

    def load_model(self, filepath):
        """
        加载模型

        Args:
            filepath: 模型路径

        Returns:
            加载的模型
        """
        model = joblib.load(filepath)
        print(f"模型已从 {filepath} 加载")
        return model

    def train_and_evaluate(self, X, y, test_size=0.2, random_state=42):
        """
        完整的训练和评估流程

        Args:
            X: 特征数据
            y: 目标变量
            test_size: 测试集比例
            random_state: 随机种子

        Returns:
            训练好的模型和评估结果
        """
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 训练所有模型
        models = self.train_all_models(X_train, y_train)

        # 评估所有模型
        results_df = self.evaluate_all_models(X_test, y_test)

        # 获取最佳模型
        best_model, best_model_name = self.get_best_model()

        return {
            'models': models,
            'results': results_df,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }


def train_random_forest(X_train, y_train, n_estimators=200):
    """训练随机森林模型"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train):
    """训练逻辑回归模型"""
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    """训练梯度提升模型"""
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model
