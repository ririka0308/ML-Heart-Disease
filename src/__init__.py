"""
心脏病预测系统 - 源代码包

包含数据预处理、特征工程、模型训练、评估、可视化、数据库和验证模块
"""

__version__ = '2.0.0'
__author__ = '毕业设计'

from .data_preprocessing import DataPreprocessor, load_and_preprocess_data
from .feature_engineering import FeatureEngineer
from .model_training import ModelTrainer
from .model_evaluation import ModelEvaluator
from .visualization import Visualizer
from .prediction import HeartDiseasePredictor
from .database import DatabaseManager, get_database
from .validators import InputValidator, DataValidator, ValidationResult

__all__ = [
    'DataPreprocessor',
    'load_and_preprocess_data',
    'FeatureEngineer',
    'ModelTrainer',
    'ModelEvaluator',
    'Visualizer',
    'HeartDiseasePredictor',
    'DatabaseManager',
    'get_database',
    'InputValidator',
    'DataValidator',
    'ValidationResult'
]
