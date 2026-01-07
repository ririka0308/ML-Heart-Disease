# -*- coding: utf-8 -*-
"""
全局配置文件
集中管理所有系统配置，包括路径、数据库、日志等
"""

import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent.absolute()


class Config:
    """项目配置类（保留原有配置）"""

    # 项目基本信息
    PROJECT_NAME = "基于机器学习的心脏病数据分析与预测系统"
    VERSION = "2.0.0"
    AUTHOR = "毕业设计"

    # 数据路径
    BASE_DIR = str(ROOT_DIR)
    DATA_DIR = str(ROOT_DIR / 'data')
    MODEL_DIR = str(ROOT_DIR / 'models')
    NOTEBOOKS_DIR = str(ROOT_DIR / 'notebooks')

    # 数据文件路径
    DATA_FILE = str(ROOT_DIR / 'data' / 'Medicaldataset.csv')

    # 模型保存路径
    MODEL_SAVE_PATH = str(ROOT_DIR / 'models' / 'best_model.pkl')
    SCALER_SAVE_PATH = str(ROOT_DIR / 'models' / 'scaler.pkl')
    FEATURES_SAVE_PATH = str(ROOT_DIR / 'models' / 'features.pkl')

    # 机器学习参数
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5

    # 模型参数
    RANDOM_FOREST_N_ESTIMATORS = 200
    LOGISTIC_REGRESSION_MAX_ITER = 1000

    # 特征参数
    TARGET_COLUMN = 'Result'
    NUMERIC_COLUMNS = [
        'Age', 'Heart rate', 'Systolic blood pressure',
        'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin'
    ]
    CATEGORICAL_COLUMNS = ['Gender']

    # 可视化参数
    FIGURE_SIZE = (12, 8)
    DPI = 100
    COLOR_PALETTE = 'viridis'

    # Streamlit配置
    PAGE_TITLE = "心脏病数据分析与预测系统"
    PAGE_ICON = ""
    LAYOUT = "wide"

    # 风险阈值
    RISK_THRESHOLDS = {
        '低风险': 0.2,
        '中低风险': 0.5,
        '中等风险': 0.7,
        '中高风险': 0.85,
        '高风险': 1.0
    }

    # 可用模型列表
    AVAILABLE_MODELS = [
        '逻辑回归',
        '决策树',
        '随机森林',
        '梯度提升',
        '支持向量机',
        'K近邻',
        '朴素贝叶斯'
    ]

    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        dirs = [cls.DATA_DIR, cls.MODEL_DIR, cls.NOTEBOOKS_DIR]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)


# ==================== 路径配置 ====================
class PathConfig:
    """路径配置"""
    # 数据目录
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    # 默认数据文件
    DEFAULT_DATA_FILE = DATA_DIR / "Medicaldataset.csv"
    
    # 数据库目录和文件
    DATABASE_DIR = ROOT_DIR / "database"
    DATABASE_FILE = DATABASE_DIR / "heart_system.db"
    
    # 模型目录
    MODEL_DIR = ROOT_DIR / "models"
    MODELS_DIR = ROOT_DIR / "models"
    SAVED_MODELS_DIR = MODELS_DIR / "saved"
    
    # 资源目录
    ASSETS_DIR = ROOT_DIR / "assets"
    
    # 日志目录
    LOGS_DIR = ROOT_DIR / "logs"
    LOG_FILE = LOGS_DIR / "system.log"
    
    @classmethod
    def ensure_dirs(cls):
        """确保所有必需的目录存在"""
        dirs = [
            cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR,
            cls.DATABASE_DIR, cls.MODELS_DIR, cls.SAVED_MODELS_DIR,
            cls.ASSETS_DIR, cls.LOGS_DIR
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# ==================== 数据库配置 ====================
class DatabaseConfig:
    """数据库配置"""
    # SQLite数据库路径
    DB_PATH = str(PathConfig.DATABASE_FILE)
    
    # 默认管理员账户
    DEFAULT_ADMIN = {
        'username': 'admin',
        'password': 'admin123',
        'role': 'admin',
        'real_name': '系统管理员',
        'department': '信息科'
    }
    
    # 默认医生账户（用于演示）
    DEFAULT_DOCTOR = {
        'username': 'doctor',
        'password': 'doctor123',
        'role': 'doctor',
        'real_name': '张医生',
        'department': '心内科'
    }


# ==================== 模型配置 ====================
class ModelConfig:
    """模型配置"""
    # 默认模型路径
    DEFAULT_MODEL_PATH = PathConfig.SAVED_MODELS_DIR / "best_model.pkl"
    SCALER_PATH = PathConfig.SAVED_MODELS_DIR / "scaler.pkl"
    FEATURES_PATH = PathConfig.SAVED_MODELS_DIR / "features.pkl"
    
    # 使用的特征
    FEATURE_COLUMNS = ['Gender', 'Age', 'CK-MB', 'Troponin']
    
    # 连续型特征（需要标准化）
    CONTINUOUS_FEATURES = ['Age', 'CK-MB', 'Troponin']
    
    # 目标变量
    TARGET_COLUMN = 'Result'
    
    # 模型超参数默认值
    DEFAULT_PARAMS = {
        'RandomForest': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        },
        'GradientBoosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42
        },
        'LogisticRegression': {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42
        },
        'SVM': {
            'C': 1.0,
            'kernel': 'rbf',
            'probability': True,
            'random_state': 42
        },
        'MLP': {
            'hidden_layer_sizes': (100,),
            'max_iter': 500,
            'random_state': 42
        }
    }
    
    # 交叉验证折数
    CV_FOLDS = 5
    
    # 测试集比例
    TEST_SIZE = 0.2


# ==================== UI配置 ====================
class UIConfig:
    """界面配置"""
    # 页面标题
    PAGE_TITLE = "心脏病数据分析与预测系统"
    PAGE_ICON = ""
    LAYOUT = "wide"
    
    # 主题颜色
    PRIMARY_COLOR = "#3498db"
    SUCCESS_COLOR = "#2ecc71"
    WARNING_COLOR = "#f39c12"
    DANGER_COLOR = "#e74c3c"
    
    # 侧边栏配置
    SIDEBAR_WIDTH = 300
    
    # 导航选项
    NAV_OPTIONS_GUEST = ["用户登录", "系统介绍"]
    NAV_OPTIONS_DOCTOR = [
        "全景概览", 
        "智能诊断", 
        "模型实验室", 
        "历史病历"
    ]
    NAV_OPTIONS_ADMIN = [
        "全景概览", 
        "智能诊断", 
        "模型实验室", 
        "历史病历",
        "用户管理",
        "系统日志"
    ]


# ==================== 日志配置 ====================
class LogConfig:
    """日志配置"""
    # 日志级别
    LEVEL = "INFO"
    
    # 日志格式
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 日志文件
    FILE = str(PathConfig.LOG_FILE)
    
    # 最大文件大小（MB）
    MAX_SIZE = 10
    
    # 备份文件数量
    BACKUP_COUNT = 5


# ==================== 医学参考值 ====================
class MedicalReference:
    """医学参考值配置"""
    NORMAL_RANGES = {
        'Age': {'min': 1, 'max': 120, 'unit': '岁'},
        'Heart rate': {'min': 60, 'max': 100, 'unit': '次/分钟', 'warning_low': 60, 'warning_high': 100},
        'Systolic blood pressure': {'min': 90, 'max': 140, 'unit': 'mmHg', 'warning_high': 140},
        'Diastolic blood pressure': {'min': 60, 'max': 90, 'unit': 'mmHg', 'warning_high': 90},
        'Blood sugar': {'min': 3.9, 'max': 6.1, 'unit': 'mmol/L', 'warning_high': 7.0},
        'CK-MB': {'min': 0, 'max': 5, 'unit': 'ng/mL', 'warning_high': 5.0},
        'Troponin': {'min': 0, 'max': 0.04, 'unit': 'ng/mL', 'warning_high': 0.04}
    }
    
    # 特征中文名称映射
    FEATURE_NAMES_CN = {
        'Gender': '性别',
        'Age': '年龄',
        'Heart rate': '心率',
        'Systolic blood pressure': '收缩压',
        'Diastolic blood pressure': '舒张压',
        'Blood sugar': '血糖',
        'CK-MB': '肜酸激酶同工酶',
        'Troponin': '肌钙蛋白',
        'Result': '诊断结果'
    }
    
    # 特征医学意义
    FEATURE_MEANINGS = {
        'Gender': '患者性别，男性心脏病风险通常较高',
        'Age': '患者年龄，心脏病风险随年龄增加',
        'Heart rate': '心率，正常范围60-100次/分钟',
        'Systolic blood pressure': '收缩压，正常范围<140mmHg',
        'Diastolic blood pressure': '舒张压，正常范围<90mmHg',
        'Blood sugar': '血糖水平，空腹正常范围3.9-6.1mmol/L',
        'CK-MB': '肜酸激酶同工酶，心肌损伤标志物',
        'Troponin': '肌钙蛋白，心肌损伤的金标准指标'
    }


# 初始化时确保目录存在
PathConfig.ensure_dirs()
Config.create_dirs()
