"""
配置与常量模块
包含特征列表、列名映射、医学权重、参数范围等所有配置常量
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.config import AppConfig

try:
    from streamlit import cache_data
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


def cached_func(func):
    """条件缓存装饰器：如果 Streamlit 可用则使用缓存"""
    if HAS_STREAMLIT:
        return cache_data(func)
    return func


TARGET_COLUMN = AppConfig.TARGET_COLUMN


BASE_NUMERIC_FEATURES = [
    "Age", "Heart rate", "Systolic blood pressure",
    "Diastolic blood pressure", "Blood sugar", "CK-MB", "Troponin",
]
EXTENDED_NUMERIC_FEATURES = [
    "Cholesterol", "Max HR", "ST depression", "Chest pain type",
]
DERIVED_NUMERIC_FEATURES = ["Age_squared", "Age_log", "Age_over_40", "Age_over_60"]
CATEGORICAL_FEATURES = ["Gender", "Age_group"]
EXTENDED_CATEGORICAL_FEATURES = [
    "Exercise angina", "Number of vessels", "Thallium scan",
]
ALL_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + EXTENDED_NUMERIC_FEATURES
ALL_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + EXTENDED_CATEGORICAL_FEATURES
FEATURE_COLUMNS = CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES
EXTENDED_FEATURE_COLUMNS = ALL_CATEGORICAL_FEATURES + ALL_NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES
NUMERIC_FEATURES = ALL_NUMERIC_FEATURES


MEDICAL_PRIOR_WEIGHTS: dict[str, int] = {
    "Troponin": 10, "CK-MB": 9, "Age": 7, "Blood sugar": 6,
    "Cholesterol": 6, "Systolic blood pressure": 5, "Heart rate": 4,
    "Max HR": 4, "ST depression": 4, "Chest pain type": 4,
    "Exercise angina": 4, "Gender": 3, "Diastolic blood pressure": 3,
    "Number of vessels": 3, "Thallium scan": 3,
    "Age_squared": 7, "Age_log": 7, "Age_over_40": 7, "Age_over_60": 7, "Age_group": 7,
}


COLUMN_ALIASES: dict[str, list[str]] = {
    "Age": ["年龄", "age", "AGE", "年纪", "years"],
    "Gender": ["性别", "Sex", "sex", "SEX", "gender"],
    "Heart rate": ["心率", "heart_rate", "HR", "hr", "pulse", "Pulse", "heartrate"],
    "Systolic blood pressure": [
        "收缩压", "SBP", "sbp", "systolic", "Systolic", "systolic bp",
        "blood pressure", "BP", "bp",
    ],
    "Diastolic blood pressure": ["舒张压", "DBP", "dbp", "diastolic", "Diastolic"],
    "Blood sugar": ["血糖", "blood_sugar", "glucose", "Glucose", "FBS", "fbs"],
    "CK-MB": ["ck_mb", "CKMB", "ckmb", "ck"],
    "Troponin": ["肌钙蛋白", "troponin", "trop", "Trop"],
    "Cholesterol": ["胆固醇", "chol", "Chol", "CHOL"],
    "Max HR": ["最大心率", "max_hr", "maxhr", "MaxHR", "maximum heart rate"],
    "ST depression": ["ST段压低", "st_depression", "oldpeak", "Oldpeak",
                       "ST depression induced by exercise relative to rest"],
    "Chest pain type": ["胸痛类型", "chest_pain", "cp", "Cp"],
    "Exercise angina": ["运动心绞痛", "exang", "Exang", "exercise_angina",
                         "exercise induced angina"],
    "Number of vessels": ["血管数", "ca", "Ca", "num_vessels", "number of major vessels",
                          "Number of vessels fluro"],
    "Thallium scan": ["铊扫描", "thal", "Thal", "thallium"],
    "Result": ["诊断结果", "diagnosis", "Diagnosis", "target", "Target",
               "label", "Label", "result"],
}

RESULT_VALUE_MAP: dict[str, str] = {
    "positive": "positive", "pos": "positive", "1": "positive", "yes": "positive",
    "negative": "negative", "neg": "negative", "0": "negative", "no": "negative",
}


SCREENING_FEATURES = [
    "Age", "Age_squared", "Age_log", "Age_group", "Age_over_40", "Age_over_60",
    "Gender", "Heart rate", "Systolic blood pressure", "Diastolic blood pressure",
    "Blood sugar", "Cholesterol", "Max HR", "ST depression", "Chest pain type",
    "Exercise angina", "Number of vessels", "Thallium scan",
]

CLINICAL_FEATURES = [
    "Age", "Age_squared", "Age_log", "Age_group", "Age_over_40", "Age_over_60",
    "Gender", "Heart rate", "Systolic blood pressure", "Diastolic blood pressure",
    "Blood sugar", "CK-MB", "Troponin",
    "Cholesterol", "Max HR", "ST depression", "Chest pain type",
    "Exercise angina", "Number of vessels", "Thallium scan",
]

MODE_LABELS: dict[str, str] = {
    "clinical": "临床辅助模型（专业版）",
    "screening": "基础预测模型（筛查版）",
}
MODE_HINTS: dict[str, str] = {
    "clinical": "包含所有指标（Troponin 和 CK-MB），用于医学辅助诊断场景。",
    "screening": "剔除 Troponin 和 CK-MB，仅使用常规指标，适用于居家/体检筛查场景。",
}

CLINICAL_RANGES: dict[str, tuple[float, float]] = {
    "Age": (1, 120),
    "Heart rate": (20, 220),
    "Systolic blood pressure": (60, 250),
    "Diastolic blood pressure": (30, 180),
    "Blood sugar": (20, 600),
    "CK-MB": (0, 300),
    "Troponin": (0, 12),
    "Cholesterol": (50, 600),
    "Max HR": (50, 250),
    "ST depression": (0, 10),
    "Chest pain type": (1, 4),
    "Exercise angina": (0, 1),
    "Number of vessels": (0, 4),
    "Thallium scan": (3, 7),
}

# 风险等级对应颜色（集中定义，保持多页面一致）
RISK_COLORS: dict[str, str] = {
    "高风险": "#ef4444",
    "中风险": "#f59e0b",
    "低风险": "#10b981",
}


@dataclass
class DatasetSummary:
    rows_before: int
    rows_after: int
    duplicate_rows: int
    missing_before: dict[str, int]
    missing_after: dict[str, int]
    clipped_cells: int
    positive_rate: float


def encode_target(series: pd.Series) -> pd.Series:
    """将 positive/negative 标签编码为 1/0"""
    return series.eq(AppConfig.POSITIVE_LABEL).astype(int)
