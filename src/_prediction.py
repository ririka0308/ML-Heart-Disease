"""
风险预测模块
包括单病例构建、数据校验、风险概率预测、业务逻辑约束
"""

from __future__ import annotations

import pandas as pd

from ._config import (
    CLINICAL_RANGES,
    EXTENDED_FEATURE_COLUMNS,
)
from ._data import load_dataset
from ._features import (
    engineer_age_features_dict,
    get_features_by_mode,
)


def build_single_case(defaults=None):
    """构建单病例输入的默认值字典"""
    if defaults is None:
        defaults = load_dataset()[EXTENDED_FEATURE_COLUMNS].median(numeric_only=True)
    return {
        "Gender": "Male",
        "Age": int(defaults.get("Age", 58)),
        "Heart rate": int(defaults.get("Heart rate", 75)),
        "Systolic blood pressure": int(defaults.get("Systolic blood pressure", 125)),
        "Diastolic blood pressure": int(defaults.get("Diastolic blood pressure", 72)),
        "Blood sugar": float(defaults.get("Blood sugar", 120.0)),
        "CK-MB": float(defaults.get("CK-MB", 2.8)),
        "Troponin": float(defaults.get("Troponin", 0.02)),
    }


def validate_and_clamp_value(key, value):
    """验证并截断极端值到合理范围"""
    if key not in CLINICAL_RANGES:
        return value, False, ""
    min_val, max_val = CLINICAL_RANGES[key]
    if value < min_val:
        return min_val, True, f"{key} 值 {value} 低于合理范围 [{min_val}, {max_val}]，已修正为 {min_val}"
    if value > max_val:
        return max_val, True, f"{key} 值 {value} 超出合理范围 [{min_val}, {max_val}]，已修正为 {max_val}"
    return value, False, ""


def predict_risk(model, case_data, mode="clinical"):
    """执行个体风险预测"""
    case_data = case_data.copy()

    warnings = []
    for key, value in list(case_data.items()):
        if isinstance(value, (int, float)):
            corrected_value, was_clamped, message = validate_and_clamp_value(key, float(value))
            if was_clamped:
                case_data[key] = corrected_value
                warnings.append(message)

    case_data = engineer_age_features_dict(case_data)

    expected_features = []
    if hasattr(model, "feature_names_in_"):
        expected_features = model.feature_names_in_.tolist()
    else:
        expected_features = get_features_by_mode(mode)

    for feature in expected_features:
        if feature not in case_data:
            if feature in ("CK-MB", "Troponin"):
                case_data[feature] = 0.0
            elif feature in ("Age_squared", "Age_log", "Age_group", "Age_over_40", "Age_over_60"):
                pass
            else:
                case_data[feature] = 0.0

    df = pd.DataFrame([case_data])[expected_features]
    probability = float(model.predict_proba(df)[0, 1])

    if probability >= 0.7:
        label = "高风险"
        pred = 1
    elif probability >= 0.4:
        label = "中风险"
        pred = 0
    else:
        label = "低风险"
        pred = 0

    return {
        "prediction": "positive" if pred == 1 else "negative",
        "probability": probability,
        "risk_level": label,
        "warnings": warnings,
        "corrected_case": case_data,
        "mode": mode,
    }
