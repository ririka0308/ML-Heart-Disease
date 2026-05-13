"""
特征工程模块
包括年龄特征派生、特征评分、相关性分析、预处理管道等
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ._config import (
    ALL_NUMERIC_FEATURES,
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    CLINICAL_FEATURES,
    DERIVED_NUMERIC_FEATURES,
    EXTENDED_FEATURE_COLUMNS,
    EXTENDED_NUMERIC_FEATURES,
    SCREENING_FEATURES,
    TARGET_COLUMN,
    cached_func,
    encode_target,
)


# ============================================================
# 年龄特征工程（共享函数，消除代码重复）
# ============================================================
def engineer_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    为 DataFrame 添加派生年龄特征。

    根据医学研究，年龄与心脏病风险呈 S 型曲线：
    年轻时增长缓慢，中年加速，老年放缓。
    包括平方项、对数项、分段标签、阈值标记。

    参数:
        df: 输入 DataFrame（必须包含 Age 列才能生效）

    返回:
        添加了派生年龄特征的新 DataFrame
    """
    if "Age" not in df.columns:
        return df

    df = df.copy()

    if "Age_squared" not in df.columns:
        df["Age_squared"] = df["Age"] ** 2
    if "Age_log" not in df.columns:
        df["Age_log"] = np.log(df["Age"] + 1)
    if "Age_group" not in df.columns:
        df["Age_group"] = pd.cut(
            df["Age"],
            bins=[0, 40, 60, 80, 120],
            labels=["young", "middle", "senior", "elderly"],
            include_lowest=True,
        ).astype(str)
    if "Age_over_40" not in df.columns:
        df["Age_over_40"] = (df["Age"] >= 40).astype(int)
    if "Age_over_60" not in df.columns:
        df["Age_over_60"] = (df["Age"] >= 60).astype(int)

    return df


def engineer_age_features_dict(case_data: dict[str, Any]) -> dict[str, Any]:
    """为单病例字典添加派生年龄特征（dict 版，适用于 predict_risk）"""
    if "Age" not in case_data:
        return case_data

    age = case_data["Age"]

    if "Age_squared" not in case_data:
        case_data["Age_squared"] = age ** 2
    if "Age_log" not in case_data:
        case_data["Age_log"] = np.log(age + 1)
    if "Age_group" not in case_data:
        if age < 40:
            case_data["Age_group"] = "young"
        elif age < 60:
            case_data["Age_group"] = "middle"
        elif age < 80:
            case_data["Age_group"] = "senior"
        else:
            case_data["Age_group"] = "elderly"
    if "Age_over_40" not in case_data:
        case_data["Age_over_40"] = 1 if age >= 40 else 0
    if "Age_over_60" not in case_data:
        case_data["Age_over_60"] = 1 if age >= 60 else 0

    return case_data


# ============================================================
# 特征预处理
# ============================================================
def get_all_numeric_features() -> list[str]:
    """获取所有数值特征，包括扩展特征及派生的年龄特征"""
    return ALL_NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    if numeric_features is None:
        numeric_features = get_all_numeric_features()

    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers)


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    feature_names: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(columns)
        else:
            encoder = transformer.named_steps["onehot"]
            feature_names.extend(encoder.get_feature_names_out(columns).tolist())
    return feature_names


def get_features_by_mode(mode: str = "clinical") -> list[str]:
    """根据模式获取特征集"""
    if mode == "screening":
        return SCREENING_FEATURES
    return CLINICAL_FEATURES


# ============================================================
# 特征评分与选择
# ============================================================
def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """从数据框中分离特征和目标变量，并确保派生年龄特征存在"""
    df = df.copy()

    # 确保所有派生年龄特征存在
    df = engineer_age_features(df)

    # 只选择数据集中实际存在的特征列（使用扩展特征集）
    available_features = [col for col in EXTENDED_FEATURE_COLUMNS if col in df.columns]
    X = df[available_features].copy()
    if TARGET_COLUMN not in df.columns:
        return X, None
    y = encode_target(df[TARGET_COLUMN])
    return X, y


@cached_func
def compute_feature_scores(df: pd.DataFrame, top_k: int | None = None) -> pd.DataFrame:
    """计算特征的综合评分（互信息 + 随机森林重要性）"""
    X, y = prepare_xy(df)

    if y is None:
        return pd.DataFrame({"feature": X.columns, "combined_score": [0.0] * len(X.columns)}).sort_values("feature")

    numeric_features = [f for f in X.columns if f not in CATEGORICAL_FEATURES]
    categorical_features = [f for f in X.columns if f in CATEGORICAL_FEATURES]

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    transformed = preprocessor.fit_transform(X)
    transformed_names = get_transformed_feature_names(preprocessor)
    transformed_df = pd.DataFrame(transformed, columns=transformed_names)

    selector = SelectKBest(score_func=mutual_info_classif, k=top_k or transformed_df.shape[1])
    selector.fit(transformed_df, y)

    rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    rf.fit(transformed_df, y)

    scores = pd.DataFrame(
        {
            "feature": transformed_names,
            "mutual_info": selector.scores_,
            "rf_importance": rf.feature_importances_,
        }
    )

    scores["mutual_info_norm"] = (
        (scores["mutual_info"] - scores["mutual_info"].min())
        / (scores["mutual_info"].max() - scores["mutual_info"].min() + 1e-10)
    )
    scores["rf_importance_norm"] = (
        (scores["rf_importance"] - scores["rf_importance"].min())
        / (scores["rf_importance"].max() - scores["rf_importance"].min() + 1e-10)
    )
    scores["combined_score"] = scores["mutual_info_norm"] * 0.5 + scores["rf_importance_norm"] * 0.5
    scores = scores.sort_values("combined_score", ascending=False).reset_index(drop=True)

    return scores


def extract_feature_importance(model_pipeline: Pipeline, reference_df: pd.DataFrame) -> pd.DataFrame:
    """从已训练的模型中提取特征重要性"""
    preprocessor = model_pipeline.named_steps["preprocess"]
    transformed_names = get_transformed_feature_names(preprocessor)
    model = model_pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    result = pd.DataFrame({"feature": transformed_names, "importance": importance})
    return result.sort_values("importance", ascending=False).reset_index(drop=True)


# ============================================================
# 相关性分析
# ============================================================
def build_correlation_frame(df: pd.DataFrame) -> pd.DataFrame:
    """构建特征相关性矩阵"""
    corr_df = df.copy()
    if "Gender" in corr_df.columns:
        corr_df["Gender"] = corr_df["Gender"].map({"Female": 0, "Male": 1}).fillna(0)
    if TARGET_COLUMN in corr_df.columns:
        corr_df["ResultBinary"] = encode_target(corr_df[TARGET_COLUMN])

    # 确保派生年龄特征存在
    corr_df = engineer_age_features(corr_df)

    # 只选择数据集中实际存在的列
    available_columns = []

    for feature_group in [BASE_NUMERIC_FEATURES, EXTENDED_NUMERIC_FEATURES, DERIVED_NUMERIC_FEATURES]:
        for col in feature_group:
            if col in corr_df.columns:
                available_columns.append(col)

    if "ResultBinary" in corr_df.columns:
        available_columns.append("ResultBinary")

    if not available_columns:
        return pd.DataFrame()

    return corr_df[available_columns].corr(numeric_only=True)
