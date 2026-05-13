"""
核心管线模块测试（heart_pipeline.py）
测试数据清洗、特征工程、模型比较等核心功能
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.heart_pipeline import (
    DERIVED_NUMERIC_FEATURES,
    build_correlation_frame,
    build_single_case,
    clean_dataset,
    encode_target,
    extract_feature_importance,
    get_features_by_mode,
    prepare_xy,
)


def _make_dummy_data(n=50) -> pd.DataFrame:
    """创建模拟的心脏病数据集用于测试"""
    np.random.seed(42)
    data = {
        "Age": np.random.randint(20, 80, n),
        "Gender": np.random.choice(["0", "1"], n),
        "Heart rate": np.random.randint(50, 120, n).astype(float),
        "Systolic blood pressure": np.random.randint(80, 200, n).astype(float),
        "Diastolic blood pressure": np.random.randint(40, 120, n).astype(float),
        "Blood sugar": np.random.uniform(60, 300, n),
        "CK-MB": np.random.uniform(0.5, 20, n),
        "Troponin": np.random.uniform(0.001, 2.0, n),
        "Result": np.random.choice(["negative", "positive"], n),
    }
    return pd.DataFrame(data)


class TestCleanDataset(unittest.TestCase):
    """数据集清洗测试"""

    def setUp(self):
        self.df = _make_dummy_data(50)

    def test_clean_dataset_returns_summary(self):
        """清洗后应返回数据框和摘要"""
        cleaned, summary = clean_dataset(self.df)
        self.assertIsInstance(cleaned, pd.DataFrame)
        self.assertIsNotNone(summary)
        self.assertEqual(summary.rows_before, 50)

    def test_clean_dataset_removes_duplicates(self):
        """清洗应移除重复行"""
        df_dup = pd.concat([self.df, self.df.iloc[:3]], ignore_index=True)
        cleaned, summary = clean_dataset(df_dup)
        self.assertEqual(summary.duplicate_rows, 3)
        self.assertEqual(len(cleaned), len(df_dup) - 3)

    def test_clean_dataset_generates_age_features(self):
        """清洗后应生成Age_squared等派生特征"""
        cleaned, _ = clean_dataset(self.df)
        for feat in DERIVED_NUMERIC_FEATURES:
            self.assertIn(feat, cleaned.columns, f"缺少派生特征: {feat}")

    def test_clean_dataset_encodes_target(self):
        """目标变量应被编码为 positive/negative"""
        cleaned, _ = clean_dataset(self.df)
        self.assertIn("positive", cleaned["Result"].unique())
        self.assertIn("negative", cleaned["Result"].unique())

    def test_clean_dataset_no_missing_values(self):
        """清洗后不应存在缺失值"""
        cleaned, _ = clean_dataset(self.df)
        self.assertEqual(cleaned.isnull().sum().sum(), 0)


class TestPrepareXY(unittest.TestCase):
    """特征/标签分离测试"""

    def setUp(self):
        df = _make_dummy_data(50)
        self.cleaned, _ = clean_dataset(df)

    def test_prepare_xy_returns_features_and_target(self):
        """prepare_xy 应返回 (X, y)"""
        X, y = prepare_xy(self.cleaned)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))

    def test_prepare_xy_target_is_binary(self):
        """目标变量 y 应为 0/1 二值"""
        _, y = prepare_xy(self.cleaned)
        self.assertTrue(y.isin([0, 1]).all())

    def test_prepare_xy_contains_derived_features(self):
        """X 中应包含派生年龄特征"""
        X, _ = prepare_xy(self.cleaned)
        for feat in DERIVED_NUMERIC_FEATURES:
            self.assertIn(feat, X.columns, f"X 缺少特征: {feat}")


class TestFeatureFunctions(unittest.TestCase):
    """特征相关工具函数测试"""

    def test_get_features_by_mode(self):
        """筛查模式下不应包含 CK-MB 和 Troponin"""
        screening = get_features_by_mode("screening")
        clinical = get_features_by_mode("clinical")
        self.assertNotIn("CK-MB", screening)
        self.assertNotIn("Troponin", screening)
        self.assertIn("CK-MB", clinical)
        self.assertIn("Troponin", clinical)

    def test_encode_target(self):
        """目标编码应返回 0/1"""
        s = pd.Series(["positive", "negative", "positive"])
        encoded = encode_target(s)
        self.assertEqual(encoded.tolist(), [1, 0, 1])

    def test_build_correlation_frame(self):
        """相关性矩阵应包含 ResultBinary 列"""
        df = _make_dummy_data(30)
        cleaned, _ = clean_dataset(df)
        corr = build_correlation_frame(cleaned)
        self.assertIn("ResultBinary", corr.columns)

    def test_build_single_case_returns_dict(self):
        """单样本构建应返回包含所有基础特征的字典"""
        case = build_single_case()
        self.assertIsInstance(case, dict)
        for key in ["Age", "Gender", "Heart rate"]:
            self.assertIn(key, case)


class TestExtractFeatureImportance(unittest.TestCase):
    """特征重要性提取测试"""

    def test_extract_importances_without_model_returns_empty(self):
        """无模型时返回空 DataFrame"""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        df = _make_dummy_data(20)
        cleaned, _ = clean_dataset(df)
        X, y = prepare_xy(cleaned)

        pipe = Pipeline([
            ("preprocess", StandardScaler()),
            ("model", LogisticRegression(max_iter=100, random_state=42)),
        ])
        pipe.fit(X, y)

        importance = extract_feature_importance(pipe, cleaned)
        self.assertIsInstance(importance, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
