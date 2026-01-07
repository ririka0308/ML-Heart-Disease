"""
数据预处理模块测试
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import DataPreprocessor


class TestDataPreprocessing(unittest.TestCase):
    """测试数据预处理功能"""

    def setUp(self):
        """设置测试数据"""
        self.preprocessor = DataPreprocessor()
        self.test_data = pd.DataFrame({
            'Age': [25, 30, 35, 40, None],
            'Gender': [0, 1, 0, 1, 0],
            'Heart rate': [70, 80, 90, 75, 85],
            'Systolic blood pressure': [120, 130, 125, 128, 122],
            'Diastolic blood pressure': [80, 85, 82, 84, 81],
            'Blood sugar': [100, 110, 105, 108, 102],
            'CK-MB': [2.0, 2.5, 2.2, 2.4, 2.1],
            'Troponin': [0.05, 0.06, 0.055, 0.058, 0.052],
            'Result': ['negative', 'negative', 'negative', 'negative', 'negative']
        })

    def test_handle_missing_values(self):
        """测试缺失值处理"""
        processed = self.preprocessor.handle_missing_values(self.test_data)
        self.assertFalse(processed['Age'].isnull().any())
        self.assertEqual(processed['Age'].isnull().sum(), 0)

    def test_detect_outliers(self):
        """测试异常值检测"""
        outliers = self.preprocessor.detect_outliers(self.test_data)
        self.assertIn('Age', outliers)
        self.assertIn('Heart rate', outliers)

    def test_handle_duplicates(self):
        """测试重复值处理"""
        test_data = pd.concat([self.test_data, self.test_data.iloc[:1]], ignore_index=True)
        processed = self.preprocessor.handle_duplicates(test_data)
        self.assertEqual(len(processed), len(test_data) - 1)

    def test_normalize_data(self):
        """测试数据标准化"""
        processed, scaler = self.preprocessor.normalize_data(self.test_data)
        self.assertIsNotNone(scaler)


if __name__ == '__main__':
    unittest.main()
