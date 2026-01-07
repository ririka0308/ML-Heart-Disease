"""
预测模块
包含单样本和批量预测功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class HeartDiseasePredictor:
    """心脏病预测器"""

    def __init__(self, model, scaler, feature_names):
        """
        初始化预测器

        Args:
            model: 训练好的模型
            scaler: 标准化器
            feature_names: 特征名称列表
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names

    def predict_single(self, patient_data: Dict) -> Dict:
        """
        单样本预测

        Args:
            patient_data: 患者数据字典

        Returns:
            预测结果字典
        """
        # 转换为DataFrame
        input_df = pd.DataFrame([patient_data])

        # 确保特征顺序正确
        input_df = input_df[self.feature_names]

        # 标准化
        numeric_cols = input_df.select_dtypes(include=[np.number]).columns
        input_df[numeric_cols] = self.scaler.transform(input_df[numeric_cols])

        # 预测
        prediction = self.model.predict(input_df)[0]
        probability = self.model.predict_proba(input_df)[0]

        # 结果
        result = {
            'prediction': '阳性' if prediction == 1 else '阴性',
            'positive_probability': probability[1],
            'negative_probability': probability[0],
            'risk_level': self._get_risk_level(probability[1]),
            'recommendations': self._get_recommendations(prediction, probability[1])
        }

        return result

    def predict_batch(self, batch_data: pd.DataFrame) -> pd.DataFrame:
        """
        批量预测

        Args:
            batch_data: 批量数据DataFrame

        Returns:
            包含预测结果的DataFrame
        """
        # 确保特征顺序正确
        input_df = batch_data[self.feature_names].copy()

        # 标准化
        numeric_cols = input_df.select_dtypes(include=[np.number]).columns
        input_df[numeric_cols] = self.scaler.transform(input_df[numeric_cols])

        # 预测
        predictions = self.model.predict(input_df)
        probabilities = self.model.predict_proba(input_df)

        # 添加结果
        result_df = batch_data.copy()
        result_df['Prediction'] = ['Positive' if p == 1 else 'Negative' for p in predictions]
        result_df['Positive_Probability'] = probabilities[:, 1]
        result_df['Negative_Probability'] = probabilities[:, 0]
        result_df['Risk_Level'] = result_df['Positive_Probability'].apply(self._get_risk_level)

        return result_df

    def _get_risk_level(self, probability: float) -> str:
        """
        根据概率获取风险等级

        Args:
            probability: 阳性概率

        Returns:
            风险等级字符串
        """
        if probability < 0.2:
            return '低风险'
        elif probability < 0.5:
            return '中低风险'
        elif probability < 0.7:
            return '中等风险'
        elif probability < 0.85:
            return '中高风险'
        else:
            return '高风险'

    def _get_recommendations(self, prediction: int, probability: float) -> List[str]:
        """
        根据预测结果获取建议

        Args:
            prediction: 预测结果 (0或1)
            probability: 阳性概率

        Returns:
            建议列表
        """
        if prediction == 1:
            recommendations = [
                "🏥 立即就医: 进行全面心脏检查",
                "🚭 戒烟限酒: 避免不良生活习惯",
                "💪 控制三高: 监测并控制血压、血糖、血脂",
                "🏃 适量运动: 在医生指导下进行适度运动",
                "🥗 健康饮食: 低盐低脂,多蔬果",
                "😴 充足睡眠: 保证每日7-8小时睡眠",
                "😌 心态平和: 避免过度焦虑和压力",
                f"⚠️ 阳性概率: {probability:.1%},请重视!"
            ]
        else:
            recommendations = [
                "✨ 继续保持: 保持现有健康生活习惯",
                "📅 定期体检: 每年进行心脏健康检查",
                "🥗 均衡饮食: 保持营养均衡",
                "🏃 规律运动: 每周3-5次有氧运动",
                "💧 充足饮水: 每日饮水1.5-2L",
                "😴 优质睡眠: 保证充足睡眠质量",
                "😌 心理健康: 保持积极乐观的心态"
            ]

        return recommendations

    def get_feature_importance(self, patient_data: Dict) -> pd.DataFrame:
        """
        获取单个样本的特征贡献度

        Args:
            patient_data: 患者数据字典

        Returns:
            特征重要性DataFrame
        """
        if not hasattr(self.model, 'feature_importances_'):
            return None

        input_df = pd.DataFrame([patient_data])
        input_df = input_df[self.feature_names]

        numeric_cols = input_df.select_dtypes(include=[np.number]).columns
        input_df[numeric_cols] = self.scaler.transform(input_df[numeric_cols])

        # 获取特征重要性
        importance = self.model.feature_importances_

        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        return importance_df


def create_patient_data(age, gender, heart_rate, sbp, dbp, blood_sugar, ck_mb, troponin):
    """
    创建患者数据字典

    Args:
        age: 年龄
        gender: 性别 (0: 女性, 1: 男性)
        heart_rate: 心率
        sbp: 收缩压
        dbp: 舒张压
        blood_sugar: 血糖
        ck_mb: 肌酸激酶同工酶
        troponin: 肌钙蛋白

    Returns:
        患者数据字典
    """
    return {
        'Age': age,
        'Gender': gender,
        'Heart rate': heart_rate,
        'Systolic blood pressure': sbp,
        'Diastolic blood pressure': dbp,
        'Blood sugar': blood_sugar,
        'CK-MB': ck_mb,
        'Troponin': troponin
    }


def validate_patient_data(patient_data: Dict) -> bool:
    """
    验证患者数据的有效性

    Args:
        patient_data: 患者数据字典

    Returns:
        是否有效
    """
    required_fields = [
        'Age', 'Gender', 'Heart rate',
        'Systolic blood pressure', 'Diastolic blood pressure',
        'Blood sugar', 'CK-MB', 'Troponin'
    ]

    # 检查必填字段
    for field in required_fields:
        if field not in patient_data:
            return False
        if patient_data[field] is None or patient_data[field] == '':
            return False

    # 检查数值范围
    if patient_data['Age'] < 0 or patient_data['Age'] > 120:
        return False
    if patient_data['Gender'] not in [0, 1]:
        return False
    if patient_data['Heart rate'] < 0 or patient_data['Heart rate'] > 300:
        return False

    return True
