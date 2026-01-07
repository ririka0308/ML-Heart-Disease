# -*- coding: utf-8 -*-
"""
预测逻辑模块
封装预测逻辑、SHAP值计算、相似病例检索
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
import sys
import logging
import json

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ModelConfig, PathConfig, MedicalReference
from database import get_db

logger = logging.getLogger(__name__)


class HeartDiseasePredictor:
    """心脏病预测器"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = ModelConfig.FEATURE_COLUMNS
        self.db = get_db()
        self._load_model()
    
    def _load_model(self):
        """加载模型和标准化器"""
        try:
            if ModelConfig.DEFAULT_MODEL_PATH.exists():
                self.model = joblib.load(ModelConfig.DEFAULT_MODEL_PATH)
                logger.info("已加载保存的模型")
            
            if ModelConfig.SCALER_PATH.exists():
                self.scaler = joblib.load(ModelConfig.SCALER_PATH)
                logger.info("已加载标准化器")
            
        except Exception as e:
            logger.warning(f"模型加载失败: {e}")
    
    def _ensure_model(self, X: pd.DataFrame = None, y: pd.Series = None):
        """确保模型已加载，如果没有则训练默认模型"""
        if self.model is None:
            logger.info("训练默认随机森林模型")
            self.model = RandomForestClassifier(
                **ModelConfig.DEFAULT_PARAMS['RandomForest']
            )
            if X is not None and y is not None:
                # 标准化
                self.scaler = StandardScaler()
                X_scaled = X.copy()
                X_scaled[ModelConfig.CONTINUOUS_FEATURES] = self.scaler.fit_transform(
                    X[ModelConfig.CONTINUOUS_FEATURES]
                )
                self.model.fit(X_scaled, y)
                
                # 保存模型
                self.save_model()
    
    def save_model(self):
        """保存模型和标准化器"""
        try:
            PathConfig.SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            
            if self.model:
                joblib.dump(self.model, ModelConfig.DEFAULT_MODEL_PATH)
                logger.info(f"模型已保存: {ModelConfig.DEFAULT_MODEL_PATH}")
            
            if self.scaler:
                joblib.dump(self.scaler, ModelConfig.SCALER_PATH)
                logger.info(f"标准化器已保存: {ModelConfig.SCALER_PATH}")
                
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
    
    def predict(self, patient_data: Dict) -> Dict:
        """
        执行预测
        
        Args:
            patient_data: 患者数据字典，包含所有特征
            
        Returns:
            预测结果字典
        """
        # 准备输入数据
        input_df = pd.DataFrame([{
            'Gender': patient_data.get('gender', 0),
            'Age': patient_data.get('age', 0),
            'CK-MB': patient_data.get('ck_mb', 0),
            'Troponin': patient_data.get('troponin', 0)
        }])
        
        # 标准化
        if self.scaler:
            input_scaled = input_df.copy()
            input_scaled[ModelConfig.CONTINUOUS_FEATURES] = self.scaler.transform(
                input_df[ModelConfig.CONTINUOUS_FEATURES]
            )
        else:
            input_scaled = input_df
        
        # 预测
        prediction = self.model.predict(input_scaled)[0]
        probability = self.model.predict_proba(input_scaled)[0][1]
        
        # 风险等级
        risk_level = self._get_risk_level(probability)
        
        # 计算SHAP值
        shap_values, shap_explanation = self._calculate_shap(input_scaled)
        
        # 生成临床解读
        interpretation = self._generate_interpretation(
            prediction, probability, patient_data, shap_values
        )
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': risk_level,
            'shap_values': shap_values,
            'shap_explanation': shap_explanation,
            'interpretation': interpretation,
            'input_data': patient_data
        }
    
    def _get_risk_level(self, probability: float) -> str:
        """根据概率获取风险等级"""
        if probability >= 0.7:
            return "高风险"
        elif probability >= 0.4:
            return "中风险"
        else:
            return "低风险"
    
    def _calculate_shap(self, input_data: pd.DataFrame) -> Tuple[Optional[np.ndarray], str]:
        """
        计算SHAP值
        
        Returns:
            (shap_values数组, 文字解释)
        """
        try:
            import shap
            
            # 创建解释器
            if hasattr(self.model, 'estimators_'):
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(input_data)
                
                # 对于二分类，取阳性类的SHAP值
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1][0]
                else:
                    shap_vals = shap_values[0]
                
                # 生成解释文字
                explanation = self._generate_shap_explanation(shap_vals)
                
                return shap_vals.tolist(), explanation
                
        except ImportError:
            logger.warning("SHAP库未安装")
        except Exception as e:
            logger.error(f"SHAP计算失败: {e}")
        
        return None, "SHAP分析不可用"
    
    def _generate_shap_explanation(self, shap_values: np.ndarray) -> str:
        """生成SHAP解释文字"""
        feature_names = self.features
        feature_names_cn = [MedicalReference.FEATURE_NAMES_CN.get(f, f) for f in feature_names]
        
        # 按绝对值排序
        sorted_idx = np.argsort(np.abs(shap_values))[::-1]
        
        explanations = []
        for idx in sorted_idx:
            name = feature_names_cn[idx]
            value = shap_values[idx]
            
            if value > 0.1:
                explanations.append(f"{name}显著增加了患病风险（贡献度: +{value:.2f}）")
            elif value < -0.1:
                explanations.append(f"{name}降低了患病风险（贡献度: {value:.2f}）")
        
        if explanations:
            return "；".join(explanations[:3]) + "。"
        else:
            return "各特征对预测结果影响均衡。"
    
    def _generate_interpretation(self, prediction: int, probability: float,
                                 patient_data: Dict, shap_values: Optional[List]) -> List[str]:
        """生成临床解读"""
        interpretations = []
        
        # 基本预测结果
        if prediction == 1:
            interpretations.append(f"模型预测该患者存在心脏病发病风险（概率: {probability:.1%}）")
            
            if probability > 0.8:
                interpretations.append("风险概率很高，建议立即进行详细心脏检查")
            elif probability > 0.6:
                interpretations.append("风险概率较高，建议进行进一步检查并密切观察")
        else:
            interpretations.append(f"模型预测该患者心脏病发病风险较低（概率: {probability:.1%}）")
            
            if probability > 0.3:
                interpretations.append("虽然预测为阴性，但仍有一定风险，建议定期随访")
        
        # 具体指标分析
        troponin = patient_data.get('troponin', 0)
        ck_mb = patient_data.get('ck_mb', 0)
        age = patient_data.get('age', 0)
        
        ref = MedicalReference.NORMAL_RANGES
        
        if troponin > ref['Troponin']['warning_high']:
            interpretations.append(
                f"肌钙蛋白水平升高（{troponin} ng/mL > {ref['Troponin']['warning_high']} ng/mL），"
                "提示可能存在急性心肌损伤"
            )
        
        if ck_mb > ref['CK-MB']['warning_high']:
            interpretations.append(
                f"CK-MB水平升高（{ck_mb} ng/mL > {ref['CK-MB']['warning_high']} ng/mL），"
                "需进一步排除心肌梗死"
            )
        
        if age > 65:
            interpretations.append("患者属于高龄人群，心血管疾病风险本身较高")
        
        return interpretations
    
    def get_similar_cases(self, patient_data: Dict, k: int = 3) -> pd.DataFrame:
        """
        检索相似病例
        
        Args:
            patient_data: 当前患者数据
            k: 返回的相似病例数量
            
        Returns:
            相似病例DataFrame
        """
        return self.db.search_similar_cases(patient_data, k)
    
    def get_health_recommendations(self, prediction: int, patient_data: Dict) -> List[str]:
        """
        根据预测结果生成健康建议
        """
        recommendations = []
        
        if prediction == 1:  # 高风险
            recommendations = [
                "建议立即就医，进行心脏专科检查（心电图、心脏超声等）",
                "密切监测心脏相关生物标志物（肌钙蛋白、CK-MB等）",
                "保持情绪稳定，避免剧烈运动和情绪激动",
                "严格控制危险因素：血压、血糖、血脂",
                "遵医嘱可能需要进行药物治疗"
            ]
        else:  # 低风险
            recommendations = [
                "保持健康的生活方式",
                "定期进行体检，关注心脏健康",
                "合理饮食：低盐、低脂、多蔬果",
                "适量运动：每周至少150分钟中等强度有氧运动",
                "戒烟限酒，保持充足睡眠"
            ]
        
        # 根据具体指标添加个性化建议
        if patient_data.get('systolic_bp', 0) > 140:
            recommendations.append("血压偏高，建议定期监测并考虑降压治疗")
        
        if patient_data.get('blood_sugar', 0) > 7.0:
            recommendations.append("血糖偏高，建议进行糖耐量检测并控制饮食")
        
        return recommendations
    
    @property
    def feature_importance(self) -> Optional[Dict]:
        """获取特征重要性"""
        if self.model and hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return dict(zip(self.features, importance))
        return None


class InputValidator:
    """输入验证器"""
    
    @staticmethod
    def validate_patient_data(patient_data: Dict) -> Tuple[bool, List[str], List[str]]:
        """
        验证患者输入数据
        
        Returns:
            (是否有效, 错误列表, 警告列表)
        """
        errors = []
        warnings = []
        
        ref = MedicalReference.NORMAL_RANGES
        
        # 验证年龄
        age = patient_data.get('age', 0)
        if age < 1 or age > 120:
            errors.append(f"年龄超出有效范围（1-120岁），当前值: {age}")
        elif age > 70:
            warnings.append("高龄患者心脏病风险增加")
        
        # 验证心率
        hr = patient_data.get('heart_rate', 0)
        if hr < 30 or hr > 250:
            errors.append(f"心率超出有效范围，当前值: {hr}")
        elif hr < 60 or hr > 100:
            warnings.append(f"心率异常（正常范围60-100），当前值: {hr}")
        
        # 验证血压
        sbp = patient_data.get('systolic_bp', 0)
        dbp = patient_data.get('diastolic_bp', 0)
        
        if sbp < 60 or sbp > 300:
            errors.append(f"收缩压超出有效范围，当前值: {sbp}")
        elif sbp > 140:
            warnings.append(f"收缩压偏高，当前值: {sbp} mmHg")
        
        if dbp < 30 or dbp > 200:
            errors.append(f"舒张压超出有效范围，当前值: {dbp}")
        elif dbp > 90:
            warnings.append(f"舒张压偏高，当前值: {dbp} mmHg")
        
        if sbp <= dbp:
            errors.append(f"血压数据异常：收缩压({sbp})应大于舒张压({dbp})")
        
        # 验证血糖
        bs = patient_data.get('blood_sugar', 0)
        if bs < 1 or bs > 50:
            errors.append(f"血糖超出有效范围，当前值: {bs}")
        elif bs > 7.0:
            warnings.append(f"血糖偏高，当前值: {bs} mmol/L")
        
        # 验证CK-MB
        ck = patient_data.get('ck_mb', 0)
        if ck < 0 or ck > 1000:
            errors.append(f"CK-MB超出有效范围，当前值: {ck}")
        elif ck > 5:
            warnings.append(f"CK-MB升高，当前值: {ck} ng/mL")
        
        # 验证肌钙蛋白
        tn = patient_data.get('troponin', 0)
        if tn < 0 or tn > 100:
            errors.append(f"肌钙蛋白超出有效范围，当前值: {tn}")
        elif tn > 0.04:
            warnings.append(f"肌钙蛋白升高，当前值: {tn} ng/mL")
        
        return len(errors) == 0, errors, warnings


# 全局预测器实例
predictor = HeartDiseasePredictor()
