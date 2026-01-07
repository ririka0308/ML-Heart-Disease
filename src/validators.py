# -*- coding: utf-8 -*-
"""
输入验证模块
提供完善的数据校验和异常处理
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import re


@dataclass
class ValidationResult:
    """验证结果类"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class InputValidator:
    """输入验证器类"""
    
    # 医学正常值参考范围
    NORMAL_RANGES = {
        'age': (1, 120, '年龄'),
        'heart_rate': (40, 200, '心率'),
        'systolic_bp': (60, 250, '收缩压'),
        'diastolic_bp': (40, 150, '舒张压'),
        'blood_sugar': (2.0, 30.0, '血糖'),
        'ck_mb': (0.0, 500.0, 'CK-MB'),
        'troponin': (0.0, 20.0, '肌钙蛋白')
    }
    
    # 临床警告阈值
    WARNING_THRESHOLDS = {
        'age': {'high': 70, 'msg': '高龄患者心脏病风险增加'},
        'heart_rate': {'low': 60, 'high': 100, 'msg': '心率异常'},
        'systolic_bp': {'high': 140, 'msg': '收缩压偏高，可能存在高血压'},
        'diastolic_bp': {'high': 90, 'msg': '舒张压偏高，可能存在高血压'},
        'blood_sugar': {'high': 7.0, 'msg': '血糖偏高，需关注糖代谢'},
        'ck_mb': {'high': 5.0, 'msg': 'CK-MB升高，可能存在心肌损伤'},
        'troponin': {'high': 0.04, 'msg': '肌钙蛋白升高，需排除急性心肌损伤'}
    }
    
    @classmethod
    def validate_patient_data(cls, gender: int, age: float, heart_rate: float,
                              systolic_bp: float, diastolic_bp: float,
                              blood_sugar: float, ck_mb: float, 
                              troponin: float) -> ValidationResult:
        """
        验证患者输入数据
        
        Returns:
            ValidationResult: 包含验证结果、错误和警告信息
        """
        errors = []
        warnings = []
        
        # 验证性别
        if gender not in [0, 1]:
            errors.append("性别输入无效，请选择男性或女性")
        
        # 验证年龄
        if not cls._validate_range('age', age, errors):
            pass
        elif age > 70:
            warnings.append(f"提示: {cls.WARNING_THRESHOLDS['age']['msg']}")
        
        # 验证心率
        if cls._validate_range('heart_rate', heart_rate, errors):
            if heart_rate < 60 or heart_rate > 100:
                warnings.append(f"提示: {cls.WARNING_THRESHOLDS['heart_rate']['msg']} (当前值: {heart_rate})")
        
        # 验证血压
        if cls._validate_range('systolic_bp', systolic_bp, errors):
            if systolic_bp > 140:
                warnings.append(f"提示: {cls.WARNING_THRESHOLDS['systolic_bp']['msg']} (当前值: {systolic_bp} mmHg)")
        
        if cls._validate_range('diastolic_bp', diastolic_bp, errors):
            if diastolic_bp > 90:
                warnings.append(f"提示: {cls.WARNING_THRESHOLDS['diastolic_bp']['msg']} (当前值: {diastolic_bp} mmHg)")
        
        # 验证血压逻辑
        if systolic_bp <= diastolic_bp:
            errors.append(f"血压数据异常: 收缩压({systolic_bp})应大于舒张压({diastolic_bp})")
        
        # 验证血糖
        if cls._validate_range('blood_sugar', blood_sugar, errors):
            if blood_sugar > 7.0:
                warnings.append(f"提示: {cls.WARNING_THRESHOLDS['blood_sugar']['msg']} (当前值: {blood_sugar} mmol/L)")
        
        # 验证CK-MB
        if cls._validate_range('ck_mb', ck_mb, errors):
            if ck_mb > 5.0:
                warnings.append(f"提示: {cls.WARNING_THRESHOLDS['ck_mb']['msg']} (当前值: {ck_mb} ng/mL)")
        
        # 验证肌钙蛋白
        if cls._validate_range('troponin', troponin, errors):
            if troponin > 0.04:
                warnings.append(f"提示: {cls.WARNING_THRESHOLDS['troponin']['msg']} (当前值: {troponin} ng/mL)")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    @classmethod
    def _validate_range(cls, field: str, value: float, errors: List[str]) -> bool:
        """验证数值是否在有效范围内"""
        if field not in cls.NORMAL_RANGES:
            return True
        
        min_val, max_val, label = cls.NORMAL_RANGES[field]
        
        if value is None:
            errors.append(f"{label}不能为空")
            return False
        
        if value < min_val or value > max_val:
            errors.append(f"{label}超出有效范围 ({min_val}-{max_val})，当前值: {value}")
            return False
        
        return True
    
    @staticmethod
    def validate_username(username: str) -> Tuple[bool, str]:
        """
        验证用户名
        
        Returns:
            (是否有效, 错误信息)
        """
        if not username:
            return False, "用户名不能为空"
        
        if len(username) < 3:
            return False, "用户名长度至少3个字符"
        
        if len(username) > 20:
            return False, "用户名长度不能超过20个字符"
        
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            return False, "用户名只能包含字母、数字和下划线"
        
        return True, ""
    
    @staticmethod
    def validate_password(password: str) -> Tuple[bool, str]:
        """
        验证密码强度
        
        Returns:
            (是否有效, 错误信息)
        """
        if not password:
            return False, "密码不能为空"
        
        if len(password) < 6:
            return False, "密码长度至少6个字符"
        
        if len(password) > 50:
            return False, "密码长度不能超过50个字符"
        
        return True, ""
    
    @staticmethod
    def get_clinical_interpretation(prediction: int, probability: float,
                                    troponin: float, ck_mb: float,
                                    age: int, systolic_bp: float) -> List[str]:
        """
        生成临床解读
        
        Returns:
            临床解读列表
        """
        interpretations = []
        
        if prediction == 1:
            interpretations.append("模型预测该患者存在心脏病发病风险")
            
            if probability > 0.8:
                interpretations.append(f"预测概率较高 ({probability:.1%})，建议立即进行进一步检查")
            elif probability > 0.6:
                interpretations.append(f"预测概率中等 ({probability:.1%})，建议密切关注并进行随访")
        else:
            interpretations.append("模型预测该患者心脏病发病风险较低")
            
            if probability > 0.3:
                interpretations.append(f"虽然预测为阴性，但概率值 ({probability:.1%}) 提示仍需关注")
        
        # 具体指标解读
        if troponin > 0.04:
            interpretations.append(f"肌钙蛋白升高 ({troponin} ng/mL)，提示可能存在心肌损伤")
        
        if ck_mb > 5.0:
            interpretations.append(f"CK-MB升高 ({ck_mb} ng/mL)，需进一步排除心肌梗死")
        
        if age > 65 and systolic_bp > 140:
            interpretations.append("高龄合并高血压，心血管风险增加")
        
        return interpretations


class DataValidator:
    """数据集验证器"""
    
    REQUIRED_COLUMNS = ['Age', 'Gender', 'Heart rate', 'Systolic blood pressure',
                       'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 
                       'Troponin', 'Result']
    
    @classmethod
    def validate_dataset(cls, df) -> ValidationResult:
        """
        验证数据集格式
        
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        # 检查必需列
        missing_cols = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            errors.append(f"数据集缺少必需列: {', '.join(missing_cols)}")
        
        # 检查数据量
        if len(df) < 10:
            errors.append("数据集样本数量过少，至少需要10条记录")
        elif len(df) < 100:
            warnings.append(f"数据集样本较少 ({len(df)}条)，可能影响模型训练效果")
        
        # 检查缺失值
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_ratio > 0.3:
            errors.append(f"数据集缺失值比例过高 ({missing_ratio:.1%})，请先进行数据清洗")
        elif missing_ratio > 0.1:
            warnings.append(f"数据集存在 {missing_ratio:.1%} 的缺失值，建议进行处理")
        
        # 检查目标变量分布
        if 'Result' in df.columns:
            value_counts = df['Result'].value_counts(normalize=True)
            if len(value_counts) != 2:
                errors.append("目标变量应为二分类")
            else:
                min_ratio = value_counts.min()
                if min_ratio < 0.1:
                    warnings.append(f"目标变量分布不平衡 (少数类占比: {min_ratio:.1%})，建议采用过采样等方法")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
