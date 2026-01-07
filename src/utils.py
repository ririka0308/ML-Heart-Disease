# -*- coding: utf-8 -*-
"""
工具函数模块
包含报告生成、数据处理等通用工具函数
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import json
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MedicalReference


def make_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """
    将DataFrame转换为Arrow兼容的格式
    解决Streamlit中的PyArrow序列化错误
    """
    df = df.copy()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].astype(str)
        elif hasattr(df[column].dtype, 'numpy_dtype'):
            df[column] = df[column].to_numpy()
        elif pd.api.types.is_integer_dtype(df[column]):
            df[column] = df[column].astype('float64')
        elif pd.api.types.is_float_dtype(df[column]):
            df[column] = df[column].astype('float64')
    return df


def generate_report_html(patient_data: Dict, prediction_result: Dict,
                        recommendations: List[str], 
                        similar_cases: pd.DataFrame = None) -> str:
    """
    生成HTML格式的预测报告
    
    Args:
        patient_data: 患者数据
        prediction_result: 预测结果
        recommendations: 健康建议列表
        similar_cases: 相似病例DataFrame
        
    Returns:
        HTML字符串
    """
    risk_level = prediction_result.get('risk_level', '未知')
    probability = prediction_result.get('probability', 0)
    risk_color = "#e74c3c" if risk_level == "高风险" else "#2ecc71" if risk_level == "低风险" else "#f39c12"
    
    # 解释说明
    interpretations = prediction_result.get('interpretation', [])
    shap_explanation = prediction_result.get('shap_explanation', '')
    
    # 相似病例HTML
    similar_cases_html = ""
    if similar_cases is not None and not similar_cases.empty:
        similar_cases_html = """
        <div class="section">
            <h2>相似病例参考</h2>
            <table>
                <tr>
                    <th>年龄</th>
                    <th>性别</th>
                    <th>CK-MB</th>
                    <th>肌钙蛋白</th>
                    <th>诊断结果</th>
                </tr>
        """
        for _, case in similar_cases.iterrows():
            result_text = "阳性" if case.get('predict_result', 0) == 1 else "阴性"
            gender_text = "男" if case.get('gender', 0) == 1 else "女"
            similar_cases_html += f"""
                <tr>
                    <td>{case.get('age', 'N/A')}</td>
                    <td>{gender_text}</td>
                    <td>{case.get('ck_mb', 'N/A'):.2f}</td>
                    <td>{case.get('troponin', 'N/A'):.3f}</td>
                    <td>{result_text}</td>
                </tr>
            """
        similar_cases_html += "</table></div>"
    
    html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>心脏病风险预测报告</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Microsoft YaHei', 'SimHei', Arial, sans-serif;
                line-height: 1.8;
                color: #333;
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f7fa;
            }}
            .header {{
                background: linear-gradient(135deg, #2c3e50, #3498db);
                color: white;
                padding: 30px;
                text-align: center;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .header h1 {{
                font-size: 28px;
                margin-bottom: 10px;
            }}
            .header p {{
                opacity: 0.9;
            }}
            .content {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .section {{
                margin-bottom: 30px;
            }}
            h2 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            .risk-level {{
                text-align: center;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                background-color: {risk_color};
                color: white;
            }}
            .risk-level .level {{
                font-size: 32px;
                font-weight: bold;
            }}
            .risk-level .prob {{
                font-size: 18px;
                opacity: 0.9;
            }}
            .patient-info {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }}
            .info-item {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
            }}
            .info-label {{
                color: #666;
                font-size: 0.9em;
            }}
            .info-value {{
                font-size: 1.1em;
                font-weight: bold;
                color: #2c3e50;
            }}
            .interpretation {{
                background: #e8f4fd;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 10px 0;
            }}
            .recommendations ul {{
                padding-left: 20px;
            }}
            .recommendations li {{
                margin-bottom: 10px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 10px 0;
            }}
            th, td {{
                padding: 10px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            .footer {{
                text-align: center;
                color: #666;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
            }}
            @media print {{
                body {{
                    background: white;
                }}
                .no-print {{
                    display: none;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>心脏病风险预测报告</h1>
            <p>基于机器学习的临床决策支持</p>
            <p>生成时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>1. 患者基本信息</h2>
                <div class="patient-info">
                    <div class="info-item">
                        <div class="info-label">患者姓名</div>
                        <div class="info-value">{patient_data.get('patient_name', '未填写')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">性别</div>
                        <div class="info-value">{'男' if patient_data.get('gender', 0) == 1 else '女'}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">年龄</div>
                        <div class="info-value">{patient_data.get('age', 'N/A')} 岁</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">心率</div>
                        <div class="info-value">{patient_data.get('heart_rate', 'N/A')} 次/分钟</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">血压</div>
                        <div class="info-value">{patient_data.get('systolic_bp', 'N/A')}/{patient_data.get('diastolic_bp', 'N/A')} mmHg</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">血糖</div>
                        <div class="info-value">{patient_data.get('blood_sugar', 'N/A')} mmol/L</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">CK-MB</div>
                        <div class="info-value">{patient_data.get('ck_mb', 'N/A')} ng/mL</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">肌钙蛋白</div>
                        <div class="info-value">{patient_data.get('troponin', 'N/A')} ng/mL</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>2. 预测结果</h2>
                <div class="risk-level">
                    <div class="level">{risk_level}</div>
                    <div class="prob">心脏病发病概率：{probability:.1%}</div>
                </div>
            </div>
            
            <div class="section">
                <h2>3. 分析解读</h2>
                {''.join([f'<div class="interpretation">{i}</div>' for i in interpretations])}
                
                <h3 style="margin-top: 20px;">SHAP特征贡献分析</h3>
                <p>{shap_explanation}</p>
            </div>
            
            {similar_cases_html}
            
            <div class="section recommendations">
                <h2>4. 健康建议</h2>
                <ul>
                    {''.join([f'<li>{r}</li>' for r in recommendations])}
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p>本报告由心脏病数据分析与预测系统自动生成</p>
            <p>仅供参考，不替代专业医生的诊断和建议</p>
        </div>
    </body>
    </html>
    """
    
    return html


def format_number(value, decimals: int = 2) -> str:
    """格式化数字显示"""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def get_risk_color(risk_level: str) -> str:
    """根据风险等级获取颜色"""
    colors = {
        "高风险": "#e74c3c",
        "中风险": "#f39c12",
        "低风险": "#2ecc71"
    }
    return colors.get(risk_level, "#6c757d")


def calculate_bmi(weight: float, height: float) -> float:
    """计算BMI指数"""
    if height <= 0:
        return 0
    height_m = height / 100  # 转换为米
    return weight / (height_m ** 2)


def get_bmi_category(bmi: float) -> str:
    """获取BMI分类"""
    if bmi < 18.5:
        return "偏瘦"
    elif bmi < 24:
        return "正常"
    elif bmi < 28:
        return "超重"
    else:
        return "肥胖"


def export_to_excel(df: pd.DataFrame, filename: str) -> bytes:
    """导出DataFrame为Excel文件"""
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='数据')
    return output.getvalue()


def export_to_csv(df: pd.DataFrame) -> bytes:
    """导出DataFrame为CSV文件"""
    return df.to_csv(index=False).encode('utf-8-sig')


def format_datetime(dt) -> str:
    """格式化日期时间"""
    if dt is None:
        return "N/A"
    if isinstance(dt, str):
        return dt
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def truncate_string(s: str, max_length: int = 50) -> str:
    """截断字符串"""
    if len(s) <= max_length:
        return s
    return s[:max_length-3] + "..."


class DataProcessor:
    """数据处理器"""
    
    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """预处理数据"""
        df = df.copy()
        
        # 处理缺失值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # 处理分类变量
        if 'Result' in df.columns and df['Result'].dtype == 'object':
            df['Result'] = df['Result'].map({'negative': 0, 'positive': 1})
        
        return df
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, column: str, 
                       method: str = 'iqr') -> pd.Series:
        """
        检测异常值
        
        Args:
            df: 数据框
            column: 列名
            method: 方法 ('iqr' 或 'zscore')
            
        Returns:
            布尔序列，True表示是异常值
        """
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            return (df[column] < lower) | (df[column] > upper)
        
        elif method == 'zscore':
            mean = df[column].mean()
            std = df[column].std()
            z_scores = np.abs((df[column] - mean) / std)
            return z_scores > 3
        
        return pd.Series([False] * len(df))
    
    @staticmethod
    def get_statistics(df: pd.DataFrame) -> Dict:
        """获取数据统计信息"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        return {
            'total_records': len(df),
            'total_features': len(df.columns),
            'numeric_features': len(numeric_df.columns),
            'missing_values': df.isnull().sum().sum(),
            'missing_ratio': df.isnull().sum().sum() / (len(df) * len(df.columns)),
            'duplicate_rows': df.duplicated().sum()
        }
