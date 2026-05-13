"""
PDF报告生成器
用于生成心脏病风险预测报告的PDF文件
"""

import datetime
import os
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 注册中文字体
FONT_REGISTRED = False
CHINESE_FONT_NAME = "Helvetica"

def register_chinese_font():
    """注册中文字体，返回 (字体名称, 是否找到中文字体)"""
    global FONT_REGISTRED, CHINESE_FONT_NAME

    if FONT_REGISTRED:
        return CHINESE_FONT_NAME, True

    font_found = False
    # 尝试注册中文字体
    font_paths = [
        # Windows 字体路径
        "C:/Windows/Fonts/simhei.ttf",      # 黑体
        "C:/Windows/Fonts/simsun.ttc",       # 宋体
        "C:/Windows/Fonts/msyh.ttc",         # 微软雅黑
        "C:/Windows/Fonts/simkai.ttf",       # 楷体
        # Linux 字体路径
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",  # macOS
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                CHINESE_FONT_NAME = 'ChineseFont'
                FONT_REGISTRED = True
                font_found = True
                break
            except:
                continue

    if not font_found:
        import warnings as _w
        _w.warn("未找到中文字体，PDF 中的中文可能显示为乱码。"
                "建议安装文泉驿微米黑 (wqy-microhei) 或 Windows 字体。")

    return CHINESE_FONT_NAME, font_found

def get_font_name():
    """获取当前可用的字体名称"""
    return register_chinese_font()[0]

# 医学建议值参考范围
MEDICAL_REFERENCE_RANGES = {
    'Age': {
        'name': '[Age]',
        'unit': 'years',
        'optimal': '18-65',
        'warning': '>65',
        'critical': '>80'
    },
    'Heart rate': {
        'name': '[Heart rate]',
        'unit': 'bpm',
        'optimal': '60-100',
        'warning': '50-60 or 100-120',
        'critical': '<50 or >120'
    },
    'Systolic blood pressure': {
        'name': '[Systolic BP]',
        'unit': 'mmHg',
        'optimal': '<120',
        'warning': '120-139',
        'critical': '>=140'
    },
    'Diastolic blood pressure': {
        'name': '[Diastolic BP]',
        'unit': 'mmHg',
        'optimal': '<80',
        'warning': '80-89',
        'critical': '>=90'
    },
    'Blood sugar': {
        'name': '[Blood sugar]',
        'unit': 'mg/dL',
        'optimal': '<100',
        'warning': '100-125',
        'critical': '>=126'
    },
    'CK-MB': {
        'name': '[CK-MB]',
        'unit': 'ng/mL',
        'optimal': '<2.5',
        'warning': '2.5-6.0',
        'critical': '>=6.0'
    },
    'Troponin': {
        'name': '[Troponin]',
        'unit': 'ng/mL',
        'optimal': '<0.04',
        'warning': '0.04-0.4',
        'critical': '>=0.4'
    },
    'Gender': {
        'name': '[Gender]',
        'unit': '',
        'optimal': 'M/F',
        'warning': '-',
        'critical': '-'
    }
}


def create_pdf_report_for_prediction(input_data, prediction_result, mode="clinical"):
    """
    为预测结果创建PDF报告

    Args:
        input_data: 输入的患者数据
        prediction_result: 预测结果字典
        mode: 预测模式 ("clinical" 或 "screening")

    Returns:
        bytes: PDF文件的字节流
    """
    # 生化指标列表（筛查模式下不显示）
    bio_chemical_markers = ['CK-MB', 'Troponin'] if mode == "screening" else []
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # 获取中文字体
    font_name = get_font_name()
    
    # 自定义样式
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName=font_name,
        fontSize=18,
        spaceAfter=30,
        alignment=1,  # 居中
        textColor=colors.HexColor('#1e3a5f')
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=12,
        spaceAfter=20,
        alignment=1,  # 居中
        textColor=colors.HexColor('#475569')
    )
    
    section_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontName=font_name,
        fontSize=14,
        spaceAfter=12,
        textColor=colors.HexColor('#1e40af')
    )
    
    content_style = ParagraphStyle(
        'Content',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10,
        spaceAfter=6
    )
    
    # 构建文档内容
    elements = []

    # 标题
    elements.append(Paragraph("Heart Disease Risk Assessment Report", title_style))
    elements.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    elements.append(Spacer(1, 20))

    # 预测结果摘要
    elements.append(Paragraph("=== Prediction Summary ===", section_style))
    risk_color = colors.HexColor('#ef4444') if prediction_result.get('risk_level') == 'High Risk' else \
                 colors.HexColor('#f59e0b') if prediction_result.get('risk_level') == 'Medium Risk' else \
                 colors.HexColor('#10b981')

    summary_data = [
        ["Diagnosis", prediction_result.get('prediction', 'N/A')],
        ["Probability", f"{prediction_result.get('probability', 0)*100:.2f}%"],
        ["Risk Level", prediction_result.get('risk_level', 'N/A')]
    ]

    summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('TEXTCOLOR', (1, 2), (1, 2), risk_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    # 输入参数与参考值对照
    elements.append(Paragraph("=== Input Parameters ===", section_style))

    # 创建带参考值的表格
    feature_data = [['Parameter', 'Value', 'Unit', 'Reference Range', 'Status']]

    for key, value in input_data.items():
        if key in MEDICAL_REFERENCE_RANGES:
            # 筛查模式下跳过生化指标
            if key in bio_chemical_markers:
                continue
            ref = MEDICAL_REFERENCE_RANGES[key]
            param_name = ref['name']
            unit = ref['unit']

            # 判断状态
            status = 'Normal'
            status_color = colors.HexColor('#10b981')

            try:
                val = float(value)
                # 根据参考值判断状态
                if key == 'Systolic blood pressure':
                    if val >= 140:
                        status = 'High'
                        status_color = colors.HexColor('#ef4444')
                    elif val >= 120:
                        status = 'Warning'
                        status_color = colors.HexColor('#f59e0b')
                elif key == 'Diastolic blood pressure':
                    if val >= 90:
                        status = 'High'
                        status_color = colors.HexColor('#ef4444')
                    elif val >= 80:
                        status = 'Warning'
                        status_color = colors.HexColor('#f59e0b')
                elif key == 'Blood sugar':
                    if val >= 126:
                        status = 'High'
                        status_color = colors.HexColor('#ef4444')
                    elif val >= 100:
                        status = 'Warning'
                        status_color = colors.HexColor('#f59e0b')
                elif key == 'CK-MB':
                    if val >= 6.0:
                        status = 'High'
                        status_color = colors.HexColor('#ef4444')
                    elif val >= 2.5:
                        status = 'Warning'
                        status_color = colors.HexColor('#f59e0b')
                elif key == 'Troponin':
                    if val >= 0.4:
                        status = 'High'
                        status_color = colors.HexColor('#ef4444')
                    elif val >= 0.04:
                        status = 'Warning'
                        status_color = colors.HexColor('#f59e0b')
                elif key == 'Heart rate':
                    if val < 50 or val > 120:
                        status = 'High'
                        status_color = colors.HexColor('#ef4444')
                    elif val < 60 or val > 100:
                        status = 'Warning'
                        status_color = colors.HexColor('#f59e0b')
            except:
                pass

            feature_data.append([param_name, str(value), unit, ref['optimal'], status])

    feature_table = Table(feature_data, colWidths=[1.5*inch, 0.8*inch, 0.6*inch, 1.2*inch, 0.7*inch])
    feature_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), font_name),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('FONTNAME', (0, 1), (-1, -1), font_name),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    elements.append(feature_table)
    elements.append(Spacer(1, 20))

    # 参考值说明
    elements.append(Paragraph("=== Reference Guide ===", section_style))
    ref_explanation = [
        ['Status', 'Meaning', 'Description'],
        ['Normal', 'Optimal range', 'Value within healthy range'],
        ['Warning', 'Caution', 'Value approaching risk threshold'],
        ['High', 'Risk', 'Value exceeds safe range, consult doctor']
    ]
    ref_table = Table(ref_explanation, colWidths=[0.8*inch, 1*inch, 3*inch])
    ref_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
    ]))
    elements.append(ref_table)
    elements.append(Spacer(1, 20))

    # 健康建议
    elements.append(Paragraph("=== Health Recommendations ===", section_style))

    recommendations = prediction_result.get('recommendations', [])
    if not recommendations:
        # 根据风险等级生成默认建议
        risk_level = prediction_result.get('risk_level', '')
        if risk_level == 'High Risk':
            recommendations = [
                'Consult cardiologist immediately for comprehensive evaluation',
                'Follow medical advice for further diagnosis and treatment',
                'Rest well, avoid strenuous exercise and emotional excitement',
                'Monitor blood pressure and blood sugar regularly'
            ]
        elif risk_level == 'Medium Risk':
            recommendations = [
                'Schedule cardiology consultation for evaluation soon',
                'Adjust lifestyle, control diet and weight',
                'Engage in moderate aerobic exercise such as brisk walking, swimming',
                'Quit smoking, limit alcohol, maintain good sleep habits'
            ]
        else:
            recommendations = [
                'Current indicators are good, maintain healthy lifestyle',
                'Have regular health checkups, focus on cardiovascular health',
                'Keep balanced diet and moderate exercise',
                'Seek medical attention promptly if discomfort occurs'
            ]

    for i, rec in enumerate(recommendations, 1):
        elements.append(Paragraph(f"{i}. {rec}", content_style))

    elements.append(Spacer(1, 20))

    # 报告说明
    disclaimer = """
    <para alignment='left' fontSize='9'>
    <b>Disclaimer:</b> This report is generated by machine learning model for reference and auxiliary diagnosis only.
    It should not replace professional medical diagnosis. Please consult medical professionals for health concerns.
    </para>
    """
    elements.append(Paragraph(disclaimer, content_style))

    # 页脚
    footer = f"""
    <para alignment='center' fontSize='8'
    textColor='#94a3b8'
    >
    Report ID: {hash(str(input_data) + str(datetime.datetime.now())) % 1000000:06d} | Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </para>
    """
    elements.append(Paragraph(footer, content_style))

    # Generate PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
