# -*- coding: utf-8 -*-
"""
页面模块
包含所有Streamlit页面的实现 - 数据挖掘全生命周期
"""

from .overview import render_overview
from .eda import render_eda
from .feature_engineering import render_feature_engineering
from .model_lab import render_model_lab
from .prediction import render_prediction

__all__ = [
    'render_overview',
    'render_eda',
    'render_feature_engineering', 
    'render_model_lab',
    'render_prediction'
]
