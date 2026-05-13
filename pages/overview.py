import streamlit as st

from config.config import FEATURE_LABELS, PathConfig
from src.heart_pipeline import TARGET_COLUMN, clean_dataset, describe_class_balance, load_dataset


def render_overview() -> None:
    # ── 页面标题 ──────────────────────────────────────────────────────────────
    st.title("基于机器学习的心脏病数据分析与预测")
    st.caption("数据清洗 · 特征筛选 · 多模型对比 · 风险预测")
    st.divider()

    # ── 加载数据 ───────────────────────────────────────────────────────────────
    raw = load_dataset(PathConfig.DEFAULT_DATA_FILE)
    clean, summary = clean_dataset(raw)
    balance = describe_class_balance(clean)

    # ── 数据概况指标 ──────────────────────────────────────────────────────────
    st.subheader("数据集概况")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("原始样本", summary.rows_before)
    c2.metric("清洗后样本", summary.rows_after)
    c3.metric("特征数", len(clean.columns) - 1)
    c4.metric("阳性占比", f"{summary.positive_rate * 100:.1f}%")

    st.divider()

    # ── 研究目标 ──────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("研究目标")
        st.markdown("""
1. 对心脏病患者诊疗数据进行**清洗与集成**
2. 从不同维度开展**统计分析与可视化**
3. 构建并比较多种**机器学习模型**，优化参数
4. 集成到交互界面，实现**个体风险预测**
        """)

        st.subheader("技术栈")
        st.markdown("""
`Python` &nbsp;`Scikit-learn` &nbsp;`XGBoost` &nbsp;`SHAP` &nbsp;`Streamlit` &nbsp;`Plotly`
        """)

    with col_right:
        st.subheader("数据集字段")
        field_rows = [
            {"英文字段": k, "中文含义": v}
            for k, v in FEATURE_LABELS.items()
        ]
        st.dataframe(field_rows, hide_index=True, use_container_width=True)

    st.divider()

    # ── 使用教程（面向零基础用户）────────────────────────────────────────────────
    st.subheader("如何使用本系统（4 步完成）")

    with st.container(border=True):
        st.markdown("""
**第 1 步：数据探索（左侧菜单第 2 项）**
- 点击"使用内置数据集"按钮加载数据
- 查看数据概览、清洗详情、分布图表
- 系统会自动处理缺失值和异常值，你不需要做任何操作

**第 2 步：特征工程（左侧菜单第 3 项）**
- 选择模式：专业版（全部指标）或 筛查版（无生化指标）
- 拖动滑块选择保留的特征数量（越多信息越丰富，但可能过拟合）
- 查看系统自动评分出的最重要的特征

**第 3 步：模型训练（左侧菜单第 4 项）**
- 点击"开始多模型训练"按钮（等待约 10-30 秒）
- 系统会同时训练 8 种不同的模型并自动评比
- 在"参数优化"标签页可以对胜出的模型进一步调优
- 在"结果验证"标签页可以查看模型的详细表现

**第 4 步：风险预测（左侧菜单第 5 项）**
- 输入患者的基本指标（年龄、血压、血糖等）
- 点击"执行风险预测"查看结果
- 可以下载 PDF 报告
        """)

    with st.container(border=True):
        st.markdown("""
**一些简单的解释：**

| 术语 | 大白话解释 |
|------|------------|
| 机器学习模型 | 就是让电脑从数据中"学习"规律，然后用来预测新数据 |
| 训练 | 电脑学习规律的过程，就像学生做题练习 |
| 准确率 | 模型猜对的概率（猜对/总次数） |
| 召回率 | 有病的人里面，模型成功找出多少个 |
| F1 分数 | 综合了准确率和召回率的评分，越高越好 |
| AUC | 模型区分有病/没病的能力，1.0 完美，0.5 等于瞎猜 |
| 过拟合 | 模型把训练数据背下来了，但遇到新数据就表现不好 |
| SMOTE | 一种解决"数据不平衡"的方法，好比考试卷上把少的题型多出几道 |"过采样"
        """)

    st.divider()

    # ── 数据预览 ──────────────────────────────────────────────────────────────
    st.subheader("数据预览")
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.caption("前 10 条记录（清洗后）")
        st.dataframe(clean.head(10), use_container_width=True)

    with right:
        if TARGET_COLUMN in clean.columns:
            st.caption("阳性 / 阴性分布")
            st.bar_chart(
                balance.set_index("label")["count"],
                color="#3b82f6",
                use_container_width=True,
            )
        else:
            st.caption("数据概况")
            st.info("选择诊断结果列后可查看分布")
