# 基于机器学习的心脏病数据分析与预测系统

Heart Disease Risk Analysis & Prediction System

> 本科毕业设计作品 · 数据挖掘与机器学习全流程应用

---

## 系统功能

系统围绕"数据 → 特征 → 模型 → 预测"的标准机器学习流水线构建，提供 5 个核心交互页面：

| 页面 | 功能 |
|------|------|
| **项目概览** | 数据集介绍、技术栈说明、系统使用教程（零基础友好） |
| **数据探索** | 数据加载（内置/上传）、自动列名映射、缺失值处理、分布分析、相关性矩阵 |
| **特征工程** | 临床/筛查双模式切换、综合特征评分（互信息 + 随机森林）、特征可视化选择 |
| **模型实验** | 8 种模型多维度对比（含学习曲线）、GridSearch 参数优化、混淆矩阵与特征重要性 |
| **风险预测** | 个体风险概率输出、参考值对照、PDF 报告下载 |

---

## 技术栈

| 层 | 技术 |
|----|------|
| 前端 | Streamlit 1.28+ |
| 数据处理 | Pandas, NumPy, SciPy |
| 可视化 | Plotly |
| 机器学习 | Scikit-learn, XGBoost |
| 类别平衡 | SMOTE (imbalanced-learn) |
| PDF 生成 | ReportLab |
| 架构 | 模块化设计（配置/数据/特征/模型/预测 五模块分离） |

---

## 支持的模型

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- SVM
- KNN
- MLP Neural Network
- XGBoost

---

## 数据集

使用心脏病临床数据集 `Medicaldataset.csv`（位于 `data/` 目录），包含以下指标：

**基础指标：** 年龄、性别、心率、收缩压、舒张压、血糖

**生化指标：** CK-MB、肌钙蛋白（Troponin）

**扩展字段（用户上传数据集兼容）：** 胆固醇、最大心率、ST 段压低、胸痛类型、运动心绞痛、血管狭窄数、铊扫描

---

## 快速开始

### 本地运行

```bash
# 1. 克隆仓库
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动应用
python run_app.py
```

打开浏览器访问 `http://localhost:8505`

### 在线访问（答辩用）

部署后可在线访问：

```
https://你的仓库名.streamlit.app
```

---

## 项目结构

```
├── app.py                 # 应用入口
├── run_app.py             # 本地启动脚本（端口检测 + 缓存清理）
├── requirements.txt       # 依赖清单
├── config/
│   └── config.py          # 项目配置（路径、常量）
├── src/
│   ├── heart_pipeline.py  # 统一入口（从子模块重新导出）
│   ├── _config.py         # 配置常量（特征列表、别名、权重）
│   ├── _data.py           # 数据加载与清洗
│   ├── _features.py       # 特征工程与相关性分析
│   ├── _models.py         # 模型训练、优化与评估
│   ├── _prediction.py     # 个体风险预测
│   └── pdf_generator.py   # PDF 报告生成
├── pages/
│   ├── overview.py        # 项目概览页
│   ├── eda.py             # 数据探索页
│   ├── feature_engineering.py  # 特征工程页
│   ├── model_lab.py       # 模型实验页
│   └── prediction.py      # 风险预测页
├── data/
│   └── Medicaldataset.csv # 默认数据集
└── .streamlit/
    └── config.toml        # Streamlit 主题配置
```

---

## 模型性能（参考）

> 以下数据基于默认数据集运行结果，实际值以训练时为准。

| 模型 | Accuracy | F1 Score | AUC |
|------|----------|----------|-----|
| XGBoost | — | — | — |
| Random Forest | — | — | — |
| Gradient Boosting | — | — | — |
| Logistic Regression | — | — | — |
| SVM | — | — | — |
| KNN | — | — | — |
| Decision Tree | — | — | — |
| MLP Neural Network | — | — | — |

---

## 答辩亮点

- **双模式设计**：临床模式（含 Troponin/CK-MB 生化指标）与筛查模式（仅常规指标），体现场景化思维
- **特征工程闭环**：从特征评分到模型训练的数据流打通，特征工程结果直接影响模型输入
- **学习曲线分析**：结果验证 Tab 内置学习曲线，直观判断过拟合/欠拟合
- **模块化解耦**：配置/数据/特征/模型/预测 5 模块分离，通过 `heart_pipeline.py` 统一导出，代码可维护性强
- **已部署在线**：支持答辩现场评委通过手机/电脑直接访问体验

---

## 许可证

MIT License
