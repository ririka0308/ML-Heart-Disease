"""
数据加载与清洗模块
包括数据集加载、列名自动映射、数据清洗、数据集保存等
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.config import AppConfig, PathConfig
from ._config import (
    ALL_NUMERIC_FEATURES,
    COLUMN_ALIASES,
    DatasetSummary,
    EXTENDED_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from ._features import engineer_age_features


# ============================================================
# 列名自动映射
# ============================================================
def auto_map_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    自动将用户数据集的列名映射到系统标准列名

    返回: {系统标准列名: 数据集中实际列名} 的字典
    """
    df_cols = [str(c).strip() for c in df.columns]
    df_col_lower = [c.lower().replace(" ", "_").replace("-", "_") for c in df_cols]

    mapping: dict[str, str] = {}
    used_indices: set[int] = set()

    for std_name, aliases in COLUMN_ALIASES.items():
        # 先尝试精确匹配
        for i, col in enumerate(df_cols):
            if i in used_indices:
                continue
            if col == std_name:
                mapping[std_name] = col
                used_indices.add(i)
                break
        else:
            # 再尝试别名匹配（大小写不敏感）
            alias_lower = [a.lower().replace(" ", "_").replace("-", "_") for a in aliases]
            for i, col_lower in enumerate(df_col_lower):
                if i in used_indices:
                    continue
                if col_lower in alias_lower:
                    mapping[std_name] = df_cols[i]
                    used_indices.add(i)
                    break

    return mapping


# ============================================================
# 数据加载
# ============================================================
def load_dataset(file_path: str | Path | None = None) -> pd.DataFrame:
    """加载数据集，自动检测编码格式"""
    # 处理 Streamlit UploadedFile 对象
    if hasattr(file_path, "read"):
        import io

        file_path.seek(0)
        return pd.read_csv(io.StringIO(file_path.read().decode("utf-8")))

    path = Path(file_path) if file_path else PathConfig.DEFAULT_DATA_FILE
    # 按优先级尝试多种常见编码
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            if "Result" in df.columns or "Age" in df.columns:
                return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(path, encoding="utf-8")


# ============================================================
# 数据清洗
# ============================================================
def clean_dataset(df: pd.DataFrame, clip_outliers: bool = True) -> tuple[pd.DataFrame, DatasetSummary]:
    """执行完整的数据清洗流程：列映射、类型转换、缺失填充、异常值截断、年龄特征派生"""
    data = df.copy()
    rows_before = len(data)
    missing_before = data.isna().sum().to_dict()

    data.columns = [str(col).strip() for col in data.columns]

    # 自动映射列名
    col_map = auto_map_columns(data)
    if col_map:
        data.rename(columns={v: k for k, v in col_map.items()}, inplace=True)

    # 保留数据集中实际存在的特征列（使用扩展特征集以支持不同格式的上传数据集）
    available_cols = [col for col in EXTENDED_FEATURE_COLUMNS + [TARGET_COLUMN] if col in data.columns]
    data = data[available_cols].copy()

    # 数值特征类型转换
    available_numeric = [col for col in ALL_NUMERIC_FEATURES if col in data.columns]
    for col in available_numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce").astype(float)

    # 统一处理性别编码
    if "Gender" in data.columns:
        data["Gender"] = data["Gender"].astype(str).str.strip()
        gender_mapping = {
            "0": "Female", "1": "Male",
            "female": "Female", "male": "Male",
            "男": "Male", "女": "Female",
        }
        data["Gender"] = data["Gender"].replace(gender_mapping).fillna("Unknown")

    # 处理目标变量
    if TARGET_COLUMN in data.columns:
        data[TARGET_COLUMN] = (
            data[TARGET_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"1": AppConfig.POSITIVE_LABEL, "0": AppConfig.NEGATIVE_LABEL})
        )
    # 如果目标列不存在则由后续界面让用户手动选择

    # 删除重复行
    duplicate_rows = int(data.duplicated().sum())
    data = data.drop_duplicates().reset_index(drop=True)

    # 数值特征缺失值填充（中位数）
    for col in available_numeric:
        data[col] = data[col].fillna(data[col].median())

    # 过滤无效目标值
    if TARGET_COLUMN in data.columns:
        valid_labels = [AppConfig.POSITIVE_LABEL, AppConfig.NEGATIVE_LABEL]
        data = data[data[TARGET_COLUMN].isin(valid_labels)].reset_index(drop=True)

    # 异常值处理（IQR 方法）
    clipped_cells = 0
    if clip_outliers:
        for col in available_numeric:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            below = data[col] < lower
            above = data[col] > upper
            clipped_cells += int(below.sum() + above.sum())
            data.loc[below, col] = lower
            data.loc[above, col] = upper

    # 年龄特征工程（派生非线性特征）
    data = engineer_age_features(data)

    # 阳性率
    if TARGET_COLUMN in data.columns:
        positive_rate = float((data[TARGET_COLUMN] == AppConfig.POSITIVE_LABEL).mean())
    else:
        positive_rate = 0.0

    summary = DatasetSummary(
        rows_before=rows_before,
        rows_after=len(data),
        duplicate_rows=duplicate_rows,
        missing_before=missing_before,
        missing_after=data.isna().sum().to_dict(),
        clipped_cells=clipped_cells,
        positive_rate=positive_rate,
    )
    return data, summary


# ============================================================
# 数据导出
# ============================================================
def save_processed_dataset(df: pd.DataFrame, filename: str = "cleaned_medical_dataset.csv") -> Path:
    """保存清洗后的数据集到 processed 目录"""
    path = PathConfig.PROCESSED_DATA_DIR / filename
    df.to_csv(path, index=False)
    return path


# ============================================================
# 类别分布
# ============================================================
def describe_class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """计算目标变量的类别分布"""
    if TARGET_COLUMN not in df.columns:
        return pd.DataFrame({"label": ["unknown"], "count": [len(df)], "rate": [1.0]})
    counts = df[TARGET_COLUMN].value_counts().rename_axis("label").reset_index(name="count")
    counts["rate"] = counts["count"] / counts["count"].sum()
    return counts
