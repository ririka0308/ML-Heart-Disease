"""
heart_pipeline.py - 统一入口，从子模块重新导出所有公开 API
所有 `from src.heart_pipeline import ...` 的导入路径继续有效。
"""

# _config
from ._config import (
    ALL_CATEGORICAL_FEATURES,
    ALL_NUMERIC_FEATURES,
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    CLINICAL_FEATURES,
    CLINICAL_RANGES,
    COLUMN_ALIASES,
    DatasetSummary,
    DERIVED_NUMERIC_FEATURES,
    EXTENDED_CATEGORICAL_FEATURES,
    EXTENDED_FEATURE_COLUMNS,
    EXTENDED_NUMERIC_FEATURES,
    FEATURE_COLUMNS,
    MEDICAL_PRIOR_WEIGHTS,
    MODE_HINTS,
    MODE_LABELS,
    RISK_COLORS,
    NUMERIC_FEATURES,
    RESULT_VALUE_MAP,
    SCREENING_FEATURES,
    TARGET_COLUMN,
    cached_func,
    encode_target,
)

# _data
from ._data import (
    auto_map_columns,
    clean_dataset,
    describe_class_balance,
    load_dataset,
    save_processed_dataset,
)

# _features
from ._features import (
    build_correlation_frame,
    build_preprocessor,
    compute_feature_scores,
    engineer_age_features,
    engineer_age_features_dict,
    extract_feature_importance,
    get_all_numeric_features,
    get_features_by_mode,
    get_transformed_feature_names,
    prepare_xy,
)

# _models
from ._models import (
    HAS_SMOTE,
    SMOTE,
    XGBClassifier,
    _adjust_xgb_weight,
    _apply_smote,
    build_pipeline,
    compare_models,
    evaluate_predictions,
    get_model_catalog,
    get_param_grid,
    load_model_bundle,
    save_model_bundle,
    tune_model,
)

# _prediction
from ._prediction import (
    build_single_case,
    predict_risk,
    validate_and_clamp_value,
)
