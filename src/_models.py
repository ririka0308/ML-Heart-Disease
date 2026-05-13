"""
模型训练与评估模块
包括模型目录、多模型对比、参数优化、评估指标、模型持久化
"""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config.config import AppConfig, PathConfig
from ._config import CATEGORICAL_FEATURES
from ._features import build_preprocessor, get_features_by_mode

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    SMOTE = None
    HAS_SMOTE = False


def get_model_catalog(fast_mode: bool = False) -> dict[str, Any]:
    if fast_mode:
        models: dict[str, Any] = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
            "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
            "MLP Neural Network": MLPClassifier(
                hidden_layer_sizes=(32, 16), activation="relu",
                early_stopping=True, max_iter=300, random_state=42,
            ),
        }
        if XGBClassifier is not None:
            models["XGBoost"] = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                scale_pos_weight=1, random_state=42, n_jobs=-1,
            )
    else:
        models: dict[str, Any] = {
            "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
            "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=9, weights="distance"),
            "MLP Neural Network": MLPClassifier(
                hidden_layer_sizes=(64, 32, 16), activation="relu",
                early_stopping=True, max_iter=500, random_state=42,
            ),
        }
        if XGBClassifier is not None:
            models["XGBoost"] = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                scale_pos_weight=1, random_state=42, n_jobs=-1,
            )
    return models


def build_pipeline(model, numeric_features=None, categorical_features=None):
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def evaluate_predictions(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }


def _adjust_xgb_weight(models, y_train):
    if XGBClassifier is None:
        return
    for name in list(models.keys()):
        if isinstance(models[name], XGBClassifier):
            neg = (y_train == 0).sum()
            pos = (y_train == 1).sum()
            models[name].set_params(scale_pos_weight=neg / max(pos, 1))


def _apply_smote(X_train, y_train):
    if not HAS_SMOTE:
        return X_train, y_train
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    return (pd.DataFrame(X_resampled, columns=X_train.columns),
            pd.Series(y_resampled, name=y_train.name))


def compare_models(
    X, y,
    test_size=AppConfig.TEST_SIZE,
    random_state=AppConfig.RANDOM_STATE,
    mode="clinical",
    fast_mode=True,
    use_smote=False,
    selected_features=None,
    progress_callback=None,
):
    """多模型对比训练，自动评选最优模型"""
    if selected_features is not None:
        available_features = [f for f in selected_features if f in X.columns]
    else:
        features = get_features_by_mode(mode)
        available_features = [f for f in features if f in X.columns]
    X = X[available_features].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if use_smote:
        X_train, y_train = _apply_smote(X_train, y_train)

    cv = StratifiedKFold(n_splits=3 if fast_mode else 5, shuffle=True, random_state=random_state)
    rows = []
    trained = {}
    roc_curves = {}
    training_history = []

    numeric_features = [f for f in X_train.columns if f not in CATEGORICAL_FEATURES]
    categorical_features = [f for f in X_train.columns if f in CATEGORICAL_FEATURES]

    catalog = get_model_catalog(fast_mode=fast_mode)
    _adjust_xgb_weight(catalog, y_train)

    total = len(catalog)
    for idx, (name, estimator) in enumerate(catalog.items(), 1):
        if progress_callback:
            progress_callback(name, idx, total)

        pipe = build_pipeline(
            clone(estimator),
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        rows.append({
            "model": name,
            **metrics,
            "cv_f1_mean": float(scores.mean()),
            "cv_f1_std": float(scores.std()),
        })
        training_history.append({
            "model": name,
            "cv_f1_mean": float(scores.mean()),
            "cv_f1_std": float(scores.std()),
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
        })
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves[name] = {"fpr": fpr, "tpr": tpr}
        trained[name] = pipe

    results = pd.DataFrame(rows).sort_values(["f1", "auc"], ascending=False).reset_index(drop=True)
    best_name = results.iloc[0]["model"]

    return {
        "results": results,
        "best_name": best_name,
        "best_model": trained[best_name],
        "models": trained,
        "roc_curves": roc_curves,
        "X_train": X_train,
        "X_test": X_test[X_train.columns],
        "y_train": y_train,
        "y_test": y_test,
        "trained_mode": mode,
        "training_history": training_history,
    }


def get_param_grid(model_name, fast_mode=True):
    if fast_mode:
        grids = {
            "Logistic Regression": {"model__C": [0.1, 1.0, 10.0], "model__penalty": ["l2"], "model__solver": ["lbfgs"]},
            "Decision Tree": {"model__max_depth": [3, 5, 8], "model__min_samples_split": [2, 5], "model__min_samples_leaf": [1, 2], "model__criterion": ["gini"]},
            "Random Forest": {"model__n_estimators": [100, 200], "model__max_depth": [5, 8], "model__min_samples_split": [2, 5], "model__min_samples_leaf": [1, 2], "model__criterion": ["gini"]},
            "Gradient Boosting": {"model__n_estimators": [100, 150], "model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 4], "model__subsample": [0.9, 1.0]},
            "SVM": {"model__C": [0.5, 1.0, 2.0], "model__kernel": ["rbf"], "model__gamma": ["scale"]},
            "KNN": {"model__n_neighbors": [3, 5, 7], "model__weights": ["uniform", "distance"], "model__metric": ["euclidean"]},
            "MLP Neural Network": {"model__hidden_layer_sizes": [(32,), (32, 16)], "model__alpha": [0.0001, 0.001], "model__learning_rate_init": [0.001], "model__activation": ["relu"]},
        }
    else:
        grids = {
            "Logistic Regression": {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0], "model__penalty": ["l1", "l2", "elasticnet"], "model__solver": ["saga"]},
            "Decision Tree": {"model__max_depth": [3, 5, 8, 12], "model__min_samples_split": [2, 5, 10, 20], "model__min_samples_leaf": [1, 2, 4], "model__criterion": ["gini", "entropy"]},
            "Random Forest": {"model__n_estimators": [100, 200, 300, 500], "model__max_depth": [5, 8, 12, None], "model__min_samples_split": [2, 5, 10], "model__min_samples_leaf": [1, 2, 4], "model__criterion": ["gini", "entropy"]},
            "Gradient Boosting": {"model__n_estimators": [100, 200, 300], "model__learning_rate": [0.01, 0.03, 0.05, 0.1], "model__max_depth": [3, 4, 5], "model__subsample": [0.8, 0.9, 1.0]},
            "SVM": {"model__C": [0.1, 0.5, 1.0, 2.0, 5.0], "model__kernel": ["rbf", "linear", "poly"], "model__gamma": ["scale", "auto"]},
            "KNN": {"model__n_neighbors": [3, 5, 7, 9, 11, 15], "model__weights": ["uniform", "distance"], "model__metric": ["euclidean", "manhattan"]},
            "MLP Neural Network": {"model__hidden_layer_sizes": [(32,), (32, 16), (64, 32), (64, 32, 16), (128, 64, 32)], "model__alpha": [0.0001, 0.001, 0.01], "model__learning_rate_init": [0.001, 0.01], "model__activation": ["relu", "tanh"]},
        }
    if model_name == "XGBoost":
        if fast_mode:
            return {"model__n_estimators": [100, 150], "model__max_depth": [3, 4], "model__learning_rate": [0.05, 0.1], "model__subsample": [0.9, 1.0], "model__colsample_bytree": [0.9, 1.0], "model__scale_pos_weight": [1, 2]}
        else:
            return {"model__n_estimators": [100, 200, 300], "model__max_depth": [3, 4, 5, 6], "model__learning_rate": [0.01, 0.03, 0.05, 0.1], "model__subsample": [0.8, 0.9, 1.0], "model__colsample_bytree": [0.8, 0.9, 1.0], "model__scale_pos_weight": [1, 2, 5]}
    return grids[model_name]


def tune_model(X, y, model_name, mode="clinical", fast_mode=True, use_smote=False):
    features = get_features_by_mode(mode)
    available_features = [f for f in features if f in X.columns]
    X = X[available_features].copy()

    numeric_features = [f for f in X.columns if f not in CATEGORICAL_FEATURES]
    categorical_features = [f for f in X.columns if f in CATEGORICAL_FEATURES]

    catalog = get_model_catalog(fast_mode=fast_mode)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=AppConfig.TEST_SIZE, random_state=42, stratify=y
    )
    _adjust_xgb_weight(catalog, y_train)
    estimator = clone(catalog[model_name])
    pipeline = build_pipeline(
        estimator,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=get_param_grid(model_name, fast_mode=True),
        scoring="f1", cv=3, n_jobs=1,
    )
    if use_smote:
        X_train, y_train = _apply_smote(X_train, y_train)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    return {
        "best_model": best_model,
        "best_params": grid.best_params_,
        "best_cv_score": float(grid.best_score_),
        "metrics": evaluate_predictions(y_test, y_pred, y_prob),
        "cv_results": pd.DataFrame(grid.cv_results_).sort_values("rank_test_score"),
    }


def save_model_bundle(model, summary=None, filename="best_model.joblib"):
    path = PathConfig.MODEL_DIR / filename
    payload = {"model": model, "summary": summary}
    joblib.dump(payload, path)
    return path


def load_model_bundle(path):
    return joblib.load(path)
