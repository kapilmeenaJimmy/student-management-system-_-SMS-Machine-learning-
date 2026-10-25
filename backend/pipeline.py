"""pipeline.py — Training, conformal calibration, SHAP explainability."""
from __future__ import annotations
import logging, time
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from dataset import (
    CATEGORICAL_COLS, MACRO_COLS, NUMERIC_COLS, load_dataset, get_feature_columns
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "data" / "models"
MODEL_PATH = MODEL_DIR / "student_model.joblib"
META_PATH = MODEL_DIR / "meta.joblib"

NON_MACRO_NUMERIC = [c for c in NUMERIC_COLS if c not in MACRO_COLS]


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
            ("num", RobustScaler(), NON_MACRO_NUMERIC + ["weekly_study_hours"]),
            ("macro", StandardScaler(), MACRO_COLS),
        ],
        remainder="drop",
    )


def _make_pipelines() -> dict:
    pre = _build_preprocessor()
    candidates = {
        "Ridge": Pipeline([("pre", _build_preprocessor()), ("model", Ridge(alpha=10.0))]),
        "RandomForest": Pipeline([
            ("pre", _build_preprocessor()),
            ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ]),
    }
    if HAS_LGBM:
        candidates["LGBM"] = Pipeline([
            ("pre", _build_preprocessor()),
            ("model", LGBMRegressor(n_estimators=150, learning_rate=0.05, random_state=42, verbose=-1)),
        ])
    return candidates


def train_and_select(df: pd.DataFrame) -> Tuple[Pipeline, dict]:
    """5-fold CV → select best by MAE → return fitted pipeline + meta."""
    feature_cols = get_feature_columns()
    X = df[feature_cols].copy()
    y = df["G_Final"].copy()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    candidates = _make_pipelines()
    results = {}

    for name, pipe in candidates.items():
        mae_scores = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_absolute_error")
        r2_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
        results[name] = {"mae": mae_scores.mean(), "r2": r2_scores.mean()}
        logger.info("CV %s → MAE=%.3f R2=%.3f", name, mae_scores.mean(), r2_scores.mean())

    best_name = min(results, key=lambda k: results[k]["mae"])
    logger.info("Selected model: %s", best_name)

    # Split: 70% train, 15% cal, 15% test
    n = len(X)
    idx = np.random.default_rng(42).permutation(n)
    train_end = int(0.70 * n)
    cal_end = int(0.85 * n)

    X_train, y_train = X.iloc[idx[:train_end]], y.iloc[idx[:train_end]]
    X_cal, y_cal = X.iloc[idx[train_end:cal_end]], y.iloc[idx[train_end:cal_end]]
    X_test, y_test = X.iloc[idx[cal_end:]], y.iloc[idx[cal_end:]]

    best_pipe = candidates[best_name]
    best_pipe.fit(X_train, y_train)

    # Conformal calibration (split conformal, alpha=0.05 → 95% coverage)
    y_cal_pred = best_pipe.predict(X_cal)
    nonconformity = np.abs(y_cal.values - y_cal_pred)
    alpha = 0.05
    q_level = np.ceil((1 - alpha) * (len(nonconformity) + 1)) / len(nonconformity)
    q_level = min(q_level, 1.0)
    q_hat = float(np.quantile(nonconformity, q_level))
    logger.info("Conformal q_hat=%.3f (95%% coverage)", q_hat)

    # Test coverage
    y_test_pred = best_pipe.predict(X_test)
    covered = np.mean(
        (y_test.values >= y_test_pred - q_hat) & (y_test.values <= y_test_pred + q_hat)
    )
    logger.info("Empirical test coverage: %.3f", covered)

    # MAD for counterfactual optimizer
    mad_dict = {}
    for col in feature_cols:
        vals = X_train[col].values.astype(float)
        mad = float(np.median(np.abs(vals - np.median(vals))))
        mad_dict[col] = max(mad, 1e-6)

    meta = {
        "model_name": best_name,
        "q_hat": q_hat,
        "coverage": float(covered),
        "mad": mad_dict,
        "feature_cols": feature_cols,
        "cv_results": results,
        "n_records": n,
    }
    return best_pipe, meta


def compute_shap(pipe: Pipeline, X_row: pd.DataFrame, feature_cols: list) -> dict:
    """Compute SHAP values for a single row; return top-5 positive + top-5 risk."""
    try:
        X_transformed = pipe.named_steps["pre"].transform(X_row)
        model = pipe.named_steps["model"]

        if hasattr(model, "predict"):
            try:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_transformed)[0]
            except Exception:
                explainer = shap.LinearExplainer(model, X_transformed)
                shap_vals = explainer.shap_values(X_transformed)[0]
        else:
            return {"positive": [], "risk": []}

        # Map back to original feature names (OHE expands categoricals)
        ohe = pipe.named_steps["pre"].named_transformers_["cat"]
        cat_names = ohe.get_feature_names_out(CATEGORICAL_COLS).tolist()
        num_names = NON_MACRO_NUMERIC + ["weekly_study_hours"]
        macro_names = MACRO_COLS
        all_names = cat_names + num_names + macro_names

        # Aggregate OHE features back to original column names
        shap_by_orig: dict = {}
        for fname, sv in zip(all_names, shap_vals):
            orig = fname.split("_x0_")[0] if "_x0_" in fname else fname
            # For OHE: feature names are like "cat_x0_1" → we want the original col
            # Actually sklearn uses col_value format for OHE get_feature_names_out
            # Let's map properly
            shap_by_orig[orig] = shap_by_orig.get(orig, 0.0) + float(sv)

        # Map OHE names back: format is "colname_val"
        shap_clean: dict = {}
        for fname, sv in zip(all_names, shap_vals):
            # OHE feature names from get_feature_names_out: "marital_status_1", etc.
            matched = False
            for cat_col in CATEGORICAL_COLS:
                if fname.startswith(cat_col):
                    shap_clean[cat_col] = shap_clean.get(cat_col, 0.0) + float(sv)
                    matched = True
                    break
            if not matched:
                shap_clean[fname] = float(sv)

        sorted_items = sorted(shap_clean.items(), key=lambda x: x[1], reverse=True)
        positive = [{"feature": k, "value": float(X_row[k].iloc[0]) if k in X_row.columns else 0.0, "impact": v}
                    for k, v in sorted_items if v > 0][:5]
        risk = [{"feature": k, "value": float(X_row[k].iloc[0]) if k in X_row.columns else 0.0, "impact": v}
                for k, v in sorted_items if v < 0][-5:]

        return {"positive": positive, "risk": risk}
    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)
        return {"positive": [], "risk": []}


def ensure_model() -> Tuple[Pipeline, dict]:
    """Load from disk or retrain if missing."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and META_PATH.exists():
        logger.info("Loading cached model from %s", MODEL_PATH)
        pipe = joblib.load(MODEL_PATH)
        meta = joblib.load(META_PATH)
        return pipe, meta

    logger.info("No cached model found — training now …")
    df = load_dataset()
    pipe, meta = train_and_select(df)
    joblib.dump(pipe, MODEL_PATH)
    joblib.dump(meta, META_PATH)
    logger.info("Model saved to %s", MODEL_PATH)
    return pipe, meta
