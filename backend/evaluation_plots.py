"""evaluation_plots.py — Generates 5 quantitative diagnostic curves.

Run standalone:
    python backend/evaluation_plots.py

Outputs PNG files to: backend/data/plots/
  1. cv_mae_comparison.png       — 5-fold CV MAE bar chart per model
  2. conformal_coverage.png      — Calibration coverage curve (alpha sweep)
  3. score_distribution.png      — G_Final score distribution + quartiles
  4. residuals_qq.png            — QQ plot of prediction residuals (test set)
  5. shap_global_importance.png  — Global SHAP feature importance (top 15)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

# ── Style config ──────────────────────────────────────────────────────────────
BG = "#07090E"
SURFACE = "#0E1523"
ACCENT = "#00D4FF"
EMERALD = "#00C785"
AMBER = "#F59E0B"
RED = "#F04040"
PURPLE = "#9D6EFA"
TEXT = "#DCE2EE"
MUTED = "#52617A"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "grid.color": "#1E2A40",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.dpi": 150,
})

PLOT_DIR = Path(__file__).parent / "data" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add backend/ to path when running as script
sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_feature_columns, load_dataset
from pipeline import _build_preprocessor, _make_pipelines, ensure_model

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error


def _frame(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, color=TEXT, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, axis="y", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)


# ── Plot 1: 5-fold CV MAE Comparison ─────────────────────────────────────────
def plot_cv_mae(df: pd.DataFrame):
    logger.info("Plot 1: CV MAE comparison …")
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df["G_Final"]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    candidates = _make_pipelines()

    results = {}
    for name, pipe in candidates.items():
        scores = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_absolute_error")
        results[name] = scores
        logger.info("  %s: MAE=%.3f±%.3f", name, scores.mean(), scores.std())

    names = list(results.keys())
    means = [results[n].mean() for n in names]
    stds  = [results[n].std()  for n in names]
    colors = [ACCENT, EMERALD, PURPLE, AMBER, RED][:len(names)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, means, yerr=stds, color=colors, alpha=0.85,
                  error_kw={"ecolor": TEXT, "elinewidth": 1.5, "capsize": 6},
                  width=0.45, zorder=3)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.15,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=10, color=TEXT)
    ax.axhline(min(means), color=AMBER, linestyle=":", linewidth=1.2, label="Best MAE")
    _frame(ax, "5-Fold CV Mean Absolute Error by Model", "Model", "MAE (score points)")
    ax.legend(facecolor=SURFACE, edgecolor=MUTED, labelcolor=TEXT, fontsize=9)
    fig.tight_layout()
    out = PLOT_DIR / "cv_mae_comparison.png"
    fig.savefig(out); plt.close(fig)
    logger.info("  Saved → %s", out)


# ── Plot 2: Conformal Coverage Curve (alpha sweep) ───────────────────────────
def plot_conformal_coverage(df: pd.DataFrame):
    logger.info("Plot 2: Conformal coverage curve …")
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df["G_Final"].values

    rng = np.random.default_rng(42)
    n = len(X)
    idx = rng.permutation(n)
    train_end = int(0.70 * n)
    cal_end = int(0.85 * n)

    # Use Ridge (fastest) for this diagnostic
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("pre", _build_preprocessor()), ("model", Ridge(alpha=10.0))])
    pipe.fit(X.iloc[idx[:train_end]], y[idx[:train_end]])

    y_cal_pred = pipe.predict(X.iloc[idx[train_end:cal_end]])
    nonconf = np.abs(y[idx[train_end:cal_end]] - y_cal_pred)

    y_test_pred = pipe.predict(X.iloc[idx[cal_end:]])
    y_test_true = y[idx[cal_end:]]

    alphas = np.linspace(0.01, 0.40, 40)
    coverages = []
    for alpha in alphas:
        q_lvl = min(np.ceil((1 - alpha) * (len(nonconf) + 1)) / len(nonconf), 1.0)
        q_hat = float(np.quantile(nonconf, q_lvl))
        covered = np.mean(
            (y_test_true >= y_test_pred - q_hat) & (y_test_true <= y_test_pred + q_hat)
        )
        coverages.append(covered)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(1 - alphas, coverages, color=ACCENT, linewidth=2.5, label="Empirical coverage")
    ax.plot([0.6, 1.0], [0.6, 1.0], color=MUTED, linestyle="--", linewidth=1.2,
            label="Perfect calibration")
    ax.axvline(0.95, color=AMBER, linestyle=":", linewidth=1.2, label="α=0.05 target")
    ax.scatter([0.95], [coverages[np.argmin(np.abs(1 - alphas - 0.95))]],
               color=AMBER, s=60, zorder=5)
    ax.set_xlim(0.6, 1.0); ax.set_ylim(0.6, 1.0)
    _frame(ax, "Conformal Coverage vs. Nominal Level (Split Conformal)",
           "Nominal Coverage (1−α)", "Empirical Coverage")
    ax.legend(facecolor=SURFACE, edgecolor=MUTED, labelcolor=TEXT, fontsize=9)
    fig.tight_layout()
    out = PLOT_DIR / "conformal_coverage.png"
    fig.savefig(out); plt.close(fig)
    logger.info("  Saved → %s", out)


# ── Plot 3: Score Distribution ────────────────────────────────────────────────
def plot_score_distribution(df: pd.DataFrame):
    logger.info("Plot 3: G_Final score distribution …")
    scores = df["G_Final"].values
    q25, q50, q75 = np.percentile(scores, [25, 50, 75])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(scores, bins=50, color=ACCENT, alpha=0.75, edgecolor="none", zorder=3)
    for q, label, color in [(q25, "Q1", AMBER), (q50, "Median", EMERALD), (q75, "Q3", PURPLE)]:
        ax.axvline(q, color=color, linewidth=1.5, linestyle="--", label=f"{label}: {q:.1f}")
    ax.set_xlim(0, 100)
    _frame(ax, f"G_Final Score Distribution (n={len(scores):,})",
           "G_Final Score (0–100)", "Frequency")
    ax.legend(facecolor=SURFACE, edgecolor=MUTED, labelcolor=TEXT, fontsize=9)
    # Grade bands
    for lo, hi, lbl, col in [
        (0, 40, "F", RED), (40, 50, "C", AMBER),
        (50, 60, "B", PURPLE), (60, 100, "A–O", EMERALD),
    ]:
        ax.axvspan(lo, hi, alpha=0.04, color=col)
        ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.92, lbl,
                ha="center", fontsize=8.5, color=col, alpha=0.7)
    fig.tight_layout()
    out = PLOT_DIR / "score_distribution.png"
    fig.savefig(out); plt.close(fig)
    logger.info("  Saved → %s", out)


# ── Plot 4: Residuals QQ Plot ─────────────────────────────────────────────────
def plot_residuals_qq(df: pd.DataFrame):
    logger.info("Plot 4: Residuals QQ plot …")
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df["G_Final"].values

    rng = np.random.default_rng(42)
    n = len(X)
    idx = rng.permutation(n)
    train_end = int(0.70 * n)
    cal_end = int(0.85 * n)

    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("pre", _build_preprocessor()), ("model", Ridge(alpha=10.0))])
    pipe.fit(X.iloc[idx[:train_end]], y[idx[:train_end]])
    y_test_pred = pipe.predict(X.iloc[idx[cal_end:]])
    y_test_true = y[idx[cal_end:]]
    residuals = y_test_true - y_test_pred

    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(osm, osr, color=ACCENT, s=8, alpha=0.65, zorder=3, label="Residuals")
    fit_x = np.array([osm.min(), osm.max()])
    ax.plot(fit_x, slope * fit_x + intercept, color=AMBER, linewidth=1.8,
            linestyle="--", label=f"Normal fit (R={r:.3f})")
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=":")
    _frame(ax, "QQ Plot: Prediction Residuals vs. Normal",
           "Theoretical Quantiles", "Sample Quantiles")
    ax.legend(facecolor=SURFACE, edgecolor=MUTED, labelcolor=TEXT, fontsize=9)
    mae_val = mean_absolute_error(y_test_true, y_test_pred)
    ax.text(0.97, 0.04, f"Test MAE: {mae_val:.2f} pts",
            transform=ax.transAxes, ha="right", fontsize=9, color=MUTED)
    fig.tight_layout()
    out = PLOT_DIR / "residuals_qq.png"
    fig.savefig(out); plt.close(fig)
    logger.info("  Saved → %s", out)


# ── Plot 5: Global SHAP Feature Importance ────────────────────────────────────
def plot_shap_global(df: pd.DataFrame):
    logger.info("Plot 5: Global SHAP feature importance …")
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed, skipping plot 5")
        return

    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df["G_Final"].values

    rng = np.random.default_rng(42)
    n = len(X)
    idx = rng.permutation(n)
    train_end = int(0.70 * n)

    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("pre", _build_preprocessor()), ("model", Ridge(alpha=10.0))])
    pipe.fit(X.iloc[idx[:train_end]], y[idx[:train_end]])

    X_sample = X.iloc[idx[train_end:train_end + 200]]
    X_transformed = pipe.named_steps["pre"].transform(X_sample)

    model = pipe.named_steps["model"]
    try:
        explainer = shap.LinearExplainer(model, X_transformed)
        shap_vals = explainer.shap_values(X_transformed)
    except Exception as e:
        logger.warning("SHAP failed: %s", e)
        return

    # Get feature names from preprocessor
    from dataset import CATEGORICAL_COLS, NUMERIC_COLS, MACRO_COLS
    ohe = pipe.named_steps["pre"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(CATEGORICAL_COLS).tolist()
    num_names = [c for c in NUMERIC_COLS if c not in MACRO_COLS] + ["weekly_study_hours"]
    macro_names = MACRO_COLS
    all_names = cat_names + num_names + macro_names

    # Aggregate OHE back to original
    from dataset import CATEGORICAL_COLS as CAT_COLS
    shap_clean: dict[str, float] = {}
    for fname, sv_col in zip(all_names, shap_vals.T):
        matched = False
        for cat_col in CAT_COLS:
            if fname.startswith(cat_col):
                shap_clean[cat_col] = shap_clean.get(cat_col, 0.0) + float(np.mean(np.abs(sv_col)))
                matched = True; break
        if not matched:
            shap_clean[fname] = float(np.mean(np.abs(sv_col)))

    sorted_items = sorted(shap_clean.items(), key=lambda x: x[1], reverse=True)[:15]
    names_sorted = [x[0].replace("curricular_units_", "CU_") for x, _ in zip(sorted_items, sorted_items)]
    # Fix unpacking
    feat_names_sorted = [item[0].replace("curricular_units_", "CU_") for item in sorted_items]
    importance_sorted = [item[1] for item in sorted_items]

    colors = [EMERALD if v > np.median(importance_sorted) else ACCENT for v in importance_sorted]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(feat_names_sorted[::-1], importance_sorted[::-1],
                   color=colors[::-1], alpha=0.85, height=0.65)
    _frame(ax, "Global SHAP Feature Importance (Top 15)",
           "Mean |SHAP value|", "Feature")
    for bar, val in zip(bars, importance_sorted[::-1]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8.5, color=MUTED)
    ax.set_xlim(0, max(importance_sorted) * 1.18)
    fig.tight_layout()
    out = PLOT_DIR / "shap_global_importance.png"
    fig.savefig(out); plt.close(fig)
    logger.info("  Saved → %s", out)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Loading dataset …")
    df = load_dataset()
    logger.info("Dataset: %d rows, %d cols", len(df), len(df.columns))

    plot_cv_mae(df)
    plot_conformal_coverage(df)
    plot_score_distribution(df)
    plot_residuals_qq(df)
    plot_shap_global(df)

    logger.info("All 5 diagnostic plots saved to %s", PLOT_DIR)
    print(f"\n✓ Plots saved to: {PLOT_DIR}")
