"""Compare exploratory 60:40 and 80:20 splits for the current UCI-derived target.

This is a diagnostic report, not a validation claim: G_Final is computed from
semester-grade/admission-grade columns that are also present among the inputs.
That target/input overlap creates leakage and must be addressed before metrics
are interpreted as predictive performance on unseen future student outcomes.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from dataset import load_dataset, get_feature_columns
from pipeline import _make_pipelines

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def run() -> dict:
    df = load_dataset()
    X = df[get_feature_columns()].copy()
    y = df["G_Final"].copy()
    rows: list[dict] = []
    for test_size, split_label in ((0.40, "60:40"), (0.20, "80:20")):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        for model_name, estimator in _make_pipelines().items():
            estimator.fit(X_train, y_train)
            predictions = estimator.predict(X_test)
            rows.append({
                "split": split_label,
                "test_size": test_size,
                "model": model_name,
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "mae": float(mean_absolute_error(y_test, predictions)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "r2": float(r2_score(y_test, predictions)),
            })

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "platform": platform.platform(),
        "dataset": "UCI Predict Students' Dropout and Academic Success (downloaded/cached by backend/dataset.py)",
        "records_after_preprocessing": int(len(df)),
        "features": get_feature_columns(),
        "target": "G_Final",
        "split_seed": 42,
        "selection_rule": "No model selection; all current candidate pipelines are reported per split.",
        "metrics": rows,
        "critical_limitation": (
            "Exploratory only. G_Final is deterministically constructed from first-semester grade, "
            "second-semester grade, and admission grade, and those columns are also included in X. "
            "This target/input overlap is leakage; these metrics must not be presented as valid "
            "future-performance generalization evidence."
        ),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "split_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    md = [
        "# Exploratory split evaluation", "",
        f"Generated (UTC): `{report['generated_at_utc']}`  ",
        f"Python: `{report['python_version']}` · scikit-learn: `{report['scikit_learn_version']}`  ",
        f"Rows after preprocessing: **{report['records_after_preprocessing']}**  ",
        "Split seed: `42`", "",
        "> [!CAUTION]", "> This report is exploratory only. The target `G_Final` is derived from grade/admission columns that are also included as model inputs. This is target leakage, so these metrics are not valid evidence of future student-performance generalization.", "",
        "| Split | Model | Train rows | Test rows | MAE | RMSE | R² |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(f"| {r['split']} | {r['model']} | {r['train_rows']} | {r['test_rows']} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['r2']:.4f} |")
    md += ["", "## Interpretation", "", "A larger training partition can change estimation variance, but an 80:20 split does not inherently outperform a 60:40 split. Compare these rows only as a diagnostic under the current feature/target construction. Fix target leakage and define a prospective target before making model-quality claims.", ""]
    (REPORTS / "split_evaluation.md").write_text("\n".join(md))
    return report


if __name__ == "__main__":
    result = run()
    print(f"Wrote reports/split_evaluation.json and reports/split_evaluation.md ({len(result['metrics'])} model/split rows).")
    print(result["critical_limitation"])
