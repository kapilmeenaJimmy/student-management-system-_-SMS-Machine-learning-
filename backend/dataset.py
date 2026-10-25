"""dataset.py — UCI dataset ingestion with local fallback and target synthesis."""
from __future__ import annotations
import hashlib, os, zipfile, io, logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

UCI_URL = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
DATA_DIR = Path(__file__).parent / "data"
RAW_CSV = DATA_DIR / "data.csv"

CATEGORICAL_COLS = [
    "marital_status", "application_mode", "application_order", "course",
    "daytime_evening_attendance", "previous_qualification", "nationality",
    "mothers_qualification", "fathers_qualification",
    "mothers_occupation", "fathers_occupation",
    "displaced", "educational_special_needs", "debtor",
    "tuition_fees_up_to_date", "gender", "scholarship_holder", "international",
]

NUMERIC_COLS = [
    "previous_qualification_grade", "admission_grade", "age_at_enrollment",
    "curricular_units_1st_sem_credited", "curricular_units_1st_sem_enrolled",
    "curricular_units_1st_sem_evaluations", "curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_grade", "curricular_units_1st_sem_without_evaluations",
    "curricular_units_2nd_sem_credited", "curricular_units_2nd_sem_enrolled",
    "curricular_units_2nd_sem_evaluations", "curricular_units_2nd_sem_approved",
    "curricular_units_2nd_sem_grade", "curricular_units_2nd_sem_without_evaluations",
    "unemployment_rate", "inflation_rate", "gdp",
]

MACRO_COLS = ["unemployment_rate", "inflation_rate", "gdp"]
TARGET_COL = "Target"

# Canonical column name mapping (UCI uses Title Case with spaces)
UCI_RENAME = {
    "Marital status": "marital_status",
    "Application mode": "application_mode",
    "Application order": "application_order",
    "Course": "course",
    "Daytime/evening attendance\t": "daytime_evening_attendance",
    "Daytime/evening attendance": "daytime_evening_attendance",
    "Previous qualification": "previous_qualification",
    "Previous qualification (grade)": "previous_qualification_grade",
    "Nacionality": "nationality",
    "Mother's qualification": "mothers_qualification",
    "Father's qualification": "fathers_qualification",
    "Mother's occupation": "mothers_occupation",
    "Father's occupation": "fathers_occupation",
    "Admission grade": "admission_grade",
    "Displaced": "displaced",
    "Educational special needs": "educational_special_needs",
    "Debtor": "debtor",
    "Tuition fees up to date": "tuition_fees_up_to_date",
    "Gender": "gender",
    "Scholarship holder": "scholarship_holder",
    "Age at enrollment": "age_at_enrollment",
    "International": "international",
    "Curricular units 1st sem (credited)": "curricular_units_1st_sem_credited",
    "Curricular units 1st sem (enrolled)": "curricular_units_1st_sem_enrolled",
    "Curricular units 1st sem (evaluations)": "curricular_units_1st_sem_evaluations",
    "Curricular units 1st sem (approved)": "curricular_units_1st_sem_approved",
    "Curricular units 1st sem (grade)": "curricular_units_1st_sem_grade",
    "Curricular units 1st sem (without evaluations)": "curricular_units_1st_sem_without_evaluations",
    "Curricular units 2nd sem (credited)": "curricular_units_2nd_sem_credited",
    "Curricular units 2nd sem (enrolled)": "curricular_units_2nd_sem_enrolled",
    "Curricular units 2nd sem (evaluations)": "curricular_units_2nd_sem_evaluations",
    "Curricular units 2nd sem (approved)": "curricular_units_2nd_sem_approved",
    "Curricular units 2nd sem (grade)": "curricular_units_2nd_sem_grade",
    "Curricular units 2nd sem (without evaluations)": "curricular_units_2nd_sem_without_evaluations",
    "Unemployment rate": "unemployment_rate",
    "Inflation rate": "inflation_rate",
    "GDP": "gdp",
    "Target": "Target",
}


def _download_dataset() -> pd.DataFrame:
    """Try UCI download; fall back to ucimlrepo package."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists():
        logger.info("Loading dataset from cache: %s", RAW_CSV)
        df = pd.read_csv(RAW_CSV)
        return df

    logger.info("Attempting UCI download …")
    try:
        import urllib.request
        with urllib.request.urlopen(UCI_URL, timeout=30) as r:
            data = r.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
            with z.open(csv_name) as f:
                df = pd.read_csv(f, sep=";")
        df.to_csv(RAW_CSV, index=False)
        logger.info("Dataset cached to %s (%d rows)", RAW_CSV, len(df))
        return df
    except Exception as e:
        logger.warning("Direct download failed (%s), trying ucimlrepo …", e)

    try:
        from ucimlrepo import fetch_ucirepo
        repo = fetch_ucirepo(id=697)
        df = pd.concat([repo.data.features, repo.data.targets], axis=1)
        df.to_csv(RAW_CSV, index=False)
        logger.info("ucimlrepo download OK: %d rows", len(df))
        return df
    except Exception as e2:
        raise RuntimeError(
            f"Cannot obtain dataset. Ensure internet access or place data.csv in backend/data/. Error: {e2}"
        )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename UCI columns to snake_case."""
    df = df.rename(columns=UCI_RENAME)
    # Strip any remaining whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    return df


def synthesize_study_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic study-hours synthesis (seed=42, range 1–45 h/week)."""
    rng = np.random.default_rng(42)
    approved_total = (
        df["curricular_units_1st_sem_approved"].fillna(0)
        + df["curricular_units_2nd_sem_approved"].fillna(0)
    )
    admission_norm = df["admission_grade"].fillna(130.0) / 200.0
    base = 1.0 + (approved_total / approved_total.max()) * 30.0 + admission_norm * 10.0
    jitter = rng.uniform(-2.0, 2.0, size=len(df))
    hours = np.clip(base + jitter, 1.0, 45.0)
    df = df.copy()
    df["weekly_study_hours"] = np.round(hours, 1)
    return df


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute G1, G2, G_Final continuous targets."""
    g1 = (df["curricular_units_1st_sem_grade"] / 20.0) * 100.0
    g2_raw = (
        df["curricular_units_1st_sem_grade"] + df["curricular_units_2nd_sem_grade"]
    ) / 40.0 * 100.0
    g_final = np.clip(
        0.35 * g1
        + 0.50 * (df["curricular_units_2nd_sem_grade"] / 20.0 * 100.0)
        + 0.15 * (df["admission_grade"] / 200.0 * 100.0),
        0, 100,
    )
    df = df.copy()
    df["G1"] = g1.round(2)
    df["G2"] = g2_raw.round(2)
    df["G_Final"] = g_final.round(2)
    return df


def load_dataset() -> pd.DataFrame:
    """Full ingestion pipeline: download → normalize → synthesize → targets."""
    df = _download_dataset()
    df = _normalize_columns(df)
    # Drop rows without grade data
    df = df.dropna(subset=["curricular_units_1st_sem_grade", "curricular_units_2nd_sem_grade"])
    df = synthesize_study_hours(df)
    df = compute_targets(df)
    logger.info("Dataset ready: %d rows, %d columns", len(df), len(df.columns))
    return df


def get_feature_columns() -> list[str]:
    return CATEGORICAL_COLS + NUMERIC_COLS + ["weekly_study_hours"]
