# SMS-MH-AVIE — Student Management System

**Multi-Horizon Academic Vulnerability & Counterfactual Intervention Engine** is an experimental full-stack application combining a React dashboard, FastAPI service, a UCI dataset-backed regression pipeline, attendance utilities, academic-score calculations, timetable parsing, and SHAP-based local explanations.

> **Important limitation:** This is a prototype for learning and experimentation. It is not validated for official grading, admission, exam eligibility, disciplinary action, or other consequential decisions. Do not use real student records unless you have authorization, privacy controls, and an appropriate institutional review.

## Table of contents

- [1. Reproducible ML training and evaluation](#1-reproducible-ml-training-and-evaluation)
- [2. 60:40 versus 80:20 split comparison](#2-6040-versus-8020-split-comparison)
- [3. Dataset provenance, privacy, and licensing](#3-dataset-provenance-privacy-and-licensing)
- [4. Formulas, application logic, and corrections](#4-formulas-application-logic-and-corrections)
- [5. Visualizations and explainability](#5-visualizations-and-explainability)
- [6. macOS, Linux, and Windows setup](#6-macos-linux-and-windows-setup)
- [7. End-to-end reproduction and release status](#7-end-to-end-reproduction-and-release-status)

## Project at a glance

- **Frontend:** React 18, Vite, Tailwind CSS, Framer Motion, Lucide React, Tesseract.js.
- **API:** FastAPI, Pydantic, Uvicorn.
- **ML:** scikit-learn, Ridge, Random Forest, optional LightGBM, SHAP, SciPy.
- **Data:** UCI “Predict Students' Dropout and Academic Success” dataset, downloaded at runtime and cached locally by `backend/dataset.py`.
- **Local development ports:** Vite `5173`, API `8000`, interactive API docs `/docs`.

## Suggested repository identity

Possible GitHub repository names:

1. `sms-mh-avie`
2. `student-academic-risk-dashboard`
3. `academic-vulnerability-intervention-engine`

Suggested description (use only after reviewing the project scope):

> Full-stack student academic analytics prototype with a React dashboard, FastAPI, scikit-learn, UCI dataset ingestion, attendance planning, counterfactual optimization, and SHAP explanations. For transparent experimentation—not validated institutional decisions.

Suggested GitHub topics: `fastapi`, `react`, `vite`, `scikit-learn`, `machine-learning`, `explainable-ai`, `shap`, `student-performance`, `education-analytics`, `academic-risk`, `python`, `typescript` (verify each topic matches the actual code before applying). Do not add `streamlit`, `ednet`, `oulad`, `60gb`, or `production-ready`.

## 1. Reproducible ML training and evaluation

### Data and model workflow

The current implementation is not the Streamlit/synthetic-data project described in some earlier release briefs. This archive contains a React + FastAPI application. `backend/dataset.py` downloads UCI dataset 697 from the official UCI archive, caches it under `backend/data/data.csv`, normalizes its columns, deterministically derives `weekly_study_hours` using NumPy seed `42`, and computes the regression target `G_Final`. If the cache is absent, a working internet connection is required for the download path.

`backend/pipeline.py` builds pipelines with preprocessing and these candidate regressors:

- Ridge (`alpha=10.0`)
- Random Forest (`n_estimators=100`, `random_state=42`, `n_jobs=-1`)
- LightGBM (`n_estimators=150`, `learning_rate=0.05`, `random_state=42`) when LightGBM imports successfully

The current model-selection path uses shuffled five-fold `KFold` with seed `42`, compares mean cross-validated MAE and R², and selects the model with the lowest mean MAE. It then makes a deterministic 70% train / 15% calibration / 15% test partition, fits the selected pipeline on the 70% training partition, estimates a split-conformal interval margin on the calibration partition, and measures empirical coverage on the test partition. The saved model is trained on the 70% partition in this pipeline; do not describe it as refit on all records unless the implementation changes.

Run from the repository root in two terminals after installing dependencies as described in Section 6:

```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

The first API startup calls `ensure_model()` and can download/process the UCI dataset and train/cache the model if model files are absent. Cached model files are runtime artifacts and are excluded from version control by default. Do not load `.joblib`/pickle artifacts from untrusted sources.

### Reproducibility limits

Seeded generation and `random_state=42` make the experiment setup more repeatable. They do **not** guarantee bitwise-identical results across hardware, operating systems, Python/library versions, or parallel execution settings. The dependency file pins Python packages for the backend, but runtime versions and operating-system libraries can still affect results.

### Important validity issue: target leakage

`G_Final` is deterministically computed from semester-grade and admission-grade columns. Those columns also appear among the model inputs returned by `get_feature_columns()`. This overlap is target leakage: the resulting metrics do not establish that the model can predict a genuinely future outcome from information available before that outcome occurs. A valid research version must define the prediction time, remove unavailable/future grade fields from the feature set, and evaluate against an independently observed future target before making performance claims.

## 2. 60:40 versus 80:20 split comparison

The production training path currently uses five-fold cross-validation and a 70:15:15 train/calibration/test partition; it does not itself provide a 60:40 versus 80:20 benchmark. The added diagnostic command `backend/evaluate_splits.py` evaluates the existing candidate pipelines on both 60:40 and 80:20 random train/test splits with seed `42`, and writes its measured results to:

- `reports/split_evaluation.json` — machine-readable results and runtime versions
- `reports/split_evaluation.md` — human-readable table and limitations

Run it from the backend directory:

```bash
python evaluate_splits.py
```

The generated table reports MAE, RMSE, R², train/test row counts, model, split, timestamp, Python version, and scikit-learn version. These are regression metrics; precision, recall, and F1 are not applicable to the current continuous `G_Final` target without defining and validating a separate classification target.

Latest generated diagnostic (Python 3.13.5, scikit-learn 1.8.0, 4,424 preprocessed rows; the report timestamp and exact values are recorded in `reports/split_evaluation.json`):

| Split | Model | Train rows | Test rows | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|---:|
| 60:40 | Ridge | 2,654 | 1,770 | 0.0224 | 0.0301 | 1.0000 |
| 60:40 | Random Forest | 2,654 | 1,770 | 0.2626 | 0.5055 | 0.9994 |
| 60:40 | LightGBM | 2,654 | 1,770 | 0.2245 | 0.4517 | 0.9995 |
| 80:20 | Ridge | 3,539 | 885 | 0.0171 | 0.0233 | 1.0000 |
| 80:20 | Random Forest | 3,539 | 885 | 0.2402 | 0.4561 | 0.9995 |
| 80:20 | LightGBM | 3,539 | 885 | 0.1953 | 0.3777 | 0.9997 |

These near-perfect scores are a warning sign, not a quality claim: they are consistent with the known target leakage. Treat the values as an executed diagnostic only, not as evidence that the model generalizes to real future student outcomes.

**Do not infer that 80:20 is inherently superior.** An 80:20 split has more training rows and fewer test rows than 60:40, but one random split is a limited comparison. More importantly, the current target/input overlap makes both split results exploratory and invalid as evidence of prospective generalization. The report deliberately flags this limitation. No metric values are asserted here until the evaluator has actually been run in the release environment.

## 3. Dataset provenance, privacy, and licensing

### External dataset

- **Dataset:** [UCI Machine Learning Repository — Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- **Dataset DOI:** `10.24432/C5MC89` (as listed in the project documentation; verify against the current UCI record when publishing)
- **Ingestion code:** `backend/dataset.py::_download_dataset()` and `load_dataset()`
- **Caching:** `backend/data/data.csv` is generated at runtime and ignored by Git. The release archive should not contain a copy of the downloaded raw data by default.

This project uses the UCI dataset path in code; it does not establish that EdNet, OULAD, or a 60 GB dataset is used. No EdNet/OULAD integration was found in the audited implementation.

### Derived fields and target

The application adds `weekly_study_hours` synthetically using a seeded function of approved-unit counts, admission grade, and bounded random jitter. It derives `G_Final` from existing grade/admission columns. These are project-generated variables, not independent ground-truth observations. See Section 4 and the target-leakage warning in Section 1.

### Privacy and governance

The public release must not contain personal student records, roll numbers, private timetables, local databases, credentials, or real user screenshots. The checked-in roster has been replaced by a generic demo seed; runtime roster storage is ignored by Git. Use synthetic/de-identified examples for development and review institutional authorization, lawful basis, retention, access controls, and bias risks before any real-data use.

### Licence status

**No licence has been selected yet for this project's code.** Public visibility does not automatically grant permission to reuse, modify, or redistribute this repository. The UCI dataset's terms are separate from the project-code licence; review the current official dataset terms and preserve required attribution before redistribution. Do not assume the project code, model artifacts, or downloaded dataset share one licence. See `RELEASE_CHECKLIST.md` before publication.

## 4. Formulas, application logic, and corrections

The formulas below are taken from `backend/optimizer.py`; they describe this application's current rules, not universal university policy.

### Hybrid academic score

- Lab component (out of 20):

  `min((sum(lab_marks) / (5 × n_labs_total)) × 20, 20)`; returns `0` if `n_labs_total <= 0`.
- Midterm component: `clip(midterm_score, 0, 20)`.
- End-semester component: `clip(endsem_score, 0, 80) × 0.75` (out of 60).
- Study bonus: `min(5, 1.5 × ln(1 + max(0, weekly_study_hours)))`.
- Composite score: `clip(lab_component + midterm_component + endsem_component + study_bonus, 0, 100)`.

The implementation is in `backend/optimizer.py::_lab_component`, `_midterm_component`, `_endsem_component`, `_study_factor`, and `compute_academic_score`. Grade labels are mapped by `_letter_grade`; verify the code before treating these bands as a policy.

### Weighted attendance and eligibility

`backend/optimizer.py::exam_eligibility` treats one lecture as one effective session and one lab as two effective sessions:

- Effective held sessions: `T_eff = classes_held + 2 × labs_held`
- Effective attended sessions: `A_eff = classes_attended + 2 × labs_attended`
- Attendance ratio: `A_eff / T_eff`, when `T_eff > 0`
- The code's eligibility threshold is `75%`; this is configurable only if the implementation is changed and is not a universal academic rule.
- At or above 75%, the current implementation estimates safe skips using `floor((A_eff - 0.75 × T_eff) / 0.75)`.
- Below 75%, it estimates sessions needed using `ceil((0.75 × T_eff - A_eff) / 0.25)`.

The simple backward-compatible check `compliance_check` uses unweighted sessions. Do not confuse it with weighted lab eligibility. The code returns “DEBARRED” for a below-threshold calculation; the application has no authority to make an institution's official eligibility decision.

### ML target and calibration

`backend/dataset.py::compute_targets` computes `G1`, `G2`, and `G_Final`. The `G_Final` target is a deterministic transformation of available grade/admission columns; it is not an official institution grading formula. `backend/pipeline.py::train_and_select` uses absolute calibration residuals to estimate `q_hat`, then reports empirical test coverage. This interval is not a guarantee of calibrated coverage under distribution shift.

### Corrections and limitations

- This repository is a **FastAPI + React** project, not a Streamlit app. Do not use `streamlit run app.py` instructions.
- The model pipeline's current feature list includes variables used to calculate its target. Treat the resulting evaluation metrics as leakage-contaminated until the feature/target design is corrected.
- `backend/main.py` currently enables wildcard CORS and does not implement verified authentication/authorization. Do not expose it publicly with real student data.
- The app's academic scoring formula, 75% threshold, and two-credit lab rule are implementation choices, not validated policy recommendations.

## 5. Visualizations and explainability

The frontend presents interactive academic/attendance panels and uses API responses to display model predictions and prescriptions. The backend includes `backend/evaluation_plots.py` for diagnostic plots and `backend/pipeline.py::compute_shap()` for local SHAP attributions.

SHAP values describe how the fitted model attributes a prediction relative to its explainer/background assumptions. They do not prove causation, fairness, correctness, or that a feature should be used in an educational decision. Because the target/input overlap exists, SHAP can also explain leakage patterns rather than useful prospective signal. The study-hours synthetic field is itself derived from other fields, so its attribution should not be interpreted as a causal effect of study time.

Only add screenshots after generating them from the actual app using demo data. Do not add placeholder images or screenshots containing names, roll numbers, real attendance, or private timetables.

## 6. macOS, Linux, and Windows setup

### Prerequisites

- Git
- Python compatible with the pinned backend dependencies (Python 3.11 is the safest initial target based on the backend Dockerfile; validate other versions before claiming support)
- Node.js 20 and npm (the frontend Dockerfile uses Node 20)
- Internet access for the first UCI dataset download unless you supply an authorized local cache

### macOS / Linux (Bash or Zsh)

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

cd frontend
npm ci
npm run build
cd ..
```

Start the API in terminal 1:

```bash
cd backend
../.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Start the frontend in terminal 2:

```bash
cd frontend
npm run dev
```

### Windows PowerShell

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
Push-Location frontend
npm ci
npm run build
Pop-Location
```

Start the API in terminal 1:

```powershell
Push-Location backend
..\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Start the frontend in terminal 2:

```powershell
Push-Location frontend
npm run dev
```

If PowerShell blocks local virtual-environment activation, use the explicit `..\.venv\Scripts\python.exe` commands above without changing machine-wide execution policy. Open the URL printed by Vite (typically `http://localhost:5173`) and the API docs at `http://localhost:8000/docs`. These are local development URLs, not a claim of public deployment.

### Data, memory, and platform notes

The current UCI archive is much smaller than the 60 GB EdNet dataset mentioned in the earlier brief. The application reads/caches a single tabular dataset; Dask is not used in the current requirements or pipeline. On Apple Silicon, use a compatible Python/NumPy/scikit-learn/LightGBM build and consult dependency build errors rather than assuming a particular Homebrew path. The project has not been verified on every OS or hardware configuration.

## 7. End-to-end reproduction and release status

1. Clone or extract the repository.
2. Create a Python virtual environment and install `backend/requirements.txt`.
3. Run `npm ci` and `npm run build` from `frontend/`.
4. Start the API from `backend/`; on first startup it may download the UCI data and train/cache a model.
5. Confirm that `backend/data/data.csv` and `backend/data/models/` are generated locally. These are ignored runtime artifacts, not committed release inputs.
6. Run the split evaluator from `backend/` with `python evaluate_splits.py`. Inspect `reports/split_evaluation.json` and `reports/split_evaluation.md`, and retain the leakage warning alongside any numbers.
7. Run the integration suite only while the API is running, from the repository root: `python -m pytest tests/test_suite.py -q`.
8. Open the Vite URL and API docs, test with demo data, inspect the attendance outputs and SHAP view, and check browser/network logs. A manual UI smoke test remains to be recorded by the release owner.
9. Review security/privacy risks and choose a code licence before publishing.

### Verified versus not yet verified

- **Inspected in source:** React/Vite frontend, FastAPI backend, UCI dataset download/cache code, three candidate regression pipelines (LightGBM conditional on import), 5-fold KFold selection by MAE, 70/15/15 train/calibration/test partition, SHAP function, attendance and academic logic, and an HTTP integration test suite.
- **Changed for release hygiene:** `.gitignore`, generic demo roster seed behavior, removal of personal identifiers from sample UI/parser text, release checklist, contribution/security guidance, and an exploratory split-evaluation script.
- **Not claimed:** public deployment, a completed manual UI smoke test, fairness validation, institutional approval, a selected project-code licence, or valid prospective model performance.

See [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) for the owner decisions and pre-publication checks. Do not push or create a public release until the owner reviews the complete diff and explicitly approves it.

### Git initialization and publication templates

Run these only after choosing the final repository name, reviewing `git status`, and creating an empty GitHub repository. These commands are templates; they do not publish anything automatically.

**macOS / Linux (Bash or Zsh):**

```bash
git status
git init
git branch -M main
git add .gitignore README.md CONTRIBUTING.md SECURITY.md RELEASE_CHECKLIST.md \
  .env.example PITCH_DECK.md VIVA.md backend frontend tests run.sh docker-compose.yml
# Inspect staged content before committing:
git status --short
git diff --cached --stat
git diff --cached
git commit -m "docs: prepare evidence-based repository release"
# Only after creating the GitHub repository and approving publication:
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
git remote -v
git push -u origin main
```

**Windows PowerShell:**

```powershell
git status
git init
git branch -M main
git add .gitignore README.md CONTRIBUTING.md SECURITY.md RELEASE_CHECKLIST.md .env.example PITCH_DECK.md VIVA.md backend frontend tests run.sh docker-compose.yml
git status --short
git diff --cached --stat
git diff --cached
git commit -m "docs: prepare evidence-based repository release"
# Only after creating the GitHub repository and approving publication:
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
git remote -v
git push -u origin main
```

If `origin` already exists, inspect it with `git remote -v` and do not overwrite it blindly. Authenticate using GitHub's supported browser/Git Credential Manager flow; never put access tokens in commands or files.

**Optional Git LFS:** the default release policy is to regenerate model caches locally and keep `*.joblib`/`*.pkl` out of Git. Only if the owner explicitly chooses to distribute an approved artifact and confirms redistribution rights, install Git LFS and track the specific approved path:

```bash
git lfs install
git lfs track "backend/data/models/student_model.joblib"
git add .gitattributes backend/data/models/student_model.joblib
git status --short
git commit -m "build: track approved model artifact with Git LFS"
```

Do not run that optional block against generated or unreviewed model files. Git LFS does not grant data/model redistribution rights and may have storage/bandwidth limits.
