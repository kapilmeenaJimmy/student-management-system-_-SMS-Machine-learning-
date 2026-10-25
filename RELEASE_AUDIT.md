# Release preparation audit report

## Scope

Audited the supplied `SMS_final_drat(polishleft)(4).zip` archive as it exists, rather than applying the unrelated Streamlit project assumptions from the source brief. The live archive is a React/Vite frontend and FastAPI backend project (`SMS-MH-AVIE`). No Git repository metadata was present in the extracted project, and no remote was configured.

## Changes made

- Replaced the prior README with a seven-section, evidence-bound release README covering the actual React/FastAPI architecture, UCI data path, ML workflow, split comparison, formulas, SHAP limitations, OS setup, and publication workflow.
- Added `.gitignore`, `CONTRIBUTING.md`, `SECURITY.md`, `RELEASE_CHECKLIST.md`, and this audit report.
- Added `backend/evaluate_splits.py`; it generates `reports/split_evaluation.json` and `reports/split_evaluation.md` from a real run.
- Fixed the cached UCI CSV reader in `backend/dataset.py`: data is written using comma-separated CSV output, so cached reads now use the matching comma delimiter instead of the previous semicolon delimiter.
- Replaced identifying roster seed content and sample OCR/UI placeholders with generic demo values.
- Updated `run.sh` to use a configurable Python 3.11 executable and standard project-local `.venv`, and updated the frontend Docker build to use `npm ci` with the lockfile.
- Removed downloaded raw data, generated model caches, local student roster data, Python virtual environments, Node modules, built frontend assets, caches, and macOS archive metadata from the release tree.

## Commands and outcomes

| Check | Outcome | Notes |
|---|---|---|
| `python -m compileall -q backend tests` | Passed | Syntax compilation completed before cleanup. |
| API health endpoint | Passed | `GET /api/v1/health` returned `{"status":"ok","latency_ms":0.0}` during the test run. |
| `python -m pytest tests/test_suite.py -q --tb=short` | **24 passed** | API was running locally; model cache was regenerated from the available UCI cache for the test session. Runtime data/model files were removed afterward. |
| `python evaluate_splits.py` from `backend/` | Completed | Wrote the JSON and Markdown reports with six model/split results. |
| `npm run build` | Not passed in supplied environment | The included `node_modules` tree was platform-incompatible and lacked `@rollup/rollup-linux-x64-gnu`. |
| `npm ci --ignore-scripts --no-audit --no-fund` | Timed out | Could not complete dependency reinstall in this environment, so a clean frontend production build remains unverified. |
| Manual browser/UI smoke test | Not run | No claim of verified browser behavior is made. |
| Git remote / push | Not performed | The extracted tree was not a Git repository and the owner has not approved publication. |

## Measured split report — caution required

Evaluation environment: Python 3.13.5, scikit-learn 1.8.0, 4,424 rows after preprocessing, seed 42. These versions differ from the pinned backend requirements, so rerun in the target environment before release.

| Split | Model | Train rows | Test rows | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|---:|
| 60:40 | Ridge | 2,654 | 1,770 | 0.0224 | 0.0301 | 1.0000 |
| 60:40 | Random Forest | 2,654 | 1,770 | 0.2626 | 0.5055 | 0.9994 |
| 60:40 | LightGBM | 2,654 | 1,770 | 0.2245 | 0.4517 | 0.9995 |
| 80:20 | Ridge | 3,539 | 885 | 0.0171 | 0.0233 | 1.0000 |
| 80:20 | Random Forest | 3,539 | 885 | 0.2402 | 0.4561 | 0.9995 |
| 80:20 | LightGBM | 3,539 | 885 | 0.1953 | 0.3777 | 0.9997 |

**These scores are not valid evidence of future-outcome generalization.** The current `G_Final` target is calculated from grade/admission columns that are also present in the model features. This is target leakage and likely explains the near-perfect scores. Correct the prediction-time feature set and define a future-observed target before using evaluation metrics as evidence of model quality.

## Verified facts and unresolved risks

### Verified from source or execution

- The archive contains a React/Vite frontend and FastAPI backend, not a Streamlit app.
- The backend has UCI dataset ingestion/cache code, Ridge/Random Forest/optional LightGBM candidate pipelines, five-fold KFold selection by MAE, a 70/15/15 train/calibration/test workflow, SHAP attribution code, attendance utilities, and academic scoring logic.
- The integration suite passed all 24 tests against a locally running API in this environment.
- The source enables wildcard CORS and does not demonstrate an authentication/authorization layer.
- A project-code `LICENSE` file was not present. No licence was chosen or added.

### Owner decisions / planned work

- Choose and add a code licence only after owner approval.
- Verify and record current UCI dataset terms/attribution before public release.
- Correct target leakage and rerun evaluation using a prediction-time-valid feature set and target.
- Reinstall frontend dependencies from the lockfile in a network-enabled clean environment and verify `npm ci` plus `npm run build`.
- Conduct a manual UI smoke test and review uploads, runtime storage, privacy, and retention behavior.
- Restrict CORS and implement authentication/authorization before any internet-facing use with non-demo data.
- Review staged files and approve repository visibility, remote URL, and push explicitly.
