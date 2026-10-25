# Exploratory split evaluation

Generated (UTC): `2026-10-25T19:47:00.445156+00:00`  
Python: `3.13.5` · scikit-learn: `1.8.0`  
Rows after preprocessing: **4424**  
Split seed: `42`

> [!CAUTION]
> This report is exploratory only. The target `G_Final` is derived from grade/admission columns that are also included as model inputs. This is target leakage, so these metrics are not valid evidence of future student-performance generalization.

| Split | Model | Train rows | Test rows | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|---:|
| 60:40 | Ridge | 2654 | 1770 | 0.0224 | 0.0301 | 1.0000 |
| 60:40 | RandomForest | 2654 | 1770 | 0.2626 | 0.5055 | 0.9994 |
| 60:40 | LGBM | 2654 | 1770 | 0.2245 | 0.4517 | 0.9995 |
| 80:20 | Ridge | 3539 | 885 | 0.0171 | 0.0233 | 1.0000 |
| 80:20 | RandomForest | 3539 | 885 | 0.2402 | 0.4561 | 0.9995 |
| 80:20 | LGBM | 3539 | 885 | 0.1953 | 0.3777 | 0.9997 |

## Interpretation

A larger training partition can change estimation variance, but an 80:20 split does not inherently outperform a 60:40 split. Compare these rows only as a diagnostic under the current feature/target construction. Fix target leakage and define a prospective target before making model-quality claims.
