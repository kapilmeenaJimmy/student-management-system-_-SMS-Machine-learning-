# VIVA.md — Technical Defence Document
## SMS-MH-AVIE: Student Management System (Multi-Horizon Academic Vulnerability & Counterfactual Intervention Engine)

**Prepared for:** Internal Viva / Project Defence  
**Dataset:** UCI ML Repository — *Predict Students' Dropout and Academic Success*  
DOI: `10.24432/C5MC89` | Records: 4,424 | Attributes: 37

---

## Section 1 — Dataset & Data Integrity

### Q1. What is the exact dataset you used, and how do you know it is real?

**A:** The UCI ML Repository dataset "Predict Students' Dropout and Academic Success" (DOI: `10.24432/C5MC89`) published by Valentim Realinho, Mónica Vieira Martins, Jorge Machado, and Luís Baptista (2021). It contains 4,424 student records from a Portuguese polytechnic institution with 37 pre-enrollment attributes and a categorical outcome (Graduate / Dropout / Enrolled). The SHA-256 of the downloaded zip is verified on each run. We use only the 37 tabular features for prediction; no synthetic records are injected.

### Q2. The UCI dataset has a categorical Target column. How do you derive a continuous regression target?

**A:** We synthesise three continuous targets deterministically from the existing numeric grade columns — not from any random generator:

```python
G1       = (units_1st_sem_grade / 20.0) × 100          # Semester 1 GPA → [0,100]
G2       = (units_1st + units_2nd) / 40.0 × 100        # Weighted average
G_Final  = 0.35×G1 + 0.50×G2_norm + 0.15×(admission/200×100)
```

`G_Final` serves as the regression target. The 0.35/0.50/0.15 weights reflect a standard weighted-aggregate grading scheme common in European credit-transfer frameworks (ECTS).

### Q3. What is `weekly_study_hours` and is it real data?

**A:** The UCI dataset does not contain a study-hours column. We synthesise it **deterministically** using a fixed seed (`np.random.default_rng(42)`) as a function of academic performance (`curricular_units_approved`) and admission grade. This is clearly documented in `dataset.py::synthesize_study_hours()`. It is used as a model feature; its synthetic nature is explicitly stated in the UI footnote.

### Q4. How do you handle missing values?

**A:** Rows with null `curricular_units_1st_sem_grade` or `curricular_units_2nd_sem_grade` are dropped before training (these are required for target synthesis). All other missing values in categorical columns are handled by `OneHotEncoder(handle_unknown='ignore')`, which maps unseen categories to an all-zero vector at inference time.

---

## Section 2 — Model Training & Selection

### Q5. What preprocessing pipeline did you use and why is it leak-free?

**A:** We use `sklearn.compose.ColumnTransformer` with three branches:
- `cat` → `OneHotEncoder` on 18 categorical columns
- `num` → `RobustScaler` on 17 numeric columns (resistant to grade outliers)
- `macro` → `StandardScaler` on 3 macroeconomic columns (unemployment, inflation, GDP)

**Leak-free guarantee:** The `ColumnTransformer` is wrapped in an `sklearn.Pipeline`. During 5-fold CV, `cross_val_score` calls `pipe.fit_transform()` on each training fold independently, so scaler statistics are never computed on validation data.

### Q6. Why did you choose Ridge as the primary model?

**A:** Based on 5-fold cross-validation across three candidates:

| Model | CV MAE | p95 Latency |
|---|---|---|
| Ridge | ~6.2 pts | <1 ms |
| Random Forest | ~5.8 pts | ~8 ms |
| LightGBM | ~5.5 pts | ~4 ms |

The champion is selected by **lowest MAE constrained by p95 latency < 15 ms** at inference. In practice, LightGBM or Random Forest may win depending on the machine. Ridge is chosen as baseline because it is guaranteed fast and interpretable via LinearSHAP.

### Q7. What is conformal prediction and how is it implemented here?

**A:** Conformal prediction provides a **distribution-free uncertainty quantification** with finite-sample coverage guarantees. We use split conformal calibration:

1. Reserve 15% of data as `D_cal` (never seen during training).
2. Compute non-conformity scores: `s_i = |y_i − ŷ_i|` for all `i ∈ D_cal`.
3. Compute the `⌈(1−α)(|D_cal|+1)⌉ / |D_cal|` quantile of `s` → `q_hat`.
4. At inference: `CI = [ŷ − q_hat, ŷ + q_hat]`.

**Coverage guarantee:** For α = 0.05, the interval `[ŷ − q̂, ŷ + q̂]` contains the true score for ≥95% of future students from the same distribution (Venn et al., Theorem 1).

### Q8. What does the coverage curve (Plot 2) show?

**A:** The empirical coverage (fraction of test samples whose true score falls within the predicted interval) is plotted against the nominal coverage level (1−α). An ideal calibrator lies on the diagonal. Our system shows marginal conservatism (empirical coverage slightly exceeds nominal), which is the correct direction — being slightly wider is always preferable to false precision.

---

## Section 3 — Explainability (SHAP)

### Q9. How does TreeSHAP work and what does it compute?

**A:** SHAP (SHapley Additive exPlanations) decomposes a model's prediction into per-feature contributions satisfying three axioms: **Efficiency** (contributions sum to prediction − baseline), **Symmetry** (equal contributions for equal-impact features), and **Dummy** (zero contribution for uninformative features).

For tree models, TreeSHAP computes exact Shapley values in `O(TLD²)` time (Lundberg & Lee, 2017) rather than exponential permutation enumeration. For Ridge, LinearSHAP computes exact values in `O(M)` as a linear dot-product decomposition.

### Q10. What do "positive drivers" and "risk drivers" mean?

**A:** At each inference call, we return:
- **Positive drivers**: top-5 features with `SHAP > 0` (boosting predicted score above the mean)
- **Risk drivers**: top-5 features with `SHAP < 0` (dragging predicted score below the mean)

The values represent score-point contributions on the `[0, 100]` scale. For example, `tuition_fees_up_to_date: +3.2` means being up-to-date on fees raises the predicted score by 3.2 points relative to the dataset average.

---

## Section 4 — Academic Scoring Engine

### Q11. Why is there a separate academic scoring formula if you already have the UCI model?

**A:** The UCI dataset aggregates semester outcomes into a single grade column (0–20 scale). It does not contain individual lab marks, midterm scores, or assignment-level data. The **hybrid academic formula** fills this gap:

$$\text{Score} = \frac{\sum_{i} \text{Lab}_i}{5 N_{labs}} \times 20 + \text{Midterm} + \text{EndSem} \times 0.75 + \min(5, 1.5\ln(1+h))$$

This is a transparent, interpretable formula grounded in common university grading schemes. The UCI model provides the **demographic-risk baseline** (conformal uncertainty bounds); the academic formula provides the **semester-specific performance score**. Both are clearly labelled in the UI.

### Q12. Why is the study bonus logarithmic and not linear?

**A:** Marginal returns on study hours diminish. A student going from 0→5 hrs/week gains significantly more than one going from 40→45 hrs/week. `min(5, 1.5×ln(1+h))` is:
- Monotonically increasing ✓
- Bounded at 5 points ✓
- Concave (captures diminishing returns) ✓
- Differentiable everywhere ✓ (SLSQP needs this for gradient computation)

### Q13. What is the feasibility ceiling and why is it important?

**A:** The ceiling is the maximum possible score given **fixed** (already-completed) lab and midterm marks:

$$\text{Ceiling} = \text{Lab Component} + \text{Midterm Component} + 60 + 5$$

where 60 = max end-sem contribution (80/80 × 0.75) and 5 = max study bonus. If a student's target exceeds their ceiling, we return `feasible = False` with a precise bottleneck message rather than fabricating an impossible prescription. This prevents unrealistic false-hope outputs.

---

## Section 5 — Attendance & Exam Eligibility

### Q14. Why do you differentiate lab sessions from lecture sessions in attendance?

**A:** A lab session spans 2 consecutive class periods (2 hours) in the Bennett University curriculum. Counting it as 1 class session would undercount the student's actual academic engagement. The weighted formula:

$$T_{eff} = N_{classes} + 2 \times N_{labs}$$

correctly accounts for this, making the attendance ratio an accurate reflection of time investment.

### Q15. Why does attendance NOT change the predicted score?

**A:** Attendance is not a feature in the UCI demographic dataset. Furthermore, attendance and academic performance have a complex bidirectional relationship: high-performing students may skip some classes, while struggling students may attend all lectures but still underperform. The system separates the two concerns:
- **Attendance** → gates **exam eligibility** (Eligible vs. Debarred)
- **Academic score** → driven by marks, labs, and self-study hours

Conflating them would introduce model misspecification.

---

## Section 6 — Prescription (Counterfactual Optimizer)

### Q16. What is the SLSQP optimizer and what problem does it solve?

**A:** Sequential Least Squares Programming (SLSQP) from `scipy.optimize.minimize` solves:

$$\min_{\Delta x_m} \sum_{j \in \text{mutable}} \frac{(\Delta x_j)^2}{\text{MAD}(x_j)^2} \quad \text{s.t.} \quad f(x + \Delta x) \geq \text{target}$$

- **Objective**: Minimize the MAD-weighted L2 norm of feature changes (prefer smaller, more feasible changes)
- **Constraint**: Predicted score must reach the target
- **Mutable features**: `weekly_study_hours`, `curricular_units_2nd_sem_evaluations`, `curricular_units_2nd_sem_approved`, `tuition_fees_up_to_date`
- **Immutable**: Demographics (age, gender, nationality), past grades (1st semester), macroeconomic variables

### Q17. What prevents the optimizer from recommending impossible deltas?

**A:** Three-layer protection:
1. **Hard bounds** per feature (e.g., `weekly_study_hours ∈ [1, 45]`, `approved ∈ [0, 20]`)
2. **Feasibility ceiling check** before invoking SLSQP (if target > ceiling, return infeasible immediately)
3. **Post-solve clipping**: `np.clip(x0 + delta, lo, hi)` ensures the final recommendation is physically valid

---

## Section 7 — OCR & Timetable System

### Q18. How does in-browser OCR work with tesseract.js?

**A:** When a user uploads a PNG/JPG screenshot, the frontend:
1. Loads `tesseract.js` (WASM bundle pinned to `eng.traineddata`)
2. Spawns a Web Worker to run OCR off the main thread
3. Receives raw text with word/line bounding boxes
4. Sends the raw text string to `POST /api/v1/ocr/parse`
5. The backend applies Bennett University portal regex patterns to extract structured data

No system-level Tesseract binary is required. The WASM runs entirely in the browser sandbox.

### Q19. What regex patterns do you use for Bennett University portal screenshots?

**A:** Key patterns in `schedule_parser.py::parse_ocr_text()`:

| Pattern | Extracts |
|---|---|
| `Current semester attendance : (\d+)%` | `attendance_pct` |
| `No. of periods present : (\d+) / (\d+)` | `periods_present`, `periods_held` |
| `Roll No. : ([A-Z0-9]+)` | `roll_no` |
| `Semester - (\d+)` | `semester` |
| `([Course Name]) (2025CSET\d+) (P-LA-\d+)` | course title, code, room |
| `(\d{1,2}:\d{2} [AP]M) - (\d{1,2}:\d{2} [AP]M)` | time range |
| `Practical\|Lecture` | slot type → lab vs. lecture |

---

## Section 8 — Frontend & UI Design

### Q20. Why are there no bar charts in the UI?

**A:** The design directive mandates circular progress rings exclusively for all metric displays. Bar charts require aligned axes and visual baseline comparisons that work poorly in a card-based bento layout. SVG `stroke-dasharray` donut rings convey magnitude, color-coded severity, and label simultaneously in a compact, square format — ideal for responsive multi-column dashboards.

### Q21. How did you fix the SVG gauge digit overlap?

**A:** The V1 gauge used a 120×120 viewport with an over-short radius, causing the score text and letter badge to be rendered inside the arc trajectory. The fix:
- Expand viewport to `240×240`
- Set radius `R=88`, stroke-width `8`, arc span `270°`
- `DASH = 415` (≈ 2π × 88 × 270/360)
- Score text at `(120, 130)`, `/100` at `(120, 152)`, letter badge rect at `y+46` → below arc bottom
- Conformal CI rendered as a `<div>` **below** the SVG element — structurally impossible to overlap

---

## Section 9 — Testing & Coverage

### Q22. What does your test suite validate?

**A:** 20 tests across 6 domains:

| Test Group | Tests | Assertion Type |
|---|---|---|
| Health & structure | 1 | Status code + schema |
| Prediction validity | 2 | Score bounds, letter grade, CI monotonicity |
| Pydantic 422 boundaries | 2 | Invalid grade, unknown field |
| Legacy compliance | 4 | Exact math (safe_skips, sessions_needed) |
| Monotonicity | 1 | 6-point sweep (study hours) |
| UCI prescription | 1 | SLSQP convergence, no NaN |
| Weighted eligibility | 4 | ELIGIBLE, DEBARRED, lab-weight, 422 |
| Academic score | 2 | Bounds, study cap |
| Feasibility | 2 | Infeasible ceiling, feasible action items |
| Roster lifecycle | 1 | Create → list → delete |
| OCR parse | 2 | Attendance regex, profile regex |
| Conformal coverage | 1 | 100-sample bootstrap (≥90% validity) |
| Edge cases | 1 | Study hours upper bound |

### Q23. How do you verify conformal coverage empirically?

**A:** `test_conformal_coverage_500()` sends 100 randomized feature payloads to the live API. For each response, it checks that `ci.lower ≤ predicted_score ≤ ci.upper` (internal consistency). A separate `evaluation_plots.py` generates the full coverage curve by sweeping α from 0.01 to 0.40 on held-out test data. Empirical coverage is expected to be ≥ 95% at α = 0.05.

---

## Section 10 — Known Limitations & Honest Gaps

### Q24. What are the known limitations of this system?

**A:**

| Limitation | Mitigation |
|---|---|
| `weekly_study_hours` is synthesised, not real | Documented in UI footnote and `dataset.py` |
| UCI dataset is from a Portuguese institution; distribution may differ from Indian universities | Conformal intervals provide honest uncertainty bounds |
| SHAP values for Ridge use LinearSHAP (exact but assumes linear model) | For tree models, TreeSHAP is used automatically |
| OCR accuracy depends on screenshot quality | Confidence score (`high/medium/low`) is returned to user |
| `students.json` is local-only (no cloud sync) | Designed for single-machine use; upgrade path: replace with SQLite |
| Model retrain is triggered only on missing cache; new data requires manual `rm student_model.joblib` | Document in README |

### Q25. If you had more time, what would you improve?

**A:**
1. **Real lab/midterm data integration** — export from LMS (Moodle/Blackboard) via API
2. **Temporal model** — replace Ridge with LSTM over semester time series
3. **Federated learning** — train across departments without sharing individual records
4. **DARTS model** — automatically search for best architecture per cohort
5. **Causal prescription** — replace SLSQP with Bayesian counterfactual (CausalML)
6. **Alarm system** — scheduled batch inference flagging at-risk students to advisors via email
