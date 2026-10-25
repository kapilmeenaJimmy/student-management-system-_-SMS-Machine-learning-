"""optimizer.py — Academic scoring, feasibility ceiling, 5-track prescription,
SLSQP counterfactuals, exam eligibility (weighted attendance).

V6 additions:
  - five_track_prescribe(): contextual prescription across 5 performance tracks
  - Track 1: severe under-aiming (target well below baseline)
  - Track 2: safety buffer (target secured, maintenance protocol)
  - Track 3: tactical (Δ ≤ 6 pts, minor study/lab adjustments)
  - Track 4: structural turnaround (SLSQP on endsem + study hours)
  - Track 5: hard infeasible ceiling (mathematically unachievable)
"""
from __future__ import annotations

import logging
import math
from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ── UCI ML Mutable Features ───────────────────────────────────────────────────

MUTABLE_FEATURES = [
    "weekly_study_hours",
    "curricular_units_2nd_sem_evaluations",
    "curricular_units_2nd_sem_approved",
    "tuition_fees_up_to_date",
]

FEATURE_BOUNDS = {
    "weekly_study_hours": (1.0, 45.0),
    "curricular_units_2nd_sem_evaluations": (0, 33),
    "curricular_units_2nd_sem_approved": (0, 20),
    "tuition_fees_up_to_date": (0, 1),
}


# ── Weighted Attendance & Exam Eligibility ────────────────────────────────────

def exam_eligibility(
    classes_attended: int,
    classes_held: int,
    labs_attended: int,
    labs_held: int,
) -> dict:
    """
    Compute exam eligibility using the weighted credit formula.

    Credit weighting:
      1 lecture/class = 1 effective session
      1 lab session   = 2 effective sessions (spans 2 consecutive periods)

    T_eff = classes_held + (2 × labs_held)
    A_eff = classes_attended + (2 × labs_attended)
    R = A_eff / T_eff

    ELIGIBLE  (R ≥ 0.75): safe_skips = floor((A_eff − 0.75 × T_eff) / 0.75)
    DEBARRED  (R < 0.75): sessions_needed = ceil((0.75 × T_eff − A_eff) / 0.25)
    """
    t_eff = classes_held + 2 * labs_held
    a_eff = classes_attended + 2 * labs_attended

    if t_eff == 0:
        return {
            "eligible": False,
            "status": "DEBARRED",
            "ratio": 0.0,
            "effective_attended": 0,
            "effective_held": 0,
            "safe_skips": None,
            "sessions_needed": None,
            "message": "No sessions held yet.",
        }

    ratio = a_eff / t_eff

    if ratio >= 0.75:
        safe_skips = math.floor((a_eff - 0.75 * t_eff) / 0.75)
        return {
            "eligible": True,
            "status": "ELIGIBLE",
            "ratio": round(ratio, 4),
            "effective_attended": a_eff,
            "effective_held": t_eff,
            "safe_skips": safe_skips,
            "sessions_needed": None,
            "message": (
                f"Eligible to sit for exams. "
                f"Attendance: {a_eff}/{t_eff} effective sessions ({ratio*100:.1f}%). "
                f"You can safely miss {safe_skips} more session(s)."
            ),
        }
    else:
        sessions_needed = math.ceil((0.75 * t_eff - a_eff) / 0.25)
        return {
            "eligible": False,
            "status": "DEBARRED",
            "ratio": round(ratio, 4),
            "effective_attended": a_eff,
            "effective_held": t_eff,
            "safe_skips": None,
            "sessions_needed": sessions_needed,
            "message": (
                f"Debarred from exams. "
                f"Attendance: {a_eff}/{t_eff} effective sessions ({ratio*100:.1f}%). "
                f"Attend {sessions_needed} consecutive session(s) to become eligible."
            ),
        }


# ── Backward-Compat Simple Compliance Check ──────────────────────────────────

def compliance_check(sessions_attended: int, sessions_held: int) -> dict:
    """Simple 75% rule (no weighting). Preserved for backward compatibility."""
    ratio = sessions_attended / sessions_held
    if ratio >= 0.75:
        safe_skips = math.floor((sessions_attended - 0.75 * sessions_held) / 0.75)
        return {
            "ratio": round(ratio, 4),
            "compliant": True,
            "safe_skips": safe_skips,
            "sessions_needed": None,
            "message": f"Compliant. You can skip {safe_skips} more session(s).",
        }
    sessions_needed = math.ceil((0.75 * sessions_held - sessions_attended) / 0.25)
    return {
        "ratio": round(ratio, 4),
        "compliant": False,
        "safe_skips": None,
        "sessions_needed": sessions_needed,
        "message": f"Non-compliant. Attend {sessions_needed} consecutive session(s) to recover.",
    }


# ── Hybrid Academic Grading Engine ────────────────────────────────────────────

def _lab_component(lab_marks: List[float], n_labs_total: int) -> float:
    """Lab component out of 20: (Σ lab_marks) / (5 × N_labs) × 20."""
    if n_labs_total <= 0:
        return 0.0
    earned = sum(lab_marks)
    return round(min((earned / (5.0 * n_labs_total)) * 20.0, 20.0), 4)


def _midterm_component(midterm_score: float) -> float:
    return round(float(np.clip(midterm_score, 0.0, 20.0)), 4)


def _endsem_component(endsem_score: float) -> float:
    """End-sem out of 80, scaled to 60: score × 0.75."""
    return round(float(np.clip(endsem_score, 0.0, 80.0)) * 0.75, 4)


def _study_factor(weekly_study_hours: float) -> float:
    """Logarithmic study bonus [0, 5]: min(5, 1.5 × ln(1 + h))."""
    return round(min(5.0, 1.5 * math.log(1.0 + max(0.0, weekly_study_hours))), 4)


def _letter_grade(score: float) -> str:
    if score >= 90: return "O"
    if score >= 80: return "A+"
    if score >= 70: return "A"
    if score >= 60: return "B+"
    if score >= 50: return "B"
    if score >= 40: return "C"
    return "F"


def compute_academic_score(
    lab_marks: List[float],
    midterm_score: float,
    endsem_score: float,
    weekly_study_hours: float,
    n_labs_total: int,
) -> dict:
    """
    Hybrid academic score (0–100):
      Lab (20%) + Midterm (20%) + End-Sem (60%) + Study Bonus (≤5)
    """
    lab_comp = _lab_component(lab_marks, n_labs_total)
    mid_comp = _midterm_component(midterm_score)
    end_comp = _endsem_component(endsem_score)
    study_f = _study_factor(weekly_study_hours)
    raw = lab_comp + mid_comp + end_comp + study_f
    composite = round(float(np.clip(raw, 0.0, 100.0)), 2)
    return {
        "lab_component": lab_comp,
        "midterm_component": mid_comp,
        "endsem_component": end_comp,
        "study_factor": study_f,
        "composite_score": composite,
        "letter_grade": _letter_grade(composite),
        "labs_submitted": len(lab_marks),
        "labs_remaining": max(0, n_labs_total - len(lab_marks)),
        "n_labs_total": n_labs_total,
    }


# ── Feasibility Ceiling ────────────────────────────────────────────────────────

def compute_ceiling(
    lab_marks: List[float],
    midterm_score: float,
    n_labs_total: int,
) -> float:
    """Maximum achievable score given fixed lab/midterm marks."""
    lab_comp = _lab_component(lab_marks, n_labs_total)
    mid_comp = _midterm_component(midterm_score)
    ceiling = lab_comp + mid_comp + 60.0 + _study_factor(45.0)
    return round(float(np.clip(ceiling, 0.0, 100.0)), 2)


# ── V6: 5-Track Prescription Engine ──────────────────────────────────────────

def five_track_prescribe(
    lab_marks: List[float],
    midterm_score: float,
    endsem_score: float,
    weekly_study_hours: float,
    n_labs_total: int,
    target_score: float,
) -> dict:
    """
    5-track contextual prescription engine.

    Δ_gap = target_score − current_score

    Track 1: Δ_gap ≤ −10        → Severe under-aiming warning
    Track 2: −10 < Δ_gap ≤ 0    → Safety buffer + maintenance protocol
    Track 3: 0 < Δ_gap ≤ 6      → Tactical optimization (minor adjustments)
    Track 4: 6 < Δ_gap ≤ ceil   → Structural turnaround (SLSQP)
    Track 5: target > ceiling    → Hard infeasible (mathematically impossible)
    """
    current = compute_academic_score(
        lab_marks, midterm_score, endsem_score, weekly_study_hours, n_labs_total
    )
    current_score = current["composite_score"]
    ceiling = compute_ceiling(lab_marks, midterm_score, n_labs_total)
    gap = round(target_score - current_score, 2)

    base = {
        "gap": gap,
        "feasible": True,
        "ceiling": ceiling,
        "current_score": current_score,
        "target_score": target_score,
        "under_aiming_message": None,
        "recommended_target": None,
        "safety_cushion_pts": None,
        "maintenance_protocol": None,
        "study_hours_delta": None,
        "required_lab_avg": None,
        "tactical_notes": None,
        "required_endsem": None,
        "required_study_hours": None,
        "required_lab_avg_remaining": None,
        "achieved_score": None,
        "action_items": [],
        "infeasible_reason": None,
        "bottleneck_details": None,
    }

    # ── Track 5: Hard infeasible ──────────────────────────────────────────────
    if target_score > ceiling:
        lab_comp = current["lab_component"]
        mid_comp = current["midterm_component"]
        bottleneck_parts = []
        if mid_comp < 20.0:
            bottleneck_parts.append(
                f"Midterm: {midterm_score:.1f}/20 → component {mid_comp:.1f}/20 "
                f"(deficit {20.0-mid_comp:.1f} pts is unrecoverable)"
            )
        if lab_comp < 20.0:
            total_earned = sum(lab_marks)
            max_pos = 5.0 * n_labs_total
            bottleneck_parts.append(
                f"Labs: {total_earned:.1f}/{max_pos:.0f} marks → component "
                f"{lab_comp:.1f}/20 (past sessions cannot be re-attempted)"
            )
        return {**base,
            "track": 5,
            "track_label": "Hard Bottleneck",
            "feasible": False,
            "infeasible_reason": (
                f"Target {target_score:.1f} exceeds the mathematical ceiling {ceiling:.1f}. "
                f"Even with 80/80 on the End-Semester exam and maximum study hours (45 hrs/wk), "
                f"the highest achievable score this term is {ceiling:.1f}."
            ),
            "bottleneck_details": "; ".join(bottleneck_parts) or
                "End-semester and study hour contributions are insufficient to bridge the gap.",
        }

    # ── Track 1: Severe under-aiming ──────────────────────────────────────────
    if gap <= -10.0:
        recommended = round(min(ceiling, current_score + 5.0), 1)
        return {**base,
            "track": 1,
            "track_label": "Severe Under-Aiming",
            "under_aiming_message": (
                f"Your target of {target_score:.1f} is {abs(gap):.1f} points below your current "
                f"academic score of {current_score:.1f}. "
                "Setting targets this far below your momentum risks normalising underperformance. "
                "Academic growth is asymmetric: it is far easier to slip from your current level "
                "than to recover ground lost from deliberate low-targeting."
            ),
            "recommended_target": recommended,
        }

    # ── Track 2: Target secured / safety buffer ────────────────────────────────
    if gap <= 0.0:
        cushion = round(abs(gap), 2)
        # Safe lab avg needed to maintain current score
        fixed_base = current["lab_component"] + current["midterm_component"]
        remaining_budget = current_score - fixed_base - current["endsem_component"] - current["study_factor"]
        safe_lab_avg = round(max(0.0, (sum(lab_marks) / n_labs_total) if n_labs_total > 0 else 0.0), 2)

        maintenance = [
            f"Maintain study hours at ≥ {max(1.0, weekly_study_hours - 2.0):.0f} hrs/week "
            f"(current: {weekly_study_hours:.0f} hrs/week).",
            f"Submit remaining labs with average ≥ {safe_lab_avg:.1f}/5 to preserve lab component.",
            f"Current safety cushion: +{cushion:.1f} pts above target — do not skip End-Semester preparation.",
            "Avoid consecutive session absences to protect 75% eligibility.",
        ]
        return {**base,
            "track": 2,
            "track_label": "Target Secured",
            "safety_cushion_pts": cushion,
            "maintenance_protocol": maintenance,
            "achieved_score": current_score,
        }

    # ── Track 3: Tactical optimization (Δ ≤ 6 pts) ────────────────────────────
    if gap <= 6.0:
        # Minimal study hours increase to close gap
        fixed_base = current["lab_component"] + current["midterm_component"] + current["endsem_component"]
        needed_bonus = target_score - fixed_base
        # study_factor = 1.5 × ln(1 + h) → h = exp(needed/1.5) - 1
        needed_hours = round(min(25.0, math.exp(min(needed_bonus, 5.0) / 1.5) - 1.0), 1)
        study_delta = round(max(0.0, needed_hours - weekly_study_hours), 1)

        labs_remaining = max(0, n_labs_total - len(lab_marks))
        required_avg = 0.0
        if labs_remaining > 0:
            # What lab avg closes the gap if study is already sufficient
            lab_comp_needed = target_score - current["midterm_component"] - current["endsem_component"] - current["study_factor"]
            lab_marks_total_needed = (lab_comp_needed / 20.0) * 5.0 * n_labs_total
            extra_needed = lab_marks_total_needed - sum(lab_marks)
            required_avg = round(min(5.0, max(0.0, extra_needed / labs_remaining)), 2)

        notes = [
            f"Close {gap:.1f} pt gap with minor adjustments — no structural changes needed.",
        ]
        if study_delta > 0.0:
            notes.append(
                f"Increase study hours: {weekly_study_hours:.0f} → {weekly_study_hours + study_delta:.0f} hrs/week "
                f"(+{study_delta:.0f} hrs) for an estimated +{gap:.1f} pt boost."
            )
        if required_avg > 0.0 and labs_remaining > 0:
            notes.append(
                f"Score ≥ {required_avg:.1f}/5 on each of the {labs_remaining} remaining lab sessions."
            )
        notes.append("End-Semester preparation is already on track — no major revision uplift required.")

        return {**base,
            "track": 3,
            "track_label": "Tactical Optimization",
            "study_hours_delta": study_delta,
            "required_lab_avg": required_avg if labs_remaining > 0 else None,
            "tactical_notes": notes,
            "achieved_score": round(min(current_score + gap, ceiling), 2),
        }

    # ── Track 4: Structural turnaround (SLSQP) ────────────────────────────────
    lab_comp = _lab_component(lab_marks, n_labs_total)
    mid_comp = _midterm_component(midterm_score)
    fixed_base = lab_comp + mid_comp

    def score_from_vars(x: np.ndarray) -> float:
        return fixed_base + _endsem_component(x[0]) + _study_factor(x[1])

    def objective(x: np.ndarray) -> float:
        return (
            ((x[0] - endsem_score) / 80.0) ** 2 * 3.0 +
            ((x[1] - weekly_study_hours) / 45.0) ** 2
        )

    x0 = np.array([endsem_score, weekly_study_hours])
    try:
        result = minimize(
            objective, x0,
            method="SLSQP",
            bounds=[(0.0, 80.0), (0.0, 25.0)],
            constraints=[{"type": "ineq", "fun": lambda x: score_from_vars(x) - target_score}],
            options={"maxiter": 500, "ftol": 1e-7},
        )
        x_final = result.x
    except Exception as e:
        logger.warning("Track-4 SLSQP failed: %s", e)
        x_final = x0

    required_endsem = round(float(np.clip(x_final[0], 0.0, 80.0)), 1)
    required_study = round(float(np.clip(x_final[1], 0.0, 25.0)), 1)
    achieved = round(float(np.clip(score_from_vars(x_final), 0.0, 100.0)), 2)

    # Required lab avg for remaining labs (informational)
    labs_remaining = max(0, n_labs_total - len(lab_marks))
    if labs_remaining > 0:
        lab_comp_needed = target_score - mid_comp - _endsem_component(required_endsem) - _study_factor(required_study)
        lab_marks_needed = (lab_comp_needed / 20.0) * 5.0 * n_labs_total
        extra = lab_marks_needed - sum(lab_marks)
        req_lab_avg = round(min(5.0, max(0.0, extra / labs_remaining)), 2)
    else:
        req_lab_avg = None

    action_items = []
    if abs(required_endsem - endsem_score) > 0.5:
        action_items.append({
            "field": "endsem_score",
            "label": "End-Semester Exam",
            "current": endsem_score,
            "required": required_endsem,
            "unit": "/ 80",
        })
    if abs(required_study - weekly_study_hours) > 0.5:
        action_items.append({
            "field": "weekly_study_hours",
            "label": "Weekly Study Hours",
            "current": weekly_study_hours,
            "required": required_study,
            "unit": "hrs/week",
        })

    return {**base,
        "track": 4,
        "track_label": "Structural Turnaround",
        "required_endsem": required_endsem,
        "required_study_hours": required_study,
        "required_lab_avg_remaining": req_lab_avg,
        "achieved_score": achieved,
        "action_items": action_items,
    }


# ── V4 Legacy: feasibility_prescribe (preserved for existing tests) ───────────

def feasibility_prescribe(
    lab_marks: List[float],
    midterm_score: float,
    endsem_score: float,
    weekly_study_hours: float,
    n_labs_total: int,
    target_score: float,
) -> dict:
    """
    Feasibility-aware prescription for hybrid academic formula.
    Preserved for backward compatibility with V4 /academic/prescribe endpoint.
    """
    ceiling = compute_ceiling(lab_marks, midterm_score, n_labs_total)
    current = compute_academic_score(
        lab_marks, midterm_score, endsem_score, weekly_study_hours, n_labs_total
    )
    current_score = current["composite_score"]

    if target_score > ceiling:
        lab_comp = _lab_component(lab_marks, n_labs_total)
        mid_comp = _midterm_component(midterm_score)
        bottleneck_parts = []
        if mid_comp < 20.0:
            bottleneck_parts.append(
                f"Midterm scored {midterm_score:.1f}/20 (component: {mid_comp:.1f}/20; "
                f"deficit of {20.0-mid_comp:.1f} marks cannot be recovered)"
            )
        if lab_comp < 20.0:
            total_earned = sum(lab_marks)
            max_possible = 5.0 * n_labs_total
            bottleneck_parts.append(
                f"Lab component: {total_earned:.1f}/{max_possible:.0f} marks → "
                f"{lab_comp:.1f}/20 (past sessions cannot be re-attempted)"
            )

        return {
            "feasible": False,
            "ceiling": ceiling,
            "target_score": target_score,
            "current_score": current_score,
            "infeasible_reason": (
                f"You cannot achieve this score under current term conditions. "
                f"Maximum possible score: {ceiling:.1f}. "
                f"Deficit in Midterm ({midterm_score:.1f}/20) and Lab component "
                f"cannot be recovered even with 100% on the End-Semester exam (80/80)."
            ),
            "bottleneck_details": "; ".join(bottleneck_parts) or
                "Maximum end-semester and study contributions are insufficient.",
            "action_items": [],
            "achieved_score": None,
        }

    lab_comp = _lab_component(lab_marks, n_labs_total)
    mid_comp = _midterm_component(midterm_score)
    fixed_base = lab_comp + mid_comp

    def score_from_vars(x):
        return fixed_base + _endsem_component(x[0]) + _study_factor(x[1])

    def objective(x):
        return ((x[0] - endsem_score) / 80.0) ** 2 * 3.0 + \
               ((x[1] - weekly_study_hours) / 45.0) ** 2

    x0 = np.array([endsem_score, weekly_study_hours])
    try:
        result = minimize(
            objective, x0,
            method="SLSQP",
            bounds=[(0.0, 80.0), (0.0, 25.0)],
            constraints=[{"type": "ineq", "fun": lambda x: score_from_vars(x) - target_score}],
            options={"maxiter": 500, "ftol": 1e-7},
        )
        x_final = result.x
        converged = result.success
    except Exception as e:
        logger.warning("Feasibility SLSQP failed: %s", e)
        x_final = x0
        converged = False

    required_endsem = round(float(np.clip(x_final[0], 0.0, 80.0)), 1)
    required_study = round(float(np.clip(x_final[1], 0.0, 25.0)), 1)
    achieved = round(float(np.clip(score_from_vars(x_final), 0.0, 100.0)), 2)

    action_items = []
    if abs(required_endsem - endsem_score) > 0.5:
        action_items.append({
            "field": "endsem_score",
            "label": "End-Semester Exam",
            "current": endsem_score,
            "required": required_endsem,
            "unit": "/ 80",
        })
    if abs(required_study - weekly_study_hours) > 0.5:
        action_items.append({
            "field": "weekly_study_hours",
            "label": "Weekly Study Hours",
            "current": weekly_study_hours,
            "required": required_study,
            "unit": "hrs/week",
        })

    return {
        "feasible": True,
        "ceiling": ceiling,
        "target_score": target_score,
        "current_score": current_score,
        "infeasible_reason": None,
        "bottleneck_details": None,
        "action_items": action_items,
        "achieved_score": achieved,
    }


# ── UCI ML Pipeline Prescription ──────────────────────────────────────────────

def prescribe(
    pipe: Pipeline,
    feature_row: dict,
    feature_cols: list,
    mad: dict,
    target_score: float,
) -> dict:
    """MAD-weighted SLSQP to find minimal UCI feature delta to reach target_score."""
    x0_series = pd.Series(feature_row)[feature_cols]
    x0 = x0_series.values.astype(float)
    feature_idx = {col: i for i, col in enumerate(feature_cols)}
    mutable_idx = [feature_idx[f] for f in MUTABLE_FEATURES if f in feature_idx]

    def predict_score(x_vec):
        row = pd.DataFrame([x_vec], columns=feature_cols)
        return float(pipe.predict(row)[0])

    def objective(delta):
        return sum(
            (1.0 / max(mad.get(feature_cols[i_feat], 1.0), 1e-6)) * delta[i_d] ** 2
            for i_d, i_feat in enumerate(mutable_idx)
        )

    def constraint_score(delta):
        x_new = x0.copy()
        for i_d, i_feat in enumerate(mutable_idx):
            x_new[i_feat] = x0[i_feat] + delta[i_d]
        return predict_score(x_new) - target_score

    bounds = [
        (FEATURE_BOUNDS.get(feature_cols[i_feat], (-20, 20))[0] - x0[i_feat],
         FEATURE_BOUNDS.get(feature_cols[i_feat], (-20, 20))[1] - x0[i_feat])
        for i_feat in mutable_idx
    ]

    try:
        result = minimize(
            objective, np.zeros(len(mutable_idx)),
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": constraint_score}],
            options={"maxiter": 500, "ftol": 1e-6},
        )
        delta_final = result.x
        converged = result.success
    except Exception as e:
        logger.warning("Optimizer failed: %s", e)
        delta_final = np.zeros(len(mutable_idx))
        converged = False

    x_new = x0.copy()
    for i_d, i_feat in enumerate(mutable_idx):
        lo, hi = FEATURE_BOUNDS.get(feature_cols[i_feat], (-999, 999))
        x_new[i_feat] = np.clip(x0[i_feat] + delta_final[i_d], lo, hi)

    achieved = predict_score(x_new)
    original = float(pipe.predict(pd.DataFrame([x0], columns=feature_cols))[0])

    deltas = {
        feature_cols[mutable_idx[i_d]]: {
            "from": round(float(x0[mutable_idx[i_d]]), 3),
            "to": round(float(x0[mutable_idx[i_d]] + d), 3),
            "delta": round(float(d), 3),
        }
        for i_d, d in enumerate(delta_final)
        if abs(d) > 1e-4
    }

    return {
        "original_score": round(original, 2),
        "target_score": round(target_score, 2),
        "achieved_score": round(achieved, 2),
        "converged": converged,
        "deltas": deltas,
        "cost": round(float(objective(delta_final)), 4),
    }
