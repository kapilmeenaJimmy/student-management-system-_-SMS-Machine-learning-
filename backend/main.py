"""main.py — FastAPI application for SMS-MH-AVIE V6.

Endpoints:
  GET  /api/v1/health                 — Status + ping latency
  GET  /api/v1/students               — List roster from students.json
  POST /api/v1/students               — Create + persist a student
  DELETE /api/v1/students/{sid}       — Remove a student
  POST /api/v1/predict                — UCI ML demographic risk score
  POST /api/v1/predict/roster         — Predict for named student (audit log)
  POST /api/v1/prescribe              — UCI ML SLSQP counterfactual
  POST /api/v1/compliance             — Simple 75% check (backward-compat)
  POST /api/v1/eligibility            — Weighted attendance exam eligibility
  POST /api/v1/academic/score         — Hybrid academic grading formula
  POST /api/v1/academic/prescribe     — Feasibility-aware prescription
  POST /api/v1/academic/evaluate      — V6 SSOT unified evaluation + 5-track prescription
  POST /api/v1/timetable/upload       — PDF/XLSX/CSV timetable parsing
  POST /api/v1/timetable/project      — Calendar projection
  POST /api/v1/ocr/parse              — Parse raw OCR text from tesseract.js
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dataset import get_feature_columns
from optimizer import (
    compliance_check,
    compute_academic_score,
    compute_ceiling,
    exam_eligibility,
    feasibility_prescribe,
    five_track_prescribe,
    prescribe,
)
from pipeline import compute_shap, ensure_model
from schedule_parser import parse_ocr_text, parse_timetable_file, project_calendar
from schemas import (
    AcademicAssessmentInput,
    AcademicScoreBreakdown,
    AttendanceRequest,
    ComplianceResponse,
    ConformalInterval,
    UnifiedEvaluationRequest,
    UnifiedEvaluationResponse,
    CreateStudentRequest,
    ExamEligibilityResponse,
    FeasibilityPrescriptionRequest,
    FeasibilityPrescriptionResponse,
    HealthResponse,
    OCRExtractResult,
    OCRParseRequest,
    PredictionResponse,
    PrescriptionRequest,
    PrescriptionResponse,
    ProjectedCalendar,
    RecurrenceChoice,
    RosterPredictRequest,
    ShapAttribution,
    StudentFeatureContract,
    StudentProfile,
    TimetableUploadResponse,
    WeightedAttendanceRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Student roster file ───────────────────────────────────────────────────────
STUDENTS_FILE = Path(__file__).parent / "data" / "students.json"

def _load_students() -> List[dict]:
    if not STUDENTS_FILE.exists():
        return []
    try:
        return json.loads(STUDENTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

def _save_students(students: List[dict]) -> None:
    STUDENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STUDENTS_FILE.write_text(
        json.dumps(students, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def _avatar_color(name: str) -> str:
    """Deterministic hex color from name hash."""
    PALETTE = [
        "#00D4FF", "#00C785", "#9D6EFA", "#F59E0B",
        "#F04040", "#FF6B6B", "#4ECDC4", "#45B7D1",
        "#96CEB4", "#FECA57", "#FF9FF3", "#54A0FF",
    ]
    h = sum(ord(c) * (i + 1) for i, c in enumerate(name))
    return PALETTE[h % len(PALETTE)]


# ── Global model state ────────────────────────────────────────────────────────
_pipe = None
_meta: dict = {}


def _letter_grade(score: float) -> str:
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    return "F"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipe, _meta
    logger.info("Loading / training model …")
    _pipe, _meta = ensure_model()
    logger.info("Model ready: %s | q_hat=%.3f", _meta["model_name"], _meta["q_hat"])

    # Seed roster if empty
    if not STUDENTS_FILE.exists() or not _load_students():
        default_student = {
            "student_id": str(uuid.uuid4()),
            "student_name": "Demo Student",
            "roll_no": "DEMO001",
            "enrollment_year": 2025,
            "avatar_color": _avatar_color("Demo Student"),
        }
        _save_students([default_student])
        logger.info("Created generic demo student")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SMS-MH-AVIE API v4",
    description="Student Management System — Multi-Horizon Academic Vulnerability & Intervention Engine",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{ms:.1f}"
    return response


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/v1/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    t0 = time.perf_counter()
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return HealthResponse(status="ok", latency_ms=latency_ms)


# ── Student Roster ────────────────────────────────────────────────────────────

@app.get("/api/v1/students", response_model=List[StudentProfile], tags=["Roster"])
async def list_students():
    return _load_students()


@app.post("/api/v1/students", response_model=StudentProfile, tags=["Roster"])
async def create_student(req: CreateStudentRequest):
    students = _load_students()
    student = {
        "student_id": str(uuid.uuid4()),
        "student_name": req.student_name,
        "roll_no": req.roll_no,
        "enrollment_year": req.enrollment_year,
        "avatar_color": _avatar_color(req.student_name),
    }
    students.append(student)
    _save_students(students)
    logger.info("Created student: %s (%s)", req.student_name, student["student_id"])
    return student


@app.delete("/api/v1/students/{student_id}", tags=["Roster"])
async def delete_student(student_id: str):
    students = _load_students()
    remaining = [s for s in students if s["student_id"] != student_id]
    if len(remaining) == len(students):
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found")
    _save_students(remaining)
    return {"deleted": student_id}


# ── UCI ML Prediction ─────────────────────────────────────────────────────────

@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(features: StudentFeatureContract):
    t0 = time.perf_counter()
    feature_cols = get_feature_columns()
    row_dict = features.model_dump()
    X = pd.DataFrame([row_dict])[feature_cols]

    score = float(np.clip(_pipe.predict(X)[0], 0, 100))
    q = _meta["q_hat"]
    shap_result = compute_shap(_pipe, X, feature_cols)
    latency = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        predicted_score=round(score, 2),
        letter_grade=_letter_grade(score),
        conformal_interval=ConformalInterval(
            lower=round(max(score - q, 0), 2),
            upper=round(min(score + q, 100), 2),
        ),
        positive_drivers=[ShapAttribution(**d) for d in shap_result["positive"]],
        risk_drivers=[ShapAttribution(**d) for d in shap_result["risk"]],
        model_type=_meta.get("model_name", "unknown"),
        latency_ms=round(latency, 1),
    )


@app.post("/api/v1/predict/roster", response_model=PredictionResponse, tags=["Inference"])
async def predict_roster(req: RosterPredictRequest):
    """Predict for a named student; logs student_id for audit trail."""
    t0 = time.perf_counter()
    feature_cols = get_feature_columns()
    row_dict = req.features.model_dump()
    X = pd.DataFrame([row_dict])[feature_cols]

    score = float(np.clip(_pipe.predict(X)[0], 0, 100))
    q = _meta["q_hat"]
    shap_result = compute_shap(_pipe, X, feature_cols)
    latency = (time.perf_counter() - t0) * 1000

    logger.info("Roster predict: student_id=%s score=%.2f", req.student_id, score)
    return PredictionResponse(
        predicted_score=round(score, 2),
        letter_grade=_letter_grade(score),
        conformal_interval=ConformalInterval(
            lower=round(max(score - q, 0), 2),
            upper=round(min(score + q, 100), 2),
        ),
        positive_drivers=[ShapAttribution(**d) for d in shap_result["positive"]],
        risk_drivers=[ShapAttribution(**d) for d in shap_result["risk"]],
        model_type=_meta.get("model_name", "unknown"),
        latency_ms=round(latency, 1),
    )


# ── UCI ML Prescription ───────────────────────────────────────────────────────

@app.post("/api/v1/prescribe", response_model=PrescriptionResponse, tags=["Inference"])
async def prescribe_endpoint(req: PrescriptionRequest):
    feature_cols = get_feature_columns()
    result = prescribe(
        pipe=_pipe,
        feature_row=req.features.model_dump(),
        feature_cols=feature_cols,
        mad=_meta["mad"],
        target_score=req.target_score,
    )
    return PrescriptionResponse(**result)


# ── Attendance: Legacy Compliance ─────────────────────────────────────────────

@app.post("/api/v1/compliance", response_model=ComplianceResponse, tags=["Attendance"])
async def compliance_endpoint(req: AttendanceRequest):
    result = compliance_check(req.sessions_attended, req.sessions_held)
    return ComplianceResponse(**result)


# ── Attendance: Weighted Exam Eligibility ─────────────────────────────────────

@app.post("/api/v1/eligibility", response_model=ExamEligibilityResponse, tags=["Attendance"])
async def eligibility_endpoint(req: WeightedAttendanceRequest):
    """
    Weighted attendance exam eligibility engine.

    1 lecture/class session = 1 effective session credit
    1 lab session           = 2 effective session credits (2 consecutive periods)

    ELIGIBLE  (ratio ≥ 75%): returns safe_skips
    DEBARRED  (ratio < 75%): returns sessions_needed for recovery
    """
    result = exam_eligibility(
        classes_attended=req.classes_attended,
        classes_held=req.classes_held,
        labs_attended=req.labs_attended,
        labs_held=req.labs_held,
    )
    return ExamEligibilityResponse(**result)


# ── Hybrid Academic Score ─────────────────────────────────────────────────────

@app.post("/api/v1/academic/score", response_model=AcademicScoreBreakdown, tags=["Academic"])
async def academic_score(assessment: AcademicAssessmentInput):
    result = compute_academic_score(
        lab_marks=assessment.lab_marks,
        midterm_score=assessment.midterm_score,
        endsem_score=assessment.endsem_score,
        weekly_study_hours=assessment.weekly_study_hours,
        n_labs_total=assessment.n_labs_total,
    )
    return AcademicScoreBreakdown(**result)


# ── Feasibility-Aware Prescription ───────────────────────────────────────────

@app.post(
    "/api/v1/academic/prescribe",
    response_model=FeasibilityPrescriptionResponse,
    tags=["Academic"],
)
async def academic_prescribe(req: FeasibilityPrescriptionRequest):
    a = req.assessment
    result = feasibility_prescribe(
        lab_marks=a.lab_marks,
        midterm_score=a.midterm_score,
        endsem_score=a.endsem_score,
        weekly_study_hours=a.weekly_study_hours,
        n_labs_total=a.n_labs_total,
        target_score=req.target_score,
    )
    return FeasibilityPrescriptionResponse(**result)


# ── Timetable Upload ──────────────────────────────────────────────────────────

@app.post(
    "/api/v1/timetable/upload",
    response_model=TimetableUploadResponse,
    tags=["Timetable"],
)
async def timetable_upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    allowed = {".pdf", ".xlsx", ".xls", ".csv", ".txt"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Accepted: PDF, XLSX, CSV.",
        )
    file_bytes = await file.read()
    result = parse_timetable_file(filename=file.filename, file_bytes=file_bytes)
    return TimetableUploadResponse(**result)


# ── Calendar Projection ───────────────────────────────────────────────────────

@app.post(
    "/api/v1/timetable/project",
    response_model=ProjectedCalendar,
    tags=["Timetable"],
)
async def timetable_project(slots: list, holidays: list, recurrence: RecurrenceChoice):
    result = project_calendar(
        slots=slots,
        holidays=holidays,
        mode=recurrence.mode,
        weeks=recurrence.weeks,
        start_date_str=recurrence.start_date,
    )
    return ProjectedCalendar(**result)


# ── OCR Text Parse ────────────────────────────────────────────────────────────

@app.post("/api/v1/ocr/parse", response_model=OCRExtractResult, tags=["OCR"])
async def ocr_parse(req: OCRParseRequest):
    """
    Parse raw OCR text extracted by tesseract.js in the browser.

    Applies Bennett University portal regex rules to extract:
      - Attendance % and periods present/held
      - Student profile (name, roll no, semester, curriculum plan)
      - Timetable slots with Practical/Lecture detection
    """
    result = parse_ocr_text(raw_text=req.raw_text, source_hint=req.source_hint)
    return OCRExtractResult(**result)


# ── V6: Unified SSOT Evaluation ────────────────────────────────────────────────

@app.post(
    "/api/v1/academic/evaluate",
    response_model=UnifiedEvaluationResponse,
    tags=["Academic"],
)
async def academic_evaluate(req: UnifiedEvaluationRequest):
    """
    V6 Single Source of Truth evaluation endpoint.

    Computes:
    1. Academic formula score (shown in Hero Gauge)
    2. Conformal interval ±q_hat from UCI ML calibration
    3. (Optional) 5-track prescription if target_score provided

    This endpoint is the authoritative source for the Hero Gauge value.
    The UCI ML pipeline only supplies the ±q_hat margin — it does NOT set the score.
    """
    # ── Step 1: Academic score (formula) ──────────────────────────────────────
    score_result = compute_academic_score(
        lab_marks=req.lab_marks,
        midterm_score=req.midterm_score,
        endsem_score=req.endsem_score,
        weekly_study_hours=req.weekly_study_hours,
        n_labs_total=req.n_labs_total,
    )
    composite = score_result["composite_score"]
    ceiling = compute_ceiling(req.lab_marks, req.midterm_score, req.n_labs_total)

    # ── Step 2: UCI conformal interval (q_hat only, score-agnostic) ───────────
    q_hat = float(_meta.get("q_hat", 6.0))

    # Optionally run full UCI ML prediction for accurate conformal bounds
    # Use provided uci_features or fall back to minimal default row
    try:
        from dataset import get_feature_columns
        feature_cols = get_feature_columns()
        if req.uci_features is not None:
            feat_dict = req.uci_features.model_dump()
        else:
            feat_dict = {
                "marital_status": 1, "application_mode": 1, "application_order": 1,
                "course": 9119, "daytime_evening_attendance": 1,
                "previous_qualification": 1, "previous_qualification_grade": 122.0,
                "nationality": 1, "mothers_qualification": 1, "fathers_qualification": 1,
                "mothers_occupation": 5, "fathers_occupation": 5,
                "admission_grade": 130.0, "displaced": 0, "educational_special_needs": 0,
                "debtor": 0, "tuition_fees_up_to_date": 1, "gender": 1,
                "scholarship_holder": 0, "age_at_enrollment": 20, "international": 0,
                "curricular_units_1st_sem_credited": 0, "curricular_units_1st_sem_enrolled": 6,
                "curricular_units_1st_sem_evaluations": 6, "curricular_units_1st_sem_approved": 6,
                "curricular_units_1st_sem_grade": 13.0, "curricular_units_1st_sem_without_evaluations": 0,
                "curricular_units_2nd_sem_credited": 0, "curricular_units_2nd_sem_enrolled": 6,
                "curricular_units_2nd_sem_evaluations": 6, "curricular_units_2nd_sem_approved": 6,
                "curricular_units_2nd_sem_grade": 13.0, "curricular_units_2nd_sem_without_evaluations": 0,
                "unemployment_rate": 10.8, "inflation_rate": 1.4, "gdp": 1.74,
                "weekly_study_hours": req.weekly_study_hours,
            }
        q_hat = float(_meta.get("q_hat", 6.0))
    except Exception:
        q_hat = float(_meta.get("q_hat", 6.0))

    ci_lower = round(max(0.0, composite - q_hat), 2)
    ci_upper = round(min(100.0, composite + q_hat), 2)

    # ── Step 3: 5-track prescription (if requested) ───────────────────────────
    prescription_out = None
    if req.target_score is not None:
        track_result = five_track_prescribe(
            lab_marks=req.lab_marks,
            midterm_score=req.midterm_score,
            endsem_score=req.endsem_score,
            weekly_study_hours=req.weekly_study_hours,
            n_labs_total=req.n_labs_total,
            target_score=req.target_score,
        )
        # Convert action_items dicts to FeasibilityActionItem models
        from schemas import FeasibilityActionItem, PrescriptionTrack
        track_result["action_items"] = [
            FeasibilityActionItem(**item) for item in track_result.get("action_items", [])
        ]
        prescription_out = PrescriptionTrack(**track_result)

    return UnifiedEvaluationResponse(
        composite_score=composite,
        letter_grade=score_result["letter_grade"],
        lab_component=score_result["lab_component"],
        midterm_component=score_result["midterm_component"],
        endsem_component=score_result["endsem_component"],
        study_factor=score_result["study_factor"],
        labs_submitted=score_result["labs_submitted"],
        labs_remaining=score_result["labs_remaining"],
        conformal_interval=ConformalInterval(lower=ci_lower, upper=ci_upper),
        q_hat=q_hat,
        ceiling=ceiling,
        prescription=prescription_out,
    )
