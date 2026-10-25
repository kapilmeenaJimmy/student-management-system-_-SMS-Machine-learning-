"""schemas.py — Pydantic v2 contracts for SMS-MH-AVIE V6.

Domains:
  - UCI ML prediction pipeline (StudentFeatureContract)
  - Student roster management (StudentProfile, CreateStudentRequest)
  - Weighted attendance & exam eligibility (WeightedAttendanceRequest, ExamEligibilityResponse)
  - Hybrid academic scoring (AcademicAssessmentInput, AcademicScoreBreakdown)
  - Unified evaluation: SSOT request → composite score + CI + 5-track prescription
  - Feasibility-aware prescription (FeasibilityPrescriptionRequest/Response)
  - Timetable parsing & calendar projection (ScheduleSlot, CalendarEvent, ...)
  - OCR extract result (OCRExtractResult) for in-browser tesseract.js output
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


# ── Student Roster ─────────────────────────────────────────────────────────────

class StudentProfile(BaseModel):
    """Persisted student record stored in students.json."""
    student_id: str = Field(description="UUID or slug, assigned on creation")
    student_name: str = Field(..., min_length=1, max_length=120)
    roll_no: str = Field(default="", max_length=30)
    enrollment_year: int = Field(default=2024, ge=2000, le=2100)
    avatar_color: str = Field(
        default="#00D4FF",
        description="Hex color derived from name hash, used for avatar circle",
    )


class CreateStudentRequest(BaseModel):
    model_config = {"extra": "forbid"}
    student_name: str = Field(..., min_length=1, max_length=120)
    roll_no: str = Field(default="", max_length=30)
    enrollment_year: int = Field(default=2024, ge=2000, le=2100)


# ── UCI ML Prediction ──────────────────────────────────────────────────────────

class StudentFeatureContract(BaseModel):
    """All 36 tabular UCI predictors.

    admission_grade is kept for model compatibility but defaults to 130.0
    so the frontend UI does not expose it.
    """
    model_config = {"extra": "forbid"}

    # Demographics & Background
    marital_status: int = Field(..., ge=1, le=6)
    application_mode: int = Field(..., ge=1, le=57)
    application_order: int = Field(..., ge=0, le=9)
    course: int = Field(..., ge=33, le=9991)
    daytime_evening_attendance: int = Field(..., ge=0, le=1)
    previous_qualification: int = Field(..., ge=1, le=43)
    previous_qualification_grade: float = Field(..., ge=0.0, le=200.0)
    nationality: int = Field(..., ge=1, le=109)
    mothers_qualification: int = Field(..., ge=1, le=44)
    fathers_qualification: int = Field(..., ge=1, le=44)
    mothers_occupation: int = Field(..., ge=0, le=194)
    fathers_occupation: int = Field(..., ge=0, le=194)

    # admission_grade: optional — frontend omits it, model uses 130.0 default
    admission_grade: float = Field(default=130.0, ge=0.0, le=200.0)

    displaced: int = Field(..., ge=0, le=1)
    educational_special_needs: int = Field(..., ge=0, le=1)
    debtor: int = Field(..., ge=0, le=1)
    tuition_fees_up_to_date: int = Field(..., ge=0, le=1)
    gender: int = Field(..., ge=0, le=1)
    scholarship_holder: int = Field(..., ge=0, le=1)
    age_at_enrollment: int = Field(..., ge=15, le=70)
    international: int = Field(..., ge=0, le=1)

    # 1st Semester
    curricular_units_1st_sem_credited: int = Field(..., ge=0, le=20)
    curricular_units_1st_sem_enrolled: int = Field(..., ge=0, le=26)
    curricular_units_1st_sem_evaluations: int = Field(..., ge=0, le=45)
    curricular_units_1st_sem_approved: int = Field(..., ge=0, le=26)
    curricular_units_1st_sem_grade: float = Field(..., ge=0.0, le=20.0)
    curricular_units_1st_sem_without_evaluations: int = Field(..., ge=0, le=12)

    # 2nd Semester
    curricular_units_2nd_sem_credited: int = Field(..., ge=0, le=19)
    curricular_units_2nd_sem_enrolled: int = Field(..., ge=0, le=23)
    curricular_units_2nd_sem_evaluations: int = Field(..., ge=0, le=33)
    curricular_units_2nd_sem_approved: int = Field(..., ge=0, le=20)
    curricular_units_2nd_sem_grade: float = Field(..., ge=0.0, le=20.0)
    curricular_units_2nd_sem_without_evaluations: int = Field(..., ge=0, le=12)

    # Macroeconomic
    unemployment_rate: float = Field(..., ge=-5.0, le=25.0)
    inflation_rate: float = Field(..., ge=-5.0, le=10.0)
    gdp: float = Field(..., ge=-5.0, le=5.0)

    # Study hours (synthesized, shared with academic formula)
    weekly_study_hours: float = Field(default=10.0, ge=1.0, le=45.0)


class RosterPredictRequest(BaseModel):
    """Predict for a named roster student (student_id for audit logging)."""
    student_id: str
    features: StudentFeatureContract


class ConformalInterval(BaseModel):
    lower: float
    upper: float
    coverage: float = 0.95


class ShapAttribution(BaseModel):
    feature: str
    value: float
    impact: float


class PredictionResponse(BaseModel):
    predicted_score: float
    letter_grade: str
    conformal_interval: ConformalInterval
    positive_drivers: List[ShapAttribution]
    risk_drivers: List[ShapAttribution]
    model_type: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    latency_ms: float


# ── Old Compliance (backward-compat, used by existing tests) ───────────────────

class AttendanceRequest(BaseModel):
    sessions_attended: int = Field(..., ge=0)
    sessions_held: int = Field(..., ge=1)

    @model_validator(mode="after")
    def attended_le_held(self) -> "AttendanceRequest":
        if self.sessions_attended > self.sessions_held:
            raise ValueError("sessions_attended must be <= sessions_held")
        return self


class ComplianceResponse(BaseModel):
    ratio: float
    compliant: bool
    safe_skips: Optional[int] = None
    sessions_needed: Optional[int] = None
    message: str


# ── Weighted Attendance & Exam Eligibility ─────────────────────────────────────

class WeightedAttendanceRequest(BaseModel):
    """
    Weighted attendance model.
      1 lecture class  = 1 effective session credit
      1 lab session    = 2 effective session credits
    """
    model_config = {"extra": "forbid"}

    classes_attended: int = Field(..., ge=0)
    classes_held: int = Field(..., ge=0)
    labs_attended: int = Field(..., ge=0)
    labs_held: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> "WeightedAttendanceRequest":
        if self.classes_attended > self.classes_held:
            raise ValueError(
                f"classes_attended ({self.classes_attended}) > classes_held ({self.classes_held})"
            )
        if self.labs_attended > self.labs_held:
            raise ValueError(
                f"labs_attended ({self.labs_attended}) > labs_held ({self.labs_held})"
            )
        total = self.classes_held + 2 * self.labs_held
        if total == 0:
            raise ValueError("Total effective sessions held must be > 0")
        return self


class ExamEligibilityResponse(BaseModel):
    eligible: bool
    status: Literal["ELIGIBLE", "DEBARRED"]
    ratio: float = Field(description="A_eff / T_eff")
    effective_attended: int
    effective_held: int
    safe_skips: Optional[int] = Field(
        default=None,
        description="Whole class sessions that can be skipped while staying eligible",
    )
    sessions_needed: Optional[int] = Field(
        default=None,
        description="Consecutive sessions required to become eligible",
    )
    message: str


# ── Prescription (UCI ML pipeline) ────────────────────────────────────────────

class PrescriptionRequest(BaseModel):
    features: StudentFeatureContract
    target_score: float = Field(..., ge=0.0, le=100.0)


class PrescriptionResponse(BaseModel):
    original_score: float
    target_score: float
    achieved_score: float
    converged: bool
    deltas: dict
    cost: float


# ── Hybrid Academic Scoring ────────────────────────────────────────────────────

class AcademicAssessmentInput(BaseModel):
    model_config = {"extra": "forbid"}

    lab_marks: List[float] = Field(
        default_factory=list,
        description="Marks per lab session, each out of 5",
    )
    n_labs_total: int = Field(default=5, ge=1, le=60)
    midterm_score: float = Field(default=0.0, ge=0.0, le=20.0)
    endsem_score: float = Field(default=0.0, ge=0.0, le=80.0)
    weekly_study_hours: float = Field(default=10.0, ge=0.0, le=45.0)

    @model_validator(mode="after")
    def validate_lab_marks(self) -> "AcademicAssessmentInput":
        for i, m in enumerate(self.lab_marks):
            if not (0.0 <= m <= 5.0):
                raise ValueError(f"lab_marks[{i}]={m} must be in [0, 5]")
        if len(self.lab_marks) > self.n_labs_total:
            raise ValueError(
                f"lab_marks has {len(self.lab_marks)} entries but n_labs_total={self.n_labs_total}"
            )
        return self


class AcademicScoreBreakdown(BaseModel):
    lab_component: float
    midterm_component: float
    endsem_component: float
    study_factor: float
    composite_score: float
    letter_grade: str
    labs_submitted: int
    labs_remaining: int
    n_labs_total: int


# ── Feasibility Prescription ───────────────────────────────────────────────────

class FeasibilityPrescriptionRequest(BaseModel):
    assessment: AcademicAssessmentInput
    target_score: float = Field(..., ge=0.0, le=100.0)


class FeasibilityActionItem(BaseModel):
    field: str
    label: str
    current: float
    required: float
    unit: str


class FeasibilityPrescriptionResponse(BaseModel):
    feasible: bool
    ceiling: float
    target_score: float
    current_score: float
    infeasible_reason: Optional[str] = None
    bottleneck_details: Optional[str] = None
    action_items: List[FeasibilityActionItem] = Field(default_factory=list)
    achieved_score: Optional[float] = None


# ── V6: 5-Track Prescription Engine ───────────────────────────────────────────

TrackNumber = Literal[1, 2, 3, 4, 5]


class PrescriptionTrack(BaseModel):
    """Output of the 5-track prescription engine."""
    track: TrackNumber
    track_label: str
    gap: float = Field(description="target_score - current_score")
    feasible: bool
    ceiling: float
    current_score: float
    target_score: float

    # Track 1: under-aiming
    under_aiming_message: Optional[str] = None
    recommended_target: Optional[float] = None

    # Track 2: safety buffer
    safety_cushion_pts: Optional[float] = None
    maintenance_protocol: Optional[List[str]] = None

    # Track 3: tactical
    study_hours_delta: Optional[float] = None
    required_lab_avg: Optional[float] = None
    tactical_notes: Optional[List[str]] = None

    # Track 4: structural (SLSQP)
    required_endsem: Optional[float] = None
    required_study_hours: Optional[float] = None
    required_lab_avg_remaining: Optional[float] = None
    achieved_score: Optional[float] = None
    action_items: List[FeasibilityActionItem] = Field(default_factory=list)

    # Track 5: infeasible
    infeasible_reason: Optional[str] = None
    bottleneck_details: Optional[str] = None


# ── V6: Unified Evaluation (SSOT) ─────────────────────────────────────────────

class UnifiedEvaluationRequest(BaseModel):
    """
    Single source of truth evaluation request.

    All shared fields (weekly_study_hours, lab_marks, midterm, endsem) are
    sent here. The backend computes the academic score and — if target_score
    is provided — runs the 5-track prescription engine.

    UCI ML pipeline is called internally to fetch the ±q_hat conformal margin.
    """
    model_config = {"extra": "forbid"}

    # SSOT academic fields
    lab_marks: List[float] = Field(default_factory=list)
    n_labs_total: int = Field(default=5, ge=1, le=60)
    midterm_score: float = Field(default=0.0, ge=0.0, le=20.0)
    endsem_score: float = Field(default=0.0, ge=0.0, le=80.0)
    weekly_study_hours: float = Field(default=10.0, ge=0.0, le=45.0)

    # Optional: UCI ML features for conformal interval (all with defaults)
    uci_features: Optional[StudentFeatureContract] = None

    # Optional: prescription target (if None, skip prescription)
    target_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def validate_lab_marks(self) -> "UnifiedEvaluationRequest":
        for i, m in enumerate(self.lab_marks):
            if not (0.0 <= m <= 5.0):
                raise ValueError(f"lab_marks[{i}]={m} must be in [0, 5]")
        if len(self.lab_marks) > self.n_labs_total:
            raise ValueError("More lab_marks entries than n_labs_total")
        return self


class UnifiedEvaluationResponse(BaseModel):
    """
    Unified evaluation response.

    composite_score: academic formula score (shown in Hero Gauge)
    conformal_interval: ±q_hat around composite_score (from UCI ML calibration)
    """
    # Academic score (Hero Gauge value)
    composite_score: float
    letter_grade: str

    # Component breakdown
    lab_component: float
    midterm_component: float
    endsem_component: float
    study_factor: float
    labs_submitted: int
    labs_remaining: int

    # Conformal interval from UCI ML q_hat
    conformal_interval: ConformalInterval
    q_hat: float

    # Ceiling for prescription
    ceiling: float

    # 5-track prescription (present only if target_score was provided)
    prescription: Optional[PrescriptionTrack] = None


# ── Timetable & Calendar ───────────────────────────────────────────────────────

SlotType = Literal["lecture", "lab", "exam", "other"]
EventType = Literal["holiday", "exam", "assignment", "event"]


class ScheduleSlot(BaseModel):
    day: str
    start: str
    end: str
    course_code: str = ""
    title: str = ""
    room: str = ""
    slot_type: SlotType = "lecture"


class CalendarEvent(BaseModel):
    date: str
    title: str
    event_type: EventType = "event"


class TimetableUploadResponse(BaseModel):
    slots: List[ScheduleSlot]
    n_lectures_per_week: int
    n_labs_per_week: int
    holidays: List[CalendarEvent]
    calendar_events: List[CalendarEvent]
    parse_warnings: List[str] = Field(default_factory=list)


class RecurrenceChoice(BaseModel):
    mode: Literal["full_year", "custom"]
    weeks: Optional[int] = Field(default=None, ge=1, le=104)
    start_date: Optional[str] = None

    @model_validator(mode="after")
    def validate_custom_weeks(self) -> "RecurrenceChoice":
        if self.mode == "custom" and self.weeks is None:
            raise ValueError("weeks is required when mode='custom'")
        return self


class DaySlotOccurrence(BaseModel):
    date: str
    slot: ScheduleSlot
    week_number: int
    cancelled: bool = False
    cancellation_reason: Optional[str] = None


class WeekProjection(BaseModel):
    week_number: int
    week_start: str
    week_end: str
    occurrences: List[DaySlotOccurrence]
    holidays_this_week: List[CalendarEvent]


class ProjectedCalendar(BaseModel):
    total_weeks: int
    total_lectures: int
    total_labs: int
    total_cancelled: int
    week_projections: List[WeekProjection]


# ── OCR Extract Result (from in-browser tesseract.js → backend parse) ─────────

class OCRParseRequest(BaseModel):
    """Raw OCR text extracted by tesseract.js in the browser."""
    raw_text: str = Field(..., min_length=1, max_length=50_000)
    source_hint: Literal["attendance", "timetable", "profile", "unknown"] = "unknown"


class OCRExtractResult(BaseModel):
    """Structured data parsed from OCR raw text (Bennett University portal)."""
    # Profile fields
    student_name: Optional[str] = None
    roll_no: Optional[str] = None
    admission_no: Optional[str] = None
    semester: Optional[int] = None
    curriculum_plan: Optional[str] = None

    # Attendance fields
    attendance_pct: Optional[float] = None
    periods_present: Optional[int] = None
    periods_held: Optional[int] = None

    # Parsed timetable slots (from timetable screenshots)
    timetable_slots: List[dict] = Field(default_factory=list)

    # Dates seen in the document
    dates_found: List[str] = Field(default_factory=list)

    # How many patterns matched
    confidence: Literal["high", "medium", "low"] = "low"

    parse_notes: List[str] = Field(default_factory=list)
