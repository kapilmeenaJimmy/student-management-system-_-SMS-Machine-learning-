"""schedule_parser.py — Unified timetable + OCR text parser.

Handles:
  - PDF  (pdfplumber — tabular & grid layouts, text fallback)
  - XLSX (openpyxl  — tabular & grid layouts)
  - CSV  (stdlib csv)
  - Raw OCR text (from in-browser tesseract.js WASM) via parse_ocr_text()

Bennett University portal screenshot regex patterns (parse_ocr_text):
  - Attendance: "Current semester attendance : 85%"
                "No. of periods present : 42 / 56"
  - Profile:    "Roll No. : DEMO-001"
                "Admission No. : 2025XXXXXX"
                "Semester - 1"
                "Curriculum Plan : B.Tech CSE AI"
  - Timetable:  "22-Sep-2026"
                "Design and Analysis of Algorithms (2025CSET205) (P-LA-312)"
                "09:00 AM - 11:00 AM (120 min) Dr. Priya Sharma"
                slot_type: "Practical" → lab, "Lecture" → lecture
"""
from __future__ import annotations

import csv
import io
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_ABBREV = {
    "mon": "Monday", "tue": "Tuesday", "wed": "Wednesday",
    "thu": "Thursday", "fri": "Friday", "sat": "Saturday", "sun": "Sunday",
    "m": "Monday", "t": "Tuesday", "w": "Wednesday", "th": "Thursday",
    "f": "Friday", "s": "Saturday", "su": "Sunday",
}
LAB_KEYWORDS = {"lab", "practical", "prac", "workshop", "tutorial"}
EXAM_KEYWORDS = {"exam", "test", "midterm", "endsem", "end-sem", "final", "quiz", "viva"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_time(t: str) -> str:
    t = t.strip().upper().replace(".", ":").replace(" ", "")
    is_pm = t.endswith("PM")
    is_am = t.endswith("AM")
    t = t.replace("AM", "").replace("PM", "")
    if ":" not in t:
        t += ":00"
    parts = t.split(":")
    hour, minute = int(parts[0]), int(parts[1][:2]) if len(parts) > 1 else 0
    if is_pm and hour != 12:
        hour += 12
    if is_am and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute:02d}"


def _normalise_day(day: str) -> str:
    d = day.strip().lower().rstrip(".")
    for full in DAYS:
        if d == full.lower():
            return full
    return DAY_ABBREV.get(d, day.strip().title())


def _detect_slot_type(text: str) -> str:
    t = text.lower()
    if any(k in t for k in EXAM_KEYWORDS):
        return "exam"
    if any(k in t for k in LAB_KEYWORDS):
        return "lab"
    return "lecture"


def _extract_course_code(text: str) -> str:
    match = re.search(r"\b([A-Z0-9]{2,5}\s*\d{2,4}[A-Z]?)\b", text)
    return match.group(1).replace(" ", "") if match else ""


def _is_holiday_row(text: str) -> bool:
    kw = {"holiday", "no class", "break", "vacation", "public", "independence",
          "republic", "festival", "diwali", "christmas", "eid", "pongal", "holi"}
    t = text.lower()
    return any(k in t for k in kw)


def _parse_date_flex(date_str: str) -> str:
    date_str = date_str.strip()
    for fmt in ["%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d %b %Y",
                "%d %B %Y", "%B %d %Y", "%b %d %Y", "%d-%b-%Y",
                "%d-%b-%y", "%d/%b/%Y"]:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


def _build_slot(day: str, start: str, end: str, subject: str, room: str = "") -> dict:
    slot_type = _detect_slot_type(subject)
    code = _extract_course_code(subject)
    title = subject
    if code:
        title = re.sub(re.escape(code), "", subject, flags=re.IGNORECASE).strip(" -–:|")
    return {
        "day": _normalise_day(day),
        "start": _normalise_time(start),
        "end": _normalise_time(end),
        "course_code": code,
        "title": title.strip(),
        "room": room.strip(),
        "slot_type": slot_type,
    }


def _deduplicate(slots: List[dict]) -> List[dict]:
    seen = set()
    unique = []
    for s in slots:
        key = (s["day"], s["start"], s["end"], s["title"], s["slot_type"])
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return sorted(unique, key=lambda x: (
        DAYS.index(x["day"]) if x["day"] in DAYS else 99,
        x["start"],
    ))


# ── Bennett University Portal OCR Text Parser ─────────────────────────────────

def parse_ocr_text(raw_text: str, source_hint: str = "unknown") -> dict:
    """
    Parse raw OCR text from Bennett University portal screenshots.

    Handles three document types (auto-detected or hinted via source_hint):
      - 'attendance': extracts attendance %, periods_present, periods_held
      - 'profile':    extracts student_name, roll_no, admission_no, semester, curriculum_plan
      - 'timetable':  extracts schedule slots with dates, times, course codes, slot types
      - 'unknown':    tries all patterns
    """
    notes = []
    matched = 0
    result = {
        "student_name": None,
        "roll_no": None,
        "admission_no": None,
        "semester": None,
        "curriculum_plan": None,
        "attendance_pct": None,
        "periods_present": None,
        "periods_held": None,
        "timetable_slots": [],
        "dates_found": [],
        "confidence": "low",
        "parse_notes": [],
    }

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # ── Attendance patterns ──────────────────────────────────────────────────
    # "Current semester attendance : 85%"
    m = re.search(
        r"(?:current\s+semester\s+attendance|overall\s+percentage)\s*[:\-]\s*(\d{1,3}(?:\.\d+)?)\s*%",
        text, re.IGNORECASE,
    )
    if m:
        result["attendance_pct"] = float(m.group(1))
        matched += 1
        notes.append(f"Attendance % extracted: {result['attendance_pct']}")

    # "No. of periods present : 42 / 56"
    m = re.search(
        r"(?:no\.?\s+of\s+periods?\s+present|periods?\s+present)\s*[:\-]\s*(\d+)\s*/\s*(\d+)",
        text, re.IGNORECASE,
    )
    if m:
        result["periods_present"] = int(m.group(1))
        result["periods_held"] = int(m.group(2))
        matched += 1
        notes.append(f"Periods: {result['periods_present']}/{result['periods_held']}")

    # ── Profile patterns ─────────────────────────────────────────────────────
    # "Roll No. : DEMO-001"
    m = re.search(r"roll\s*no\.?\s*[:\-]\s*([A-Z0-9]+)", text, re.IGNORECASE)
    if m:
        result["roll_no"] = m.group(1).strip()
        matched += 1
        notes.append(f"Roll No: {result['roll_no']}")

    # "Admission No. : 2025XXXXXX"
    m = re.search(r"admission\s*no\.?\s*[:\-]\s*(\d{6,12})", text, re.IGNORECASE)
    if m:
        result["admission_no"] = m.group(1).strip()
        matched += 1

    # Student name heuristic: 2-3 capitalised words after "Name :"
    m = re.search(
        r"(?:student\s+)?name\s*[:\-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
        text, re.IGNORECASE,
    )
    if m:
        result["student_name"] = m.group(1).strip()
        matched += 1
        notes.append(f"Name: {result['student_name']}")

    # "Semester - 1" or "Semester: 3"
    m = re.search(r"semester\s*[-:\s]+(\d{1,2})", text, re.IGNORECASE)
    if m:
        result["semester"] = int(m.group(1))
        matched += 1

    # "Curriculum Plan : B.Tech CSE AI"
    m = re.search(
        r"curriculum\s+plan\s*[:\-]\s*([^\n]{4,80})",
        text, re.IGNORECASE,
    )
    if m:
        result["curriculum_plan"] = m.group(1).strip()
        matched += 1

    # ── Timetable patterns ───────────────────────────────────────────────────
    # Date blocks: "22-Sep-2026", "23-Sep-2026"
    date_pat = re.compile(
        r"\b(\d{1,2}-[A-Za-z]{3}-\d{2,4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})\b"
    )
    dates_raw = date_pat.findall(text)
    result["dates_found"] = [_parse_date_flex(d) for d in dates_raw]
    if dates_raw:
        matched += 1

    # Course + time block:
    # "Design and Analysis of Algorithms (2025CSET205) (P-LA-312)"
    # "09:00 AM - 11:00 AM (120 min) Dr. Priya Sharma"
    # Optionally preceded by Practical / Lecture keyword
    course_block_pat = re.compile(
        r"(?P<type>Practical|Lecture)?\s*"
        r"(?P<name>[A-Za-z][A-Za-z\s&,]{3,60}?)\s*"
        r"\(\s*(?P<code>[A-Z0-9]{6,15})\s*\)\s*"
        r"(?:\(\s*(?P<room>[A-Z0-9\-]+)\s*\))?\s*"
        r"(?P<start>\d{1,2}:\d{2}\s*[AP]M)\s*-\s*(?P<end>\d{1,2}:\d{2}\s*[AP]M)",
        re.IGNORECASE | re.DOTALL,
    )
    slots_found = []
    for i, m in enumerate(course_block_pat.finditer(text)):
        g = m.groupdict()
        slot_type_raw = (g["type"] or "").lower()
        if "practical" in slot_type_raw:
            slot_type = "lab"
        elif "lecture" in slot_type_raw:
            slot_type = "lecture"
        else:
            slot_type = _detect_slot_type(g["name"])

        # Use date[i] if available, else first date
        date = result["dates_found"][i] if i < len(result["dates_found"]) else (
            result["dates_found"][0] if result["dates_found"] else ""
        )
        # Determine day name from date
        day_name = "Monday"
        if date:
            try:
                from datetime import date as dt_date
                d = dt_date.fromisoformat(date)
                day_name = DAYS[d.weekday()]
            except Exception:
                pass

        slots_found.append({
            "day": day_name,
            "start": _normalise_time(g["start"]),
            "end": _normalise_time(g["end"]),
            "course_code": (g["code"] or "").strip(),
            "title": (g["name"] or "").strip(),
            "room": (g["room"] or "").strip(),
            "slot_type": slot_type,
            "date": date,
        })
        matched += 1

    result["timetable_slots"] = slots_found
    if slots_found:
        notes.append(f"{len(slots_found)} timetable slot(s) extracted")

    # Confidence scoring
    if matched >= 4:
        result["confidence"] = "high"
    elif matched >= 2:
        result["confidence"] = "medium"
    else:
        result["confidence"] = "low"
        notes.append("Few patterns matched. Check image quality or portal layout.")

    result["parse_notes"] = notes
    return result


# ── CSV Parser ────────────────────────────────────────────────────────────────

def _parse_csv(content: str) -> Tuple[List[dict], List[dict], List[str]]:
    slots, events, warnings = [], [], []
    try:
        reader = csv.DictReader(io.StringIO(content.strip()))
        for i, row in enumerate(reader):
            try:
                row = {k.strip().lower(): (v or "").strip() for k, v in row.items()}
                day = row.get("day", "")
                start = row.get("start", row.get("from", ""))
                end = row.get("end", row.get("to", ""))
                subject = row.get("subject", row.get("course", row.get("title", "")))
                room = row.get("room", row.get("venue", row.get("location", "")))
                date_col = row.get("date", "")
                if date_col and _is_holiday_row(subject):
                    events.append({"date": _parse_date_flex(date_col),
                                   "title": subject, "event_type": "holiday"})
                    continue
                if not day or not start or not end:
                    warnings.append(f"Row {i+2}: missing day/start/end — skipped.")
                    continue
                slot = _build_slot(day, start, end, subject, room)
                if "slot_type" in row and row["slot_type"] in ("lecture", "lab", "exam", "other"):
                    slot["slot_type"] = row["slot_type"]
                slots.append(slot)
            except Exception as e:
                warnings.append(f"Row {i+2}: {e} — skipped.")
    except Exception as e:
        warnings.append(f"CSV parse error: {e}")
    return _deduplicate(slots), events, warnings


# ── XLSX Parser ───────────────────────────────────────────────────────────────

def _parse_xlsx(file_bytes: bytes) -> Tuple[List[dict], List[dict], List[str]]:
    try:
        import openpyxl
    except ImportError:
        return [], [], ["openpyxl not installed"]

    slots, events, warnings = [], [], []
    try:
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
    except Exception as e:
        return [], [], [f"Failed to open XLSX: {e}"]

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        sheet_lower = sheet_name.lower()

        if any(k in sheet_lower for k in ("calendar", "holiday", "event")):
            for row in ws.iter_rows(min_row=2, values_only=True):
                if not row or all(v is None for v in row):
                    continue
                cells = [str(c).strip() if c else "" for c in row]
                if cells[0] and len(cells) > 1:
                    events.append({
                        "date": _parse_date_flex(cells[0]),
                        "title": cells[1],
                        "event_type": "holiday" if _is_holiday_row(cells[1]) else "event",
                    })
            continue

        rows = list(ws.iter_rows(min_row=1, values_only=True))
        if not rows:
            continue

        header = [str(c).strip().lower() if c else "" for c in rows[0]]
        day_col = next((i for i, h in enumerate(header) if "day" in h), None)
        start_col = next((i for i, h in enumerate(header) if "start" in h or h == "from"), None)
        end_col = next((i for i, h in enumerate(header) if "end" in h or h == "to"), None)
        subj_col = next((i for i, h in enumerate(header)
                         if any(k in h for k in ("subject", "course", "title", "module"))), None)
        room_col = next((i for i, h in enumerate(header)
                         if any(k in h for k in ("room", "venue", "location"))), None)

        if day_col is not None and start_col is not None and end_col is not None and subj_col is not None:
            for ri, row in enumerate(rows[1:], 2):
                cells = [str(c).strip() if c else "" for c in row]
                day = cells[day_col] if day_col < len(cells) else ""
                start = cells[start_col] if start_col < len(cells) else ""
                end = cells[end_col] if end_col < len(cells) else ""
                subject = cells[subj_col] if subj_col < len(cells) else ""
                room = cells[room_col] if (room_col is not None and room_col < len(cells)) else ""
                if not day or not start or not end or not subject:
                    continue
                try:
                    slots.append(_build_slot(day, start, end, subject, room))
                except Exception as e:
                    warnings.append(f"Sheet '{sheet_name}' row {ri}: {e}")
        else:
            # Grid layout
            day_headers = [_normalise_day(str(c or "").strip()) for c in rows[0][1:]]
            for row in rows[1:]:
                cells = [str(c or "").strip() for c in row]
                if not cells or not cells[0]:
                    continue
                tm = re.search(
                    r"(\d{1,2}[:.]\d{2}\s*(?:AM|PM)?)\s*[-–]+\s*(\d{1,2}[:.]\d{2}\s*(?:AM|PM)?)",
                    cells[0], re.IGNORECASE,
                )
                if not tm:
                    continue
                for col_i, day in enumerate(day_headers, 1):
                    if not day or col_i >= len(cells) or not cells[col_i] or cells[col_i] in ("-", "--"):
                        continue
                    try:
                        slots.append(_build_slot(day, tm.group(1), tm.group(2), cells[col_i]))
                    except Exception as e:
                        warnings.append(f"Grid [{day}]: {e}")

    return _deduplicate(slots), events, warnings


# ── PDF Parser ────────────────────────────────────────────────────────────────

def _parse_pdf(file_bytes: bytes) -> Tuple[List[dict], List[dict], List[str]]:
    try:
        import pdfplumber
    except ImportError:
        return [], [], ["pdfplumber not installed"]

    slots, events, warnings = [], [], []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        if not table:
                            continue
                        header_row = [str(c or "").strip().lower() for c in table[0]]
                        day_col = next((i for i, h in enumerate(header_row) if "day" in h), None)
                        start_col = next((i for i, h in enumerate(header_row) if "start" in h), None)
                        end_col = next((i for i, h in enumerate(header_row) if "end" in h), None)
                        subj_col = next((i for i, h in enumerate(header_row)
                                         if any(k in h for k in ("subject", "course", "title"))), None)
                        room_col = next((i for i, h in enumerate(header_row)
                                         if any(k in h for k in ("room", "venue"))), None)

                        if day_col is not None and start_col is not None and end_col is not None:
                            for row in table[1:]:
                                cells = [str(c or "").strip() for c in row]
                                day = cells[day_col] if day_col < len(cells) else ""
                                start = cells[start_col] if start_col < len(cells) else ""
                                end = cells[end_col] if end_col < len(cells) else ""
                                subject = cells[subj_col] if (subj_col is not None and subj_col < len(cells)) else ""
                                room = cells[room_col] if (room_col is not None and room_col < len(cells)) else ""
                                if not day or not start or not subject:
                                    continue
                                try:
                                    slots.append(_build_slot(day, start, end or start, subject, room))
                                except Exception as e:
                                    warnings.append(f"PDF p{page_num}: {e}")
                        else:
                            # Grid layout
                            day_headers = [_normalise_day(str(c or "").strip()) for c in table[0][1:]]
                            for row in table[1:]:
                                cells = [str(c or "").strip() for c in row]
                                if not cells:
                                    continue
                                tm = re.search(
                                    r"(\d{1,2}[:.]\d{2}\s*(?:AM|PM)?)\s*[-–]+\s*(\d{1,2}[:.]\d{2}\s*(?:AM|PM)?)",
                                    cells[0], re.IGNORECASE,
                                )
                                if not tm:
                                    continue
                                for col_i, day in enumerate(day_headers, 1):
                                    if not day or col_i >= len(cells) or not cells[col_i]:
                                        continue
                                    try:
                                        slots.append(_build_slot(day, tm.group(1), tm.group(2), cells[col_i]))
                                    except Exception as e:
                                        warnings.append(f"PDF grid [{day}]: {e}")
                else:
                    # Text fallback
                    text = page.extract_text() or ""
                    pat = re.compile(
                        r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
                        r"\s+(\d{1,2}[:.]\d{2})\s+(\d{1,2}[:.]\d{2})"
                        r"\s+([A-Z]{2,5}\s*\d{2,4}[A-Z]?)\s+(.+?)(?:\s{2,}|\n|$)",
                        re.IGNORECASE,
                    )
                    for m in pat.finditer(text):
                        try:
                            slots.append(_build_slot(m.group(1), m.group(2), m.group(3),
                                                     f"{m.group(4)} {m.group(5)}".strip()))
                        except Exception as e:
                            warnings.append(f"PDF text regex p{page_num}: {e}")

    except Exception as e:
        warnings.append(f"PDF error: {e}")

    return _deduplicate(slots), events, warnings


# ── Public API ────────────────────────────────────────────────────────────────

def parse_timetable_file(filename: str, file_bytes: bytes) -> dict:
    """Entry point: auto-detect format from filename extension and parse."""
    ext = Path(filename).suffix.lower()
    warnings: List[str] = []

    if ext in (".xlsx", ".xls"):
        slots, events, warnings = _parse_xlsx(file_bytes)
    elif ext == ".pdf":
        slots, events, warnings = _parse_pdf(file_bytes)
    elif ext in (".csv", ".txt"):
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1")
        slots, events, warnings = _parse_csv(text)
    else:
        return {
            "slots": [], "n_lectures_per_week": 0, "n_labs_per_week": 0,
            "holidays": [], "calendar_events": [],
            "parse_warnings": [f"Unsupported format: '{ext}'. Use PDF, XLSX, or CSV."],
        }

    n_lectures = sum(1 for s in slots if s["slot_type"] == "lecture")
    n_labs = sum(1 for s in slots if s["slot_type"] == "lab")
    holidays = [e for e in events if e["event_type"] == "holiday"]
    other_events = [e for e in events if e["event_type"] != "holiday"]

    if not slots:
        warnings.append(
            "No schedule slots found. Ensure columns: day, start, end, subject "
            "(or a grid with day headers and time rows)."
        )

    return {
        "slots": slots,
        "n_lectures_per_week": n_lectures,
        "n_labs_per_week": n_labs,
        "holidays": holidays,
        "calendar_events": other_events,
        "parse_warnings": warnings,
    }


def project_calendar(
    slots: List[dict],
    holidays: List[dict],
    mode: str,
    weeks: int | None,
    start_date_str: str | None,
) -> dict:
    """Project weekly slots across an academic calendar, cancelling on holidays."""
    from datetime import date, timedelta

    today = date.today()
    if start_date_str:
        try:
            start = date.fromisoformat(start_date_str)
        except ValueError:
            start = today
    else:
        start = today

    week_start = start - timedelta(days=start.weekday())
    n_weeks = weeks if mode == "custom" and weeks else 52

    holiday_dates = {h.get("date", "") for h in holidays if h.get("date")}

    day_to_weekday = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6,
    }

    week_projections = []
    total_lectures = total_labs = total_cancelled = 0

    for week_num in range(1, n_weeks + 1):
        this_monday = week_start + timedelta(weeks=week_num - 1)
        this_sunday = this_monday + timedelta(days=6)
        occurrences = []
        week_holidays = []

        for slot in slots:
            day_name = slot.get("day", "")
            if day_name not in day_to_weekday:
                continue
            occ_date = (this_monday + timedelta(days=day_to_weekday[day_name])).isoformat()
            cancelled = occ_date in holiday_dates
            cancel_reason = None
            if cancelled:
                for h in holidays:
                    if h.get("date") == occ_date:
                        cancel_reason = h.get("title", "Holiday")
                        break
                total_cancelled += 1
            else:
                if slot.get("slot_type") == "lecture":
                    total_lectures += 1
                elif slot.get("slot_type") == "lab":
                    total_labs += 1

            occurrences.append({
                "date": occ_date,
                "slot": slot,
                "week_number": week_num,
                "cancelled": cancelled,
                "cancellation_reason": cancel_reason,
            })

        for h in holidays:
            d = h.get("date", "")
            if d:
                try:
                    from datetime import date as dt_date
                    hdate = dt_date.fromisoformat(d)
                    if this_monday <= hdate <= this_sunday:
                        week_holidays.append(h)
                except ValueError:
                    pass

        occurrences.sort(key=lambda o: (o["date"], o["slot"].get("start", "")))
        week_projections.append({
            "week_number": week_num,
            "week_start": this_monday.isoformat(),
            "week_end": this_sunday.isoformat(),
            "occurrences": occurrences,
            "holidays_this_week": week_holidays,
        })

    return {
        "total_weeks": n_weeks,
        "total_lectures": total_lectures,
        "total_labs": total_labs,
        "total_cancelled": total_cancelled,
        "week_projections": week_projections,
    }
