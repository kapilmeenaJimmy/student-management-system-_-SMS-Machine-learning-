/**
 * Dashboard.jsx — SMS-MH-AVIE FINAL-PROD
 * Mobile-First Liquid VisionOS Build
 *
 * Architecture:
 *  • Single useReducer SSOT — ALL score fields shared, zero desync
 *  • POST /api/v1/academic/evaluate fires on every SSOT change (debounced 280ms)
 *  • Hero Gauge shows academic formula score; UCI ML supplies only ±q̂ CI margin
 *  • Framer Motion AnimatePresence for Genie drawer
 *  • .glass-card CSS class with --mouse-x/--mouse-y specular cursor tracking
 *  • 12-month calendar Sep 2026 – Aug 2027, click-to-toggle attendance
 *  • 5-track prescription engine (arithmetic + SLSQP)
 *  • Camera capture (environment-facing) + file upload in OCR dropzone
 *  • Fully responsive: flex-col on mobile → xl:grid-cols-12 on desktop
 */

import React, {
  useCallback, useEffect, useMemo, useReducer, useRef, useState,
} from 'react'
import { AnimatePresence, motion } from 'framer-motion'

// ── Palette ──────────────────────────────────────────────────────────────────
const P = {
  cyan:   '#00F0FF',
  mint:   '#00F5A0',
  amber:  '#FFB020',
  purple: '#9D6EFA',
  pink:   '#FF6B9D',
  text:   '#DCE2EE',
  muted:  '#4A5568',
  bg:     '#07090E',
}

function scoreColor(s) {
  if (s >= 70) return P.mint
  if (s >= 50) return P.amber
  return P.amber   // never hard red per spec
}

// ── Cursor-tracking hook ─────────────────────────────────────────────────────
function useCursorTrack() {
  const ref = useRef(null)
  const handleMove = useCallback((e) => {
    const el = ref.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width)  * 100
    const y = ((e.clientY - rect.top)  / rect.height) * 100
    el.style.setProperty('--mouse-x', `${x}%`)
    el.style.setProperty('--mouse-y', `${y}%`)
  }, [])
  const handleLeave = useCallback(() => {
    const el = ref.current
    if (!el) return
    el.style.setProperty('--mouse-x', '50%')
    el.style.setProperty('--mouse-y', '50%')
  }, [])
  return { ref, onMouseMove: handleMove, onMouseLeave: handleLeave }
}

// ── API Layer ────────────────────────────────────────────────────────────────
const BASE = '/api/v1'
const j = (path, opts = {}) =>
  fetch(`${BASE}${path}`, { headers: { 'Content-Type': 'application/json' }, ...opts })
    .then(r => r.ok ? r.json() : r.text().then(t => { throw new Error(t) }))

const apiHealth       = async () => { const t = performance.now(); const d = await j('/health'); return { ...d, ping_ms: Math.round(performance.now() - t) } }
const apiStudents     = ()      => j('/students')
const apiAddStudent   = (b)     => j('/students',        { method: 'POST', body: JSON.stringify(b) })
const apiDelStudent   = (id)    => j(`/students/${id}`,  { method: 'DELETE' })
const apiEligibility  = (b)     => j('/eligibility',     { method: 'POST', body: JSON.stringify(b) })
const apiOcr          = (text, hint) => j('/ocr/parse',  { method: 'POST', body: JSON.stringify({ raw_text: text, source_hint: hint }) })
const apiTimetable    = (file)  => {
  const fd = new FormData(); fd.append('file', file)
  return fetch(`${BASE}/timetable/upload`, { method: 'POST', body: fd })
    .then(r => r.ok ? r.json() : r.text().then(t => { throw new Error(t) }))
}
const apiEvaluate     = (b)     => j('/academic/evaluate', { method: 'POST', body: JSON.stringify(b) })

// ── SSOT Reducer ─────────────────────────────────────────────────────────────
const INIT = {
  // SSOT score fields — all write here, Hero Gauge reads from evaluation
  weekly_study_hours: 10.0,
  lab_marks:          [],
  n_labs_total:       5,
  midterm_score:      0.0,
  endsem_score:       0.0,
  target_score:       null,

  // Evaluation result (from /academic/evaluate)
  evaluation:   null,
  evaluating:   false,
  evalError:    null,

  // Non-score UI
  students:         [],
  activeStudentId:  null,
  health:           null,
  connected:        false,
  activeTab:        'academic', // 'academic' | 'attendance' | 'calendar' | 'ocr'

  // Attendance (eligibility panel)
  attendance: {
    classes_attended: 30, classes_held: 40,
    labs_attended: 5,     labs_held: 6,
    result: null,
  },

  // Calendar
  calendarSlots: [],

  // Roster drawer
  rosterOpen: false,
}

function reducer(s, a) {
  switch (a.type) {
    case 'STUDY_HRS':   return { ...s, weekly_study_hours: a.v }
    case 'LAB_MARKS':   return { ...s, lab_marks: a.v }
    case 'N_LABS':      return { ...s, n_labs_total: a.v }
    case 'MIDTERM':     return { ...s, midterm_score: a.v }
    case 'ENDSEM':      return { ...s, endsem_score: a.v }
    case 'TARGET':      return { ...s, target_score: a.v }
    case 'EVAL_START':  return { ...s, evaluating: true, evalError: null }
    case 'EVAL_DONE':   return { ...s, evaluating: false, evaluation: a.r, evalError: null }
    case 'EVAL_ERR':    return { ...s, evaluating: false, evalError: a.e }
    case 'STUDENTS':    return { ...s, students: a.v }
    case 'ADD_STUDENT': return { ...s, students: [...s.students, a.v] }
    case 'DEL_STUDENT': {
      const students = s.students.filter(x => x.student_id !== a.id)
      return { ...s, students, activeStudentId: s.activeStudentId === a.id ? (students[0]?.student_id ?? null) : s.activeStudentId }
    }
    case 'ACTIVE_SID':  return { ...s, activeStudentId: a.id }
    case 'HEALTH':      return { ...s, health: a.v, connected: true }
    case 'DISCONNECTED':return { ...s, connected: false }
    case 'TAB':         return { ...s, activeTab: a.v }
    case 'ATTENDANCE':  return { ...s, attendance: { ...s.attendance, ...a.v } }
    case 'CAL_SLOTS':   return { ...s, calendarSlots: a.v }
    case 'ROSTER_OPEN': return { ...s, rosterOpen: a.v }
    default:            return s
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────
const AVATARS = ['#00F0FF','#00F5A0','#9D6EFA','#FFB020','#FF6B9D','#4ECDC4','#45B7D1','#FECA57','#FF9FF3','#54A0FF']
const avatarColor = (name = '') => AVATARS[[...name].reduce((a, c, i) => a + c.charCodeAt(0) * (i + 1), 0) % AVATARS.length]
const initials    = (n = '')   => n.split(' ').slice(0,2).map(w => w[0]?.toUpperCase()||'').join('')

// ── Ring ─────────────────────────────────────────────────────────────────────
function Ring({ value = 0, max = 100, color = P.cyan, size = 72, sw = 5, label = '', sublabel = '', children }) {
  const r    = (size - sw * 2) / 2
  const circ = 2 * Math.PI * r
  const pct  = Math.max(0, Math.min(1, value / max))
  const dash = circ * pct
  const cx   = size / 2
  return (
    <div className="flex flex-col items-center" style={{ gap: 4 }}>
      <div style={{ position: 'relative', width: size, height: size }}>
        <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
          <circle cx={cx} cy={cx} r={r} fill="none"
            stroke="rgba(255,255,255,0.05)" strokeWidth={sw} />
          <circle cx={cx} cy={cx} r={r} fill="none"
            stroke={color} strokeWidth={sw} strokeLinecap="round"
            strokeDasharray={`${dash} ${circ}`}
            style={{ transition: 'stroke-dasharray 0.55s cubic-bezier(0.4,0,0.2,1)',
              filter: `drop-shadow(0 0 5px ${color}55)` }} />
        </svg>
        <div style={{ position: 'absolute', inset: 0, display: 'flex',
          flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
          {children ?? (
            <span style={{ fontSize: size < 55 ? 10 : 13, fontWeight: 600, color,
              fontFamily: 'JetBrains Mono,monospace', lineHeight: 1 }}>
              {Math.round(value)}
            </span>
          )}
        </div>
      </div>
      {label    && <div style={{ fontSize: 9, color: P.muted, textAlign: 'center', textTransform: 'uppercase', letterSpacing: 1.2 }}>{label}</div>}
      {sublabel && <div style={{ fontSize: 8, color: P.muted + '80', textAlign: 'center' }}>{sublabel}</div>}
    </div>
  )
}

// ── Avatar ────────────────────────────────────────────────────────────────────
function Avatar({ name = '', color = P.cyan, size = 36, active = false, onClick }) {
  return (
    <div onClick={onClick} style={{
      width: size, height: size, borderRadius: '50%',
      background: `${color}18`,
      border: `2px solid ${active ? color : 'rgba(255,255,255,0.08)'}`,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontSize: size * 0.33, fontWeight: 700, color, flexShrink: 0,
      boxShadow: active ? `0 0 16px ${color}45` : 'none',
      transition: 'all 0.22s', cursor: onClick ? 'pointer' : 'default',
      fontFamily: 'Inter,sans-serif', userSelect: 'none',
    }}>{initials(name)}</div>
  )
}

// ── Connection Badge ──────────────────────────────────────────────────────────
function ConnectionBadge({ connected, ping }) {
  const c = connected ? P.mint : P.amber
  return (
    <div className="flex justify-center mt-2">
      <div style={{
        display: 'inline-flex', alignItems: 'center', gap: 6,
        padding: '4px 14px', borderRadius: 999,
        background: `${c}08`, border: `1px solid ${c}28`,
      }}>
        <span className={connected ? 'status-dot-connected' : ''} style={{
          width: 6, height: 6, borderRadius: '50%', background: c, display: 'inline-block',
        }} />
        <span style={{ fontSize: 11, color: c, fontFamily: 'Inter,sans-serif', fontWeight: 500 }}>
          {connected ? 'Connected' : 'Disconnected'}
        </span>
        {connected && ping != null && (
          <span style={{ fontSize: 10, color: P.muted, fontFamily: 'JetBrains Mono,monospace' }}>
            {ping}ms
          </span>
        )}
      </div>
    </div>
  )
}

// ── Hero Gauge (V8: gradient arc, correct SVG geometry, zero collision) ────────
function HeroGauge({ score, letter, ci, loading }) {
  const cursor = useCursorTrack()

  // SVG geometry per V8 spec: viewBox 260×260, r=88, sw=10, 270° arc
  const CX = 130, CY = 130, R = 88, SW = 10
  // Circumference of full circle; 270° arc ≈ 415 units of it
  const FULL_CIRC = 2 * Math.PI * R          // ≈ 552.9
  const ARC_LEN   = FULL_CIRC * (270 / 360)  // ≈ 415
  const pct    = Math.max(0, Math.min(1, (score ?? 0) / 100))
  const filled = ARC_LEN * pct
  const offset = ARC_LEN - filled           // how much to leave transparent

  // CI band
  const ciLo = ci ? Math.max(0, Math.min(1, ci.lower / 100)) : pct
  const ciHi = ci ? Math.max(0, Math.min(1, ci.upper / 100)) : pct
  const ciBandLen = ARC_LEN * (ciHi - ciLo)
  const ciBandOff = ARC_LEN - ARC_LEN * ciHi

  const sc = scoreColor(score ?? 0)

  return (
    <div className="glass-card p-5 flex flex-col items-center" {...cursor}>
      <div className="section-label">Academic Score</div>

      {/* SVG container: w-[260px] h-[260px] per spec */}
      <div style={{ position: 'relative', width: 260, height: 260, margin: '0 auto' }}>
        <svg width={260} height={260} viewBox="0 0 260 260" style={{ display: 'block' }}>
          <defs>
            {/* V8 gradient stroke: cyan → mint */}
            <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%"   stopColor={P.cyan} />
              <stop offset="100%" stopColor={P.mint} />
            </linearGradient>
            {/* Loading shimmer gradient */}
            <linearGradient id="shimmerGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%"   stopColor="rgba(255,255,255,0.03)" />
              <stop offset="50%"  stopColor="rgba(255,255,255,0.07)" />
              <stop offset="100%" stopColor="rgba(255,255,255,0.03)" />
            </linearGradient>
          </defs>

          {/* ── Track (270° open arc, gap at bottom) ── */}
          <circle cx={CX} cy={CY} r={R} fill="none"
            stroke="rgba(255,255,255,0.04)" strokeWidth={SW}
            strokeLinecap="round"
            strokeDasharray={`${ARC_LEN} ${FULL_CIRC}`}
            strokeDashoffset={0}
            transform={`rotate(135 ${CX} ${CY})`} />

          {/* ── CI band (ghost behind score arc) ── */}
          {ci && !loading && (
            <circle cx={CX} cy={CY} r={R} fill="none"
              stroke={P.cyan} strokeWidth={SW} opacity={0.12}
              strokeLinecap="round"
              strokeDasharray={`${ciBandLen} ${FULL_CIRC}`}
              strokeDashoffset={ciBandOff}
              transform={`rotate(135 ${CX} ${CY})`}
              style={{ transition: 'all 0.5s cubic-bezier(0.4,0,0.2,1)' }} />
          )}

          {/* ── Score arc ── */}
          <circle cx={CX} cy={CY} r={R} fill="none"
            stroke={loading ? 'url(#shimmerGrad)' : 'url(#gaugeGrad)'} strokeWidth={SW}
            strokeLinecap="round"
            strokeDasharray={`${filled} ${FULL_CIRC}`}
            strokeDashoffset={0}
            transform={`rotate(135 ${CX} ${CY})`}
            className={loading ? '' : 'gauge-glow'}
            style={{ transition: 'stroke-dasharray 0.6s cubic-bezier(0.4,0,0.2,1)' }} />
        </svg>

        {/* ── Inner text stack (absolute, pointer-events-none, zero SVG collision) ── */}
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
          pointerEvents: 'none',
          paddingTop: 10,
        }}>
          {/* Score value */}
          <span style={{
            fontSize: 52, fontWeight: 300, lineHeight: 1,
            fontFamily: 'JetBrains Mono,monospace',
            color: loading ? 'rgba(255,255,255,0.08)' : P.text,
            letterSpacing: -2,
            transition: 'color 0.3s',
          }}>
            {loading ? '–' : (score ?? 0).toFixed(1)}
          </span>
          {/* Sub-label */}
          <span style={{
            fontSize: 9, letterSpacing: 3, color: P.muted,
            textTransform: 'uppercase', marginTop: 4, fontWeight: 600,
          }}>SCORE / 100</span>
          {/* Letter grade badge */}
          {!loading && letter && (
            <div style={{
              marginTop: 10, padding: '2px 12px', borderRadius: 999,
              background: 'rgba(0,240,255,0.10)',
              border: '1px solid rgba(0,240,255,0.28)',
              fontSize: 11.5, fontWeight: 700, color: P.cyan,
              fontFamily: 'JetBrains Mono,monospace',
              boxShadow: '0 0 12px rgba(0,240,255,0.18)',
            }}>
              Grade {letter}
            </div>
          )}
        </div>
      </div>

      {/* ── Conformal Interval Pill (BELOW arc, separate compartment, zero collision) ── */}
      {ci && !loading && (
        <div style={{
          marginTop: 8, padding: '6px 14px', borderRadius: 12,
          background: 'rgba(255,255,255,0.025)',
          border: '1px solid rgba(255,255,255,0.06)',
          fontSize: 11, fontFamily: 'JetBrains Mono,monospace',
          color: P.muted, display: 'flex', alignItems: 'center', gap: 6,
        }}>
          <span style={{ color: P.cyan + 'AA' }}>[{ci.lower.toFixed(1)}</span>
          <span>—</span>
          <span style={{ color: P.cyan + 'AA' }}>{ci.upper.toFixed(1)}]</span>
          <span style={{ fontSize: 9, letterSpacing: 1.5, textTransform: 'uppercase' }}>95% CI</span>
        </div>
      )}
    </div>
  )
}

// ── Score Component Rings ──────────────────────────────────────────────────────
function ScoreRings({ ev }) {
  if (!ev) return null
  const cursor = useCursorTrack()
  return (
    <div className="glass-card p-4" {...cursor}>
      <div className="section-label">Score Components</div>
      <div className="flex justify-around">
        {[
          { l: 'Lab', v: ev.lab_component,      mx: 20, c: P.purple  },
          { l: 'Mid', v: ev.midterm_component,   mx: 20, c: P.cyan    },
          { l: 'End', v: ev.endsem_component,    mx: 60, c: P.amber   },
          { l: 'Study', v: ev.study_factor,      mx: 5,  c: P.mint    },
        ].map(({ l, v, mx, c }) => (
          <Ring key={l} value={v} max={mx} color={c} size={58} sw={4}
            label={l} sublabel={`${v.toFixed(1)}/${mx}`} />
        ))}
      </div>
    </div>
  )
}

// ── Bento Class Cards ─────────────────────────────────────────────────────────
function BentoCards({ students, evaluations, onOpenRoster }) {
  if (!students.length) return null

  const scored = students
    .map(s => ({ ...s, score: evaluations[s.student_id]?.composite_score ?? 0,
                       letter: evaluations[s.student_id]?.letter_grade ?? '—' }))
    .sort((a, b) => b.score - a.score)

  const top     = scored[0]
  const atRisk  = scored[scored.length - 1]
  const avg     = scored.reduce((a, s) => a + s.score, 0) / scored.length
  const cursor  = useCursorTrack()

  const MiniStat = ({ title, student, sc, color }) => (
    <div className="glass-card p-3.5 flex items-center gap-3">
      <Ring value={sc} max={100} color={color} size={58} sw={4}>
        <Avatar name={student?.student_name} color={student?.avatar_color || color} size={38} />
      </Ring>
      <div className="flex-1 min-w-0">
        <div className="section-label mb-1">{title}</div>
        <div style={{ fontSize: 12.5, fontWeight: 600, color: P.text,
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {student?.student_name}
        </div>
        <div style={{ fontSize: 20, fontWeight: 300, color, fontFamily: 'JetBrains Mono,monospace', lineHeight: 1.2 }}>
          {Math.round(sc)}
        </div>
      </div>
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        <MiniStat title="Top Performer" student={top}    sc={top.score}    color={P.mint}  />
        <MiniStat title="Most At Risk"  student={atRisk} sc={atRisk.score} color={P.amber} />
      </div>

      {/* Class Aggregate — genie anchor */}
      <div className="glass-card p-4 cursor-pointer" {...cursor}
        onClick={onOpenRoster}
        style={{ transition: 'border-color 0.2s' }}
        onMouseEnter={e => e.currentTarget.style.borderColor = 'rgba(0,240,255,0.30)'}
        onMouseLeave={e => e.currentTarget.style.borderColor = ''}>
        <div className="flex justify-between items-center mb-3">
          <div className="section-label mb-0">Class Aggregate</div>
          <span style={{ fontSize: 10, color: P.cyan + 'AA' }}>Manage roster ›</span>
        </div>
        <div className="flex items-center gap-4">
          <Ring value={avg} max={100} color={P.cyan} size={66} sw={5} label="Avg" />
          <div>
            <div style={{ fontSize: 11, color: P.muted, marginBottom: 6 }}>
              {students.length} student{students.length !== 1 ? 's' : ''}
            </div>
            <div className="flex flex-wrap gap-2">
              {scored.slice(0, 5).map(s => (
                <div key={s.student_id} className="flex items-center gap-1.5">
                  <Avatar name={s.student_name} color={s.avatar_color} size={20} />
                  <span style={{ fontSize: 9.5, color: P.muted,
                    fontFamily: 'JetBrains Mono,monospace' }}>
                    {Math.round(s.score)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Genie Roster Drawer ───────────────────────────────────────────────────────
function RosterDrawer({ open, students, activeId, evaluations, onClose, onSelect, onAdd, onDelete }) {
  const [form, setForm] = useState({ student_name: '', roll_no: '', enrollment_year: 2025 })
  const [saving, setSaving]     = useState(false)
  const [showForm, setShowForm] = useState(false)

  const submit = async () => {
    if (!form.student_name.trim()) return
    setSaving(true)
    try { await onAdd(form) }
    finally { setSaving(false); setShowForm(false); setForm({ student_name: '', roll_no: '', enrollment_year: 2025 }) }
  }

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            key="backdrop"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.35 }}
            style={{ position: 'fixed', inset: 0, zIndex: 1900,
              background: 'rgba(0,0,0,0.70)', backdropFilter: 'blur(18px)',
              WebkitBackdropFilter: 'blur(18px)' }}
            onClick={onClose}
          />

          {/* Drawer — Genie effect */}
          <motion.div
            key="drawer"
            initial={{ opacity: 0, y: '100%', scaleY: 0.82, originY: 1 }}
            animate={{ opacity: 1, y: 0,      scaleY: 1,    originY: 1 }}
            exit={{   opacity: 0, y: '100%',  scaleY: 0.82, originY: 1 }}
            transition={{
              type: 'spring', damping: 32, stiffness: 300, mass: 0.95,
              opacity: { duration: 0.28 },
            }}
            style={{
              position: 'fixed', bottom: 0, left: 0, right: 0,
              zIndex: 2000, maxWidth: 660, margin: '0 auto',
              borderRadius: '28px 28px 0 0',
              maxHeight: '80vh', overflowY: 'auto',
            }}
            className="glass-hi"
          >
            <div style={{ padding: '28px 28px 48px' }}>
              {/* Handle */}
              <div style={{ width: 40, height: 4, borderRadius: 2, margin: '0 auto 22px',
                background: 'rgba(255,255,255,0.15)' }} />

              {/* Header */}
              <div className="flex justify-between items-start mb-5">
                <div>
                  <div style={{ fontSize: 18, fontWeight: 600, color: P.text }}>Student Roster</div>
                  <div style={{ fontSize: 11.5, color: P.muted, marginTop: 3 }}>
                    {students.length} enrolled student{students.length !== 1 ? 's' : ''}
                  </div>
                </div>
                <button onClick={() => setShowForm(v => !v)} style={{
                  padding: '7px 16px', borderRadius: 10, fontSize: 12, fontWeight: 600,
                  background: 'rgba(0,240,255,0.08)', border: '1px solid rgba(0,240,255,0.28)',
                  color: P.cyan, cursor: 'pointer',
                }}>+ Add</button>
              </div>

              {/* Add form */}
              <AnimatePresence>
                {showForm && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }} transition={{ duration: 0.25 }}
                    style={{ overflow: 'hidden', marginBottom: 16 }}>
                    <div style={{ background: 'rgba(0,0,0,0.22)', borderRadius: 14,
                      padding: '14px 16px', border: '1px solid rgba(255,255,255,0.06)',
                      display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'flex-end' }}>
                      {[
                        { k: 'student_name', lbl: 'Name *', type: 'text',   ph: 'Demo Student' },
                        { k: 'roll_no',      lbl: 'Roll No', type: 'text',   ph: 'DEMO-001' },
                        { k: 'enrollment_year', lbl: 'Year', type: 'number', ph: '2025' },
                      ].map(({ k, lbl, type, ph }) => (
                        <div key={k} style={{ flex: '1 1 100px', minWidth: 90 }}>
                          <label style={{ fontSize: 9.5, color: P.muted, display: 'block', marginBottom: 3 }}>{lbl}</label>
                          <input className="glass-input" type={type} placeholder={ph} value={form[k]}
                            onChange={e => setForm(p => ({ ...p, [k]: type === 'number' ? (parseInt(e.target.value) || 2025) : e.target.value }))} />
                        </div>
                      ))}
                      <button onClick={submit} disabled={saving} style={{
                        padding: '7px 16px', borderRadius: 8, fontSize: 12, fontWeight: 600,
                        background: 'rgba(0,240,255,0.10)', border: '1px solid rgba(0,240,255,0.30)',
                        color: P.cyan, cursor: saving ? 'wait' : 'pointer',
                      }}>{saving ? '…' : 'Save'}</button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Student list */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {students.map(s => {
                  const ev     = evaluations[s.student_id]
                  const score  = ev?.composite_score ?? 0
                  const isAct  = s.student_id === activeId
                  const color  = s.avatar_color || P.cyan
                  return (
                    <div key={s.student_id} onClick={() => { onSelect(s.student_id); onClose() }}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 14,
                        padding: '12px 16px', borderRadius: 16,
                        background: isAct ? `${color}08` : 'rgba(255,255,255,0.02)',
                        border: `1px solid ${isAct ? color + '30' : 'rgba(255,255,255,0.06)'}`,
                        cursor: 'pointer', transition: 'all 0.18s',
                      }}>
                      <Avatar name={s.student_name} color={color} size={46} active={isAct} />
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: 13.5, fontWeight: 600, color: P.text }}>
                          {s.student_name}
                          {isAct && <span style={{ marginLeft: 7, fontSize: 9.5, color: P.cyan,
                            background: 'rgba(0,240,255,0.10)', border: '1px solid rgba(0,240,255,0.25)',
                            borderRadius: 999, padding: '1px 7px' }}>Active</span>}
                        </div>
                        {s.roll_no && <div style={{ fontSize: 10.5, color: P.muted, marginTop: 1 }}>{s.roll_no}</div>}
                        <div style={{ fontSize: 10, color: P.muted, marginTop: 1 }}>Enrolled {s.enrollment_year}</div>
                      </div>
                      <div style={{ textAlign: 'right', flexShrink: 0 }}>
                        <div style={{ fontSize: 22, fontWeight: 300, color: ev ? scoreColor(score) : P.muted,
                          fontFamily: 'JetBrains Mono,monospace' }}>
                          {ev ? score.toFixed(1) : '—'}
                        </div>
                        {ev && <div style={{ fontSize: 10, color: scoreColor(score), marginTop: 1 }}>
                          {ev.letter_grade}
                        </div>}
                      </div>
                      {students.length > 1 && (
                        <button onClick={e => { e.stopPropagation(); onDelete(s.student_id) }}
                          style={{ fontSize: 10, color: P.muted + '70', background: 'none',
                            border: 'none', cursor: 'pointer', padding: '4px 6px', borderRadius: 6 }}>✕</button>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}

// ── Academic Controls (SSOT) ──────────────────────────────────────────────────
function AcademicControls({ state, dispatch }) {
  const { weekly_study_hours, lab_marks, n_labs_total, midterm_score, endsem_score } = state
  const cursor = useCursorTrack()

  const addLab    = () => { if (lab_marks.length < n_labs_total) dispatch({ type: 'LAB_MARKS', v: [...lab_marks, 0] }) }
  const updateLab = (i, v) => { const m = [...lab_marks]; m[i] = Math.max(0, Math.min(5, parseFloat(v)||0)); dispatch({ type: 'LAB_MARKS', v: m }) }
  const removeLab = (i)    => dispatch({ type: 'LAB_MARKS', v: lab_marks.filter((_,idx) => idx !== i) })

  const Knob = ({ label, value, min, max, step, onChange, unit = '', color = 'cyan' }) => (
    <div>
      <div className="flex justify-between mb-1.5">
        <label style={{ fontSize: 10.5, color: P.muted }}>{label}</label>
        <span style={{ fontSize: 10.5, color: color === 'mint' ? P.mint : color === 'amber' ? P.amber : P.cyan,
          fontFamily: 'JetBrains Mono,monospace' }}>
          {typeof value === 'number' ? (Number.isInteger(step) ? value : value.toFixed(1)) : value}{unit}
        </span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        className={color === 'mint' ? 'slider-mint' : color === 'amber' ? 'slider-amber' : ''}
        onChange={e => onChange(parseFloat(e.target.value))} />
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {/* Mutable Parameters */}
      <div className="glass-card p-4" {...cursor}>
        <div className="section-label">Mutable Parameters</div>
        <Knob label="Study Hours / Week" value={weekly_study_hours} min={1} max={45} step={0.5}
          onChange={v => dispatch({ type: 'STUDY_HRS', v })} />
        <div className="mt-4">
          <Knob label={`End-Semester Score (${endsem_score}/80)`} value={endsem_score}
            min={0} max={80} step={1} color="amber"
            onChange={v => dispatch({ type: 'ENDSEM', v })} />
        </div>
      </div>

      {/* Academic Assessment */}
      <div className="glass-card p-4" {...cursor}>
        <div className="section-label">Academic Assessment</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 9, marginBottom: 14 }}>
          {[
            { lbl: 'Total Labs', key: 'N_LABS',  val: n_labs_total,   min: 1, max: 60, step: 1,   toDispatch: v => ({ type: 'N_LABS', v: Math.round(v) }) },
            { lbl: `Midterm (${midterm_score}/20)`, key: 'MIDTERM', val: midterm_score, min: 0, max: 20, step: 0.5, toDispatch: v => ({ type: 'MIDTERM', v }) },
          ].map(({ lbl, val, min, max, step, toDispatch }) => (
            <div key={lbl}>
              <label style={{ fontSize: 9.5, color: P.muted, display: 'block', marginBottom: 4 }}>{lbl}</label>
              <input type="number" className="glass-input" value={val} min={min} max={max} step={step}
                onChange={e => dispatch(toDispatch(parseFloat(e.target.value)||0))} />
            </div>
          ))}
        </div>

        {/* Lab marks */}
        <div>
          <div className="flex justify-between items-center mb-2">
            <span className="section-label mb-0">
              Labs ({lab_marks.length}/{n_labs_total})
            </span>
            {lab_marks.length < n_labs_total && (
              <button onClick={addLab} style={{
                fontSize: 10, color: P.cyan, background: 'rgba(0,240,255,0.08)',
                border: '1px solid rgba(0,240,255,0.22)', borderRadius: 6,
                padding: '2px 8px', cursor: 'pointer',
              }}>+ Lab</button>
            )}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {lab_marks.map((m, i) => (
              <div key={i} className="flex items-center gap-1">
                <input type="number" value={m} min={0} max={5} step={0.5}
                  onChange={e => updateLab(i, e.target.value)}
                  style={{
                    width: 44, background: 'rgba(255,255,255,0.04)',
                    border: `1px solid ${scoreColor((m/5)*100)}28`,
                    borderRadius: 7, padding: '4px 4px', color: P.text,
                    fontSize: 11, outline: 'none', textAlign: 'center',
                    fontFamily: 'JetBrains Mono,monospace',
                  }} />
                <button onClick={() => removeLab(i)}
                  style={{ fontSize: 9, color: P.muted, background: 'none',
                    border: 'none', cursor: 'pointer', padding: 1 }}>✕</button>
              </div>
            ))}
            {lab_marks.length === 0 && (
              <span style={{ fontSize: 10.5, color: P.muted, fontStyle: 'italic' }}>No labs entered yet</span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── 5-Track Prescription Engine ───────────────────────────────────────────────
function FiveTrackPrescription({ state, dispatch }) {
  const { evaluation, endsem_score, weekly_study_hours } = state
  const curScore = evaluation?.composite_score ?? 0
  const ceiling  = evaluation?.ceiling ?? 100
  const defTarget = Math.min(ceiling, curScore + 10)

  const [localTarget, setLocalTarget] = useState(defTarget)
  const [running, setRunning]   = useState(false)
  const [result,  setResult]    = useState(null)
  const cursor = useCursorTrack()

  useEffect(() => {
    if (state.target_score === null) setLocalTarget(Math.min(ceiling, curScore + 10))
  }, [curScore, ceiling, state.target_score])

  const run = async () => {
    setRunning(true); setResult(null)
    try {
      const res = await apiEvaluate({
        lab_marks: state.lab_marks, n_labs_total: state.n_labs_total,
        midterm_score: state.midterm_score, endsem_score: state.endsem_score,
        weekly_study_hours: state.weekly_study_hours,
        target_score: localTarget,
      })
      setResult(res.prescription)
      dispatch({ type: 'TARGET', v: localTarget })
    } catch (e) { console.warn('Prescription error:', e.message) }
    finally { setRunning(false) }
  }

  const TRACK_COLORS = { 1: P.amber, 2: P.mint, 3: P.cyan, 4: P.purple, 5: P.amber }
  const tc  = result ? (TRACK_COLORS[result.track] || P.cyan) : P.cyan
  const gap = result?.gap ?? (localTarget - curScore)

  return (
    <div className="glass-card p-5" {...cursor}>
      <div className="section-label">Prescription Engine</div>

      {/* Target slider */}
      <div className="mb-4">
        <div className="flex justify-between mb-1.5">
          <label style={{ fontSize: 10.5, color: P.muted }}>
            Target Score: <span style={{ color: P.cyan, fontFamily: 'JetBrains Mono,monospace' }}>{localTarget.toFixed(0)}</span>
          </label>
          <span style={{ fontSize: 10.5, fontFamily: 'JetBrains Mono,monospace',
            color: gap > 0 ? P.amber : P.mint }}>
            {gap > 0 ? '+' : ''}{gap.toFixed(1)} pts
          </span>
        </div>
        <input type="range" min={0} max={100} step={0.5} value={localTarget}
          onChange={e => setLocalTarget(parseFloat(e.target.value))} />
        <div className="flex justify-between mt-1" style={{ fontSize: 9, color: P.muted + '60' }}>
          <span>0</span>
          <span style={{ color: P.mint + '80' }}>Now: {curScore.toFixed(1)}</span>
          <span style={{ color: P.amber + '80' }}>Ceiling: {ceiling.toFixed(1)}</span>
          <span>100</span>
        </div>
      </div>

      <button onClick={run} disabled={running} style={{
        width: '100%', padding: '10px 16px', borderRadius: 12, fontSize: 13, fontWeight: 600,
        background: running ? 'rgba(255,255,255,0.02)' : 'rgba(0,240,255,0.08)',
        border: `1px solid ${running ? 'rgba(255,255,255,0.06)' : 'rgba(0,240,255,0.30)'}`,
        color: running ? P.muted : P.cyan, cursor: running ? 'wait' : 'pointer',
        marginBottom: 16, fontFamily: 'Inter,sans-serif', transition: 'all 0.2s',
      }}>⚡ Analyse &amp; Prescribe</button>

      {/* Track result */}
      <AnimatePresence mode="wait">
        {result && (
          <motion.div key={result.track}
            initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.25 }}>

            {/* Track badge */}
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6,
              padding: '4px 12px', borderRadius: 999, marginBottom: 12,
              background: `${tc}0E`, border: `1px solid ${tc}28` }}>
              <span style={{ fontSize: 10, fontWeight: 700, color: tc,
                fontFamily: 'JetBrains Mono,monospace' }}>Track {result.track}</span>
              <span style={{ fontSize: 11, color: P.text }}>{result.track_label}</span>
            </div>

            {/* ── Track 1: Under-aiming ── */}
            {result.track === 1 && (
              <div style={{ background: `${P.amber}08`, border: `1px solid ${P.amber}20`,
                borderRadius: 12, padding: '14px 16px' }}>
                <div style={{ fontSize: 12.5, color: P.amber, fontWeight: 600, marginBottom: 6 }}>
                  ⚠ Raising the bar is safer than lowering it
                </div>
                <div style={{ fontSize: 11.5, color: '#E2C585', lineHeight: 1.65, marginBottom: 10 }}>
                  {result.under_aiming_message}
                </div>
                {result.recommended_target != null && (
                  <button onClick={() => setLocalTarget(result.recommended_target)}
                    style={{ fontSize: 11, color: P.cyan, background: 'rgba(0,240,255,0.08)',
                      border: '1px solid rgba(0,240,255,0.25)', borderRadius: 8,
                      padding: '5px 12px', cursor: 'pointer' }}>
                    Use recommended: {result.recommended_target.toFixed(0)}
                  </button>
                )}
              </div>
            )}

            {/* ── Track 2: Safety buffer ── */}
            {result.track === 2 && (
              <div style={{ background: `${P.mint}07`, border: `1px solid ${P.mint}20`,
                borderRadius: 12, padding: '14px 16px' }}>
                <div className="flex items-center gap-3 mb-4">
                  <Ring value={100} max={100} color={P.mint} size={54} sw={4}>
                    <span style={{ fontSize: 9.5, fontWeight: 700, color: P.mint,
                      fontFamily: 'JetBrains Mono,monospace' }}>
                      +{result.safety_cushion_pts?.toFixed(1)}
                    </span>
                  </Ring>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: P.mint }}>Target Secured</div>
                    <div style={{ fontSize: 10.5, color: P.muted }}>
                      +{result.safety_cushion_pts?.toFixed(1)} pts above target
                    </div>
                  </div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                  {(result.maintenance_protocol || []).map((note, i) => (
                    <div key={i} style={{ fontSize: 10.5, color: '#9DB8C5', lineHeight: 1.55,
                      padding: '6px 10px', background: 'rgba(0,0,0,0.15)', borderRadius: 8,
                      borderLeft: `2px solid ${P.mint}35` }}>{note}</div>
                  ))}
                </div>
              </div>
            )}

            {/* ── Track 3: Tactical ── */}
            {result.track === 3 && (
              <div style={{ background: `${P.cyan}07`, border: `1px solid ${P.cyan}18`,
                borderRadius: 12, padding: '14px 16px' }}>
                <div style={{ fontSize: 12.5, fontWeight: 600, color: P.cyan, marginBottom: 10 }}>
                  Minor adjustments — gap is closable
                </div>
                <div className="flex gap-2.5 mb-3 flex-wrap">
                  {result.study_hours_delta > 0 && (
                    <div style={{ flex: 1, padding: '8px 12px', borderRadius: 10,
                      background: 'rgba(0,0,0,0.18)', border: `1px solid ${P.cyan}18` }}>
                      <div style={{ fontSize: 9.5, color: P.muted, marginBottom: 2 }}>Study Hours</div>
                      <div style={{ fontSize: 16, fontWeight: 300, color: P.cyan,
                        fontFamily: 'JetBrains Mono,monospace' }}>
                        +{result.study_hours_delta?.toFixed(0)} hrs/wk
                      </div>
                    </div>
                  )}
                  {result.required_lab_avg != null && (
                    <div style={{ flex: 1, padding: '8px 12px', borderRadius: 10,
                      background: 'rgba(0,0,0,0.18)', border: `1px solid ${P.purple}18` }}>
                      <div style={{ fontSize: 9.5, color: P.muted, marginBottom: 2 }}>Lab Avg Target</div>
                      <div style={{ fontSize: 16, fontWeight: 300, color: P.purple,
                        fontFamily: 'JetBrains Mono,monospace' }}>
                        {result.required_lab_avg?.toFixed(1)}/5
                      </div>
                    </div>
                  )}
                </div>
                {(result.tactical_notes || []).map((n, i) => (
                  <div key={i} style={{ fontSize: 10.5, color: '#9DB8C5', lineHeight: 1.55,
                    padding: '5px 10px', marginBottom: 4, background: 'rgba(0,0,0,0.12)',
                    borderRadius: 7, borderLeft: `2px solid ${P.cyan}35` }}>{n}</div>
                ))}
              </div>
            )}

            {/* ── Track 4: Structural (SLSQP) ── */}
            {result.track === 4 && (
              <div>
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: 5,
                  background: `${P.purple}09`, border: `1px solid ${P.purple}22`,
                  borderRadius: 20, padding: '3px 12px', marginBottom: 12 }}>
                  <span style={{ fontSize: 10.5, color: P.purple }}>
                    SLSQP achieves: {result.achieved_score?.toFixed(1)}
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
                  {(result.action_items || []).map((item, i) => (
                    <div key={i} style={{ background: `${P.purple}06`,
                      border: `1px solid ${P.purple}16`, borderRadius: 11,
                      padding: '10px 14px', display: 'flex',
                      justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <div style={{ fontSize: 12, color: P.text, fontWeight: 500 }}>{item.label}</div>
                        <div style={{ fontSize: 10, color: P.muted, marginTop: 2 }}>
                          {item.current} → <span style={{ color: P.purple }}>{item.required}</span>
                          {' '}{item.unit}
                        </div>
                      </div>
                      <span style={{ fontSize: 15, fontWeight: 700, color: P.purple,
                        fontFamily: 'JetBrains Mono,monospace' }}>
                        +{(item.required - item.current).toFixed(1)}
                      </span>
                    </div>
                  ))}
                  {result.required_lab_avg_remaining != null && (
                    <div style={{ background: `${P.mint}06`, border: `1px solid ${P.mint}16`,
                      borderRadius: 11, padding: '10px 14px',
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <div style={{ fontSize: 12, color: P.text, fontWeight: 500 }}>Remaining Labs</div>
                        <div style={{ fontSize: 10, color: P.muted, marginTop: 2 }}>Target avg per lab</div>
                      </div>
                      <span style={{ fontSize: 15, fontWeight: 700, color: P.mint,
                        fontFamily: 'JetBrains Mono,monospace' }}>
                        {result.required_lab_avg_remaining?.toFixed(1)}/5
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* ── Track 5: Infeasible ── */}
            {result.track === 5 && (
              <div style={{ background: `${P.amber}08`, border: `1px solid ${P.amber}22`,
                borderRadius: 12, padding: '14px 16px' }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: P.amber, marginBottom: 6 }}>
                  ✗ Mathematically Unachievable
                </div>
                <div style={{ fontSize: 11.5, color: '#E2C585', lineHeight: 1.65, marginBottom: 12 }}>
                  {result.infeasible_reason}
                </div>
                <div className="flex items-center gap-3" style={{
                  background: 'rgba(0,0,0,0.18)', borderRadius: 10, padding: '8px 14px', marginBottom: 8 }}>
                  <span style={{ fontSize: 10.5, color: P.muted }}>Maximum attainable:</span>
                  <span style={{ fontSize: 22, fontWeight: 300, color: P.amber,
                    fontFamily: 'JetBrains Mono,monospace' }}>
                    {result.ceiling?.toFixed(1)}
                  </span>
                </div>
                {result.bottleneck_details && (
                  <div style={{ fontSize: 10.5, color: P.muted, lineHeight: 1.55,
                    borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: 10 }}>
                    <span style={{ color: P.amber + 'AA' }}>Bottleneck: </span>
                    {result.bottleneck_details}
                  </div>
                )}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ── Exam Eligibility Panel ────────────────────────────────────────────────────
function ExamEligibilityPanel({ attendance, dispatch }) {
  const { classes_attended, classes_held, labs_attended, labs_held, result } = attendance
  const [busy, setBusy] = useState(false)
  const debRef = useRef(null)
  const cursor = useCursorTrack()

  const check = useCallback(async (v) => {
    if (v.classes_attended > v.classes_held || v.labs_attended > v.labs_held) return
    if (v.classes_held + 2 * v.labs_held === 0) return
    setBusy(true)
    try { const r = await apiEligibility(v); dispatch({ type: 'ATTENDANCE', v: { result: r } }) }
    catch {}
    finally { setBusy(false) }
  }, [dispatch])

  const upd = (key, raw) => {
    const next = { classes_attended, classes_held, labs_attended, labs_held, [key]: Math.max(0, Math.round(parseFloat(raw)||0)) }
    dispatch({ type: 'ATTENDANCE', v: next })
    clearTimeout(debRef.current)
    debRef.current = setTimeout(() => check(next), 400)
  }

  const tEff    = classes_held + 2 * labs_held
  const aEff    = classes_attended + 2 * labs_attended
  const ratio   = tEff > 0 ? aEff / tEff : 0
  const eligible = result?.eligible ?? ratio >= 0.75
  const sc       = eligible ? P.mint : P.amber

  return (
    <div className="glass-card p-5" {...cursor}>
      <div className="section-label">Exam Eligibility — 75% Rule</div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 9, marginBottom: 14 }}>
        {[
          { key: 'classes_attended', lbl: 'Classes Attended', v: classes_attended },
          { key: 'classes_held',     lbl: 'Classes Held',     v: classes_held },
          { key: 'labs_attended',    lbl: 'Labs Attended',    v: labs_attended },
          { key: 'labs_held',        lbl: 'Labs Held',        v: labs_held },
        ].map(({ key, lbl, v }) => (
          <div key={key}>
            <label style={{ fontSize: 9.5, color: P.muted, display: 'block', marginBottom: 3 }}>{lbl}</label>
            <input type="number" className="glass-input" value={v} min={0}
              onChange={e => upd(key, e.target.value)} />
          </div>
        ))}
      </div>

      {/* Formula display */}
      <div style={{ background: 'rgba(0,0,0,0.20)', borderRadius: 10, padding: '9px 13px', marginBottom: 12,
        fontSize: 10, color: P.muted, fontFamily: 'JetBrains Mono,monospace',
        display: 'grid', gridTemplateColumns: '1fr auto', gap: '3px 14px' }}>
        <span>T_eff = {classes_held} + 2×{labs_held}</span>
        <span style={{ color: P.text }}>= {tEff}</span>
        <span>A_eff = {classes_attended} + 2×{labs_attended}</span>
        <span style={{ color: P.text }}>= {aEff}</span>
        <span>Ratio R</span>
        <span style={{ color: sc }}>{tEff > 0 ? (ratio * 100).toFixed(1) : '—'}%</span>
      </div>

      {/* Status */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 14,
        background: `${sc}06`, border: `1px solid ${sc}20`,
        borderRadius: 14, padding: '12px 16px' }}>
        <Ring value={ratio * 100} max={100} color={sc} size={58} sw={4}>
          <span style={{ fontSize: 10, fontWeight: 700, color: sc,
            fontFamily: 'JetBrains Mono,monospace' }}>
            {(ratio * 100).toFixed(0)}%
          </span>
        </Ring>
        <div>
          <div style={{ fontSize: 14, fontWeight: 700, color: sc, marginBottom: 3 }}>
            {busy ? '…' : (result?.status ?? (eligible ? 'ELIGIBLE' : 'DEBARRED'))}
          </div>
          <div style={{ fontSize: 10.5, color: P.muted, lineHeight: 1.55, maxWidth: 210 }}>
            {result?.message ?? (eligible
              ? `Eligible. ${Math.floor((aEff - 0.75 * tEff) / 0.75)} safe skip(s) remaining.`
              : `Debarred. Attend ${Math.ceil((0.75 * tEff - aEff) / 0.25)} consecutive session(s).`)}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── 12-Month Academic Calendar ────────────────────────────────────────────────
const MONTHS_SHORT = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
const MONTHS_FULL  = ['January','February','March','April','May','June','July','August','September','October','November','December']
const DAYS_HDR     = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

// Sep 2026 – Aug 2027
const ACAD_MONTHS = [
  { y:2026,m:8 },{ y:2026,m:9 },{ y:2026,m:10 },{ y:2026,m:11 },
  { y:2027,m:0 },{ y:2027,m:1 },{ y:2027,m:2 },{ y:2027,m:3 },
  { y:2027,m:4 },{ y:2027,m:5 },{ y:2027,m:6 },{ y:2027,m:7 },
]

function daysInMonth(y, m) { return new Date(y, m + 1, 0).getDate() }
function firstDow(y, m)    { const d = new Date(y, m, 1).getDay(); return d === 0 ? 6 : d - 1 }
function dk(y, m, d)       { return `${y}-${String(m+1).padStart(2,'0')}-${String(d).padStart(2,'0')}` }

function TwelveMonthCalendar({ calendarSlots }) {
  const [mIdx,  setMIdx]      = useState(0)
  const [att,   setAtt]       = useState({})
  const [selDay, setSelDay]   = useState(null)
  const cursor = useCursorTrack()

  const { y, m } = ACAD_MONTHS[mIdx]
  const totalDays = daysInMonth(y, m)
  const firstDay  = firstDow(y, m)

  const slotsByDow = useMemo(() => {
    const map = {}
    ;(calendarSlots || []).forEach(slot => {
      const dow = DAYS_HDR.indexOf(slot.day?.slice(0, 3))
      if (dow >= 0) { if (!map[dow]) map[dow] = []; map[dow].push(slot) }
    })
    return map
  }, [calendarSlots])

  const toggleDay = (key) => {
    setAtt(prev => {
      const cur  = prev[key]
      const next = cur === 'present' ? 'absent' : cur === 'absent' ? undefined : 'present'
      const upd  = { ...prev }
      if (next === undefined) delete upd[key]; else upd[key] = next
      return upd
    })
    setSelDay(key === selDay ? null : key)
  }

  const cells = []
  for (let i = 0; i < firstDay; i++) cells.push(null)
  for (let d = 1; d <= totalDays; d++) {
    const key  = dk(y, m, d)
    const dow  = (firstDay + d - 1) % 7
    const slots = slotsByDow[dow] || []
    cells.push({ d, key, dow, slots, weekend: dow >= 5, att: att[key] })
  }

  const SLOT_C = { lecture: P.cyan, lab: P.purple, exam: P.amber, other: P.muted }

  // Monthly stats
  const monthKeys = cells.filter(Boolean).map(c => c.key)
  const pCount = monthKeys.filter(k => att[k] === 'present').length
  const aCount = monthKeys.filter(k => att[k] === 'absent').length
  const hasSlotDays = cells.filter(c => c && c.slots.length > 0).length

  return (
    <div className="glass-card p-5" {...cursor}>
      {/* Header row */}
      <div className="flex justify-between items-center mb-3">
        <div className="section-label mb-0">Academic Calendar 2026–27</div>
        <div className="flex items-center gap-2">
          <button onClick={() => setMIdx(i => Math.max(0, i-1))}
            disabled={mIdx === 0} style={navBtnStyle(mIdx === 0)}>‹</button>
          <span style={{ fontSize: 12, color: P.text, fontWeight: 500, minWidth: 110, textAlign: 'center' }}>
            {MONTHS_FULL[m]} {y}
          </span>
          <button onClick={() => setMIdx(i => Math.min(11, i+1))}
            disabled={mIdx === 11} style={navBtnStyle(mIdx === 11)}>›</button>
        </div>
      </div>

      {/* Month pill strip */}
      <div className="flex gap-1.5 mb-4 overflow-x-auto pb-1">
        {ACAD_MONTHS.map((am, i) => (
          <button key={i} onClick={() => setMIdx(i)} style={{
            padding: '3px 8px', borderRadius: 7, fontSize: 9.5, flexShrink: 0,
            background: i === mIdx ? 'rgba(0,240,255,0.12)' : 'rgba(255,255,255,0.03)',
            border: `1px solid ${i === mIdx ? 'rgba(0,240,255,0.35)' : 'rgba(255,255,255,0.07)'}`,
            color: i === mIdx ? P.cyan : P.muted, cursor: 'pointer',
          }}>{MONTHS_SHORT[am.m]}</button>
        ))}
      </div>

      {/* Day headers */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 2, marginBottom: 4 }}>
        {DAYS_HDR.map(d => (
          <div key={d} style={{ textAlign: 'center', fontSize: 8.5, color: P.muted,
            textTransform: 'uppercase', letterSpacing: 1.2, padding: '2px 0' }}>{d}</div>
        ))}
      </div>

      {/* Calendar grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 2 }}>
        {cells.map((c, i) => {
          if (!c) return <div key={i} />
          const { d, key, slots, weekend, att: cellAtt } = c
          const isSel   = selDay === key
          const hasSlots = slots.length > 0
          let bg = 'transparent', border = 'transparent'
          if (cellAtt === 'present') { bg = 'rgba(0,245,160,0.09)'; border = 'rgba(0,245,160,0.22)' }
          else if (cellAtt === 'absent') { bg = 'rgba(255,176,32,0.07)'; border = 'rgba(255,176,32,0.18)' }
          else if (hasSlots) { bg = 'rgba(255,255,255,0.03)'; border = 'rgba(255,255,255,0.07)' }
          if (isSel) { border = 'rgba(0,240,255,0.40)' }
          return (
            <div key={key} onClick={() => hasSlots && toggleDay(key)}
              title={hasSlots ? slots.map(s => `${s.title||s.course_code} (${s.slot_type})`).join(', ') : ''}
              style={{
                aspectRatio: '1', display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center',
                borderRadius: 8, background: bg, border: `1px solid ${border}`,
                cursor: hasSlots ? 'pointer' : 'default', transition: 'all 0.12s',
              }}>
              <span style={{ fontSize: 10.5, lineHeight: 1, fontWeight: hasSlots ? 600 : 400,
                color: weekend ? P.muted + '60'
                  : cellAtt === 'present' ? P.mint
                  : cellAtt === 'absent'  ? P.amber
                  : hasSlots ? P.text : P.muted + '50' }}>{d}</span>
              {hasSlots && (
                <div style={{ display: 'flex', gap: 1.5, marginTop: 2 }}>
                  {slots.slice(0, 3).map((s, si) => (
                    <div key={si} style={{ width: 3.5, height: 3.5, borderRadius: '50%',
                      background: SLOT_C[s.slot_type] || P.muted }} />
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Stats row */}
      <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
        {[
          { lbl: 'Scheduled', val: hasSlotDays, c: P.cyan },
          { lbl: 'Present', val: pCount, c: P.mint },
          { lbl: 'Absent',  val: aCount, c: P.amber },
        ].map(({ lbl, val, c }) => (
          <div key={lbl} style={{ flex: 1, textAlign: 'center', padding: '5px 8px',
            background: `${c}07`, border: `1px solid ${c}18`, borderRadius: 9 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: c,
              fontFamily: 'JetBrains Mono,monospace' }}>{val}</div>
            <div style={{ fontSize: 8.5, color: P.muted, textTransform: 'uppercase', letterSpacing: 1 }}>{lbl}</div>
          </div>
        ))}
      </div>

      {/* Selected day detail */}
      <AnimatePresence>
        {selDay && (() => {
          const dt   = new Date(selDay + 'T12:00:00')
          const dow  = (dt.getDay() + 6) % 7
          const slots = slotsByDow[dow] || []
          const attV = att[selDay]
          return (
            <motion.div key={selDay}
              initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }} transition={{ duration: 0.22 }}
              style={{ overflow: 'hidden' }}>
              <div style={{ marginTop: 10, padding: '10px 12px',
                background: 'rgba(0,0,0,0.18)', borderRadius: 11,
                border: '1px solid rgba(255,255,255,0.05)' }}>
                <div style={{ fontSize: 11.5, fontWeight: 600, color: P.text, marginBottom: 8 }}>
                  {dt.toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long' })}
                  {' '}
                  <span style={{
                    fontSize: 9.5, fontWeight: 700, borderRadius: 999, padding: '1px 7px',
                    color: attV === 'present' ? P.mint : attV === 'absent' ? P.amber : P.muted,
                    background: attV ? `${attV === 'present' ? P.mint : P.amber}12` : 'transparent',
                    border: `1px solid ${attV ? (attV === 'present' ? P.mint : P.amber) + '28' : 'transparent'}`,
                  }}>
                    {attV === 'present' ? 'Present' : attV === 'absent' ? 'Absent' : 'Unmarked'}
                  </span>
                </div>
                {slots.map((s, si) => (
                  <div key={si} style={{
                    display: 'flex', alignItems: 'center', gap: 8,
                    padding: '6px 10px', borderRadius: 8, marginBottom: 4,
                    background: `${SLOT_C[s.slot_type]||P.muted}09`,
                    border: `1px solid ${SLOT_C[s.slot_type]||P.muted}20`,
                  }}>
                    <div style={{ width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
                      background: SLOT_C[s.slot_type]||P.muted }} />
                    <span style={{ fontSize: 11, color: P.text, fontWeight: 500, flex: 1 }}>
                      {s.title || s.course_code || s.slot_type}
                    </span>
                    {s.start && <span style={{ fontSize: 9.5, color: P.muted }}>{s.start}–{s.end}</span>}
                    {s.room  && <span style={{ fontSize: 9.5, color: P.muted }}>{s.room}</span>}
                    <span style={{ fontSize: 9, color: SLOT_C[s.slot_type]||P.muted,
                      textTransform: 'capitalize' }}>{s.slot_type}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          )
        })()}
      </AnimatePresence>

      {calendarSlots.length === 0 && (
        <div style={{ textAlign: 'center', fontSize: 11, color: P.muted,
          padding: '8px 0 2px', borderTop: '1px solid rgba(255,255,255,0.04)', marginTop: 10 }}>
          Upload a timetable in the OCR tab to populate schedule
        </div>
      )}
    </div>
  )
}

function navBtnStyle(disabled) {
  return {
    background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)',
    borderRadius: 7, width: 26, height: 26,
    color: disabled ? 'rgba(74,85,104,0.35)' : P.muted,
    cursor: disabled ? 'default' : 'pointer', fontSize: 15,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  }
}

// ── OCR Dropzone (Camera + File) — Mobile Camera Capture ─────────────────────
function OCRDropzone({ dispatch }) {
  const [dragging, setDragging] = useState(false)
  const [status,   setStatus]   = useState(null)   // null | 'ocr' | 'parsing' | 'done' | 'error'
  const [msg,      setMsg]      = useState('')
  const [parsed,   setParsed]   = useState(null)
  const [progress, setProgress] = useState(0)
  const cameraRef = useRef(null)  // environment-facing camera capture
  const fileRef   = useRef(null)  // screenshot / PDF / XLSX
  const cursor    = useCursorTrack()

  const IMG_EXT = ['.png','.jpg','.jpeg','.webp']
  const DOC_EXT = ['.pdf','.xlsx','.xls','.csv','.txt']

  const processFile = async (file) => {
    if (!file) return
    const ext = ('.' + file.name.split('.').pop()).toLowerCase()
    setParsed(null); setStatus(null); setMsg(''); setProgress(0)

    if (IMG_EXT.includes(ext)) {
      setStatus('ocr'); setMsg('Running WASM OCR…')
      try {
        const { createWorker } = await import('tesseract.js')
        const worker = await createWorker('eng', 1, {
          workerPath: 'https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/worker.min.js',
          langPath:   'https://tessdata.projectnaptha.com/4.0.0',
          corePath:   'https://cdn.jsdelivr.net/npm/tesseract.js-core@5/tesseract-core-simd-lstm.wasm.js',
          logger: m => {
            if (m.status === 'recognizing text') {
              const pct = Math.round(m.progress * 100)
              setProgress(pct)
              setMsg(`OCR ${pct}%…`)
            }
          },
        })
        const { data: { text } } = await worker.recognize(file)
        await worker.terminate()
        if (!text.trim()) { setStatus('error'); setMsg('Empty OCR output — use a clearer photo or screenshot.'); return }
        setStatus('parsing'); setMsg('Parsing Bennett portal data…')
        const hint = text.toLowerCase().includes('attendance') ? 'attendance'
          : text.toLowerCase().includes('timetable') ? 'timetable'
          : text.toLowerCase().includes('roll no') ? 'profile' : 'unknown'
        const result = await apiOcr(text, hint)
        setParsed(result)
        if (result.periods_present != null && result.periods_held != null) {
          dispatch({ type: 'ATTENDANCE', v: {
            classes_attended: result.periods_present,
            classes_held:     result.periods_held,
            result: null,
          }})
        }
        setStatus('done')
        setMsg(result.confidence === 'high'
          ? `✓ ${result.parse_notes.join('; ')}`
          : `Partial — ${result.parse_notes.join('; ')}`)
      } catch (e) { setStatus('error'); setMsg(`OCR failed: ${e.message}`) }

    } else if (DOC_EXT.includes(ext)) {
      setStatus('parsing'); setMsg('Parsing document…')
      try {
        const result = await apiTimetable(file)
        dispatch({ type: 'CAL_SLOTS', v: result.slots })
        setStatus('done'); setMsg(`✓ ${result.slots.length} timetable slots extracted`)
      } catch (e) { setStatus('error'); setMsg(`Parse failed: ${e.message}`) }
    } else {
      setStatus('error'); setMsg(`Unsupported format '${ext}'`)
    }
  }

  const sc   = status === 'done' ? P.mint : status === 'error' ? P.amber : P.cyan
  const busy = status === 'ocr' || status === 'parsing'

  return (
    <div className="glass-card p-5" {...cursor}>
      <div className="section-label">Document / Screenshot / Camera</div>

      {/* ── Two trigger buttons: Camera + File ── */}
      <div className="ocr-btn-row flex gap-3 mb-4">
        {/* Camera capture — opens environment-facing camera on mobile */}
        <button className="camera-btn flex-1"
          onClick={() => cameraRef.current?.click()}
          disabled={busy}
          aria-label="Take photo with camera">
          <span style={{ fontSize: 18 }}>&#128247;</span>
          <span>Take Photo</span>
          <input ref={cameraRef} type="file" accept="image/*" capture="environment"
            className="hidden" style={{ display: 'none' }}
            onChange={e => processFile(e.target.files[0])} />
        </button>

        {/* File / screenshot upload */}
        <button className="file-btn flex-1"
          onClick={() => fileRef.current?.click()}
          disabled={busy}
          aria-label="Upload screenshot or document">
          <span style={{ fontSize: 18 }}>&#128196;</span>
          <span>Upload File</span>
          <input ref={fileRef} type="file"
            accept="image/*,.pdf,.xlsx,.xls,.csv,.txt"
            className="hidden" style={{ display: 'none' }}
            onChange={e => processFile(e.target.files[0])} />
        </button>
      </div>

      {/* ── Drag-and-drop zone ── */}
      <div
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => { e.preventDefault(); setDragging(false); processFile(e.dataTransfer.files[0]) }}
        style={{
          border: `1.5px dashed ${dragging ? P.cyan : 'rgba(255,255,255,0.09)'}`,
          borderRadius: 14, padding: '14px 12px', textAlign: 'center',
          background: dragging ? 'rgba(0,240,255,0.04)' : 'transparent',
          transition: 'all 0.2s',
        }}>
        <div style={{ fontSize: 11, color: P.muted, lineHeight: 1.6 }}>
          {busy ? msg : 'or drag & drop here'}
        </div>
        <div style={{ fontSize: 9.5, color: P.muted + '60', marginTop: 3 }}>
          PNG · JPG · PDF · XLSX · CSV · TXT
        </div>
      </div>

      {/* ── OCR progress bar ── */}
      {status === 'ocr' && (
        <div style={{ marginTop: 10, height: 3, borderRadius: 2,
          background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
          <div style={{ height: '100%', borderRadius: 2,
            width: `${progress}%`, background: P.cyan,
            transition: 'width 0.3s ease',
            boxShadow: `0 0 8px ${P.cyan}60` }} />
        </div>
      )}

      {/* ── Status message ── */}
      {msg && !busy && (
        <div style={{ marginTop: 8, fontSize: 11, color: sc,
          padding: '6px 12px', background: `${sc}07`,
          borderRadius: 8, border: `1px solid ${sc}18`, lineHeight: 1.55 }}>{msg}</div>
      )}

      {/* ── Parsed data chips ── */}
      {parsed && status === 'done' && (
        <div className="flex flex-wrap gap-1.5 mt-2">
          {parsed.student_name && <Chip lbl="Name"    val={parsed.student_name}          c={P.cyan}   />}
          {parsed.roll_no      && <Chip lbl="Roll"    val={parsed.roll_no}               c={P.purple} />}
          {parsed.attendance_pct != null && <Chip lbl="Att" val={`${parsed.attendance_pct}%`} c={P.mint} />}
          {parsed.periods_present != null && (
            <Chip lbl="Periods" val={`${parsed.periods_present}/${parsed.periods_held}`} c={P.amber} />
          )}
          {parsed.semester != null && <Chip lbl="Sem" val={`${parsed.semester}`} c={P.purple} />}
        </div>
      )}

      {/* ── Hint text ── */}
      <div style={{ marginTop: 10, fontSize: 9.5, color: P.muted + '80',
        lineHeight: 1.55, textAlign: 'center' }}>
        Camera: photos attendance kiosks directly · WASM OCR runs in-browser—nothing uploaded
      </div>
    </div>
  )
}


const Chip = ({ lbl, val, c }) => (
  <div style={{ padding: '3px 8px', borderRadius: 999, fontSize: 10,
    background: `${c}0D`, border: `1px solid ${c}22`, color: c,
    fontFamily: 'JetBrains Mono,monospace' }}>
    <span style={{ color: P.muted }}>{lbl}: </span>{val}
  </div>
)

// ── Roster Strip ──────────────────────────────────────────────────────────────
function RosterStrip({ students, activeId, onSelect, onOpenRoster }) {
  const active = students.find(s => s.student_id === activeId)
  return (
    <div className="glass-card px-5 py-3.5 flex items-center gap-4 flex-wrap">
      <div className="section-label mb-0 flex-shrink-0">Students</div>
      <div className="flex items-center gap-2.5 flex-1 flex-wrap">
        {students.map(s => (
          <Avatar key={s.student_id} name={s.student_name} color={s.avatar_color}
            size={32} active={s.student_id === activeId}
            onClick={() => onSelect(s.student_id)} />
        ))}
        {active && (
          <div style={{ marginLeft: 6 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: P.text }}>{active.student_name}</div>
            {active.roll_no && <div style={{ fontSize: 10, color: P.muted }}>{active.roll_no}</div>}
          </div>
        )}
      </div>
      <button onClick={onOpenRoster} style={{
        padding: '6px 14px', borderRadius: 9, fontSize: 11.5, fontWeight: 600,
        background: 'rgba(0,240,255,0.07)', border: '1px solid rgba(0,240,255,0.25)',
        color: P.cyan, cursor: 'pointer', whiteSpace: 'nowrap',
      }}>Manage</button>
    </div>
  )
}

// ── Tab Bar ───────────────────────────────────────────────────────────────────
const TABS = [
  { id: 'academic',    lbl: 'Academic'    },
  { id: 'attendance',  lbl: 'Attendance'  },
  { id: 'calendar',    lbl: 'Calendar'    },
  { id: 'ocr',         lbl: 'OCR / Docs'  },
]

function TabBar({ active, onChange }) {
  return (
    <div className="tab-bar">
      {TABS.map(t => (
        <button key={t.id} className={`tab-item ${active === t.id ? 'active' : ''}`}
          onClick={() => onChange(t.id)}>{t.lbl}</button>
      ))}
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function Dashboard() {
  const [state, dispatch]       = useReducer(reducer, INIT)
  const [evaluations, setEvals] = useState({})
  const evalTimer               = useRef(null)

  // SSOT evaluation — debounced 280ms on any score field change
  const runEval = useCallback(async (s) => {
    dispatch({ type: 'EVAL_START' })
    try {
      const res = await apiEvaluate({
        lab_marks:           s.lab_marks,
        n_labs_total:        s.n_labs_total,
        midterm_score:       s.midterm_score,
        endsem_score:        s.endsem_score,
        weekly_study_hours:  s.weekly_study_hours,
      })
      dispatch({ type: 'EVAL_DONE', r: res })
      if (s.activeStudentId) setEvals(p => ({ ...p, [s.activeStudentId]: res }))
    } catch (e) {
      dispatch({ type: 'EVAL_ERR', e: e.message })
    }
  }, [])

  useEffect(() => {
    clearTimeout(evalTimer.current)
    evalTimer.current = setTimeout(() => runEval(state), 280)
    return () => clearTimeout(evalTimer.current)
  }, [state.lab_marks, state.n_labs_total, state.midterm_score, state.endsem_score, state.weekly_study_hours])

  // Health polling every 15s
  useEffect(() => {
    const check = async () => {
      try { dispatch({ type: 'HEALTH', v: await apiHealth() }) }
      catch  { dispatch({ type: 'DISCONNECTED' }) }
    }
    check()
    const t = setInterval(check, 15000)
    return () => clearInterval(t)
  }, [])

  // Load roster
  useEffect(() => {
    apiStudents().then(list => {
      dispatch({ type: 'STUDENTS', v: list })
      if (list.length) dispatch({ type: 'ACTIVE_SID', id: list[0].student_id })
    }).catch(() => {})
  }, [])

  const onAddStudent = async (form) => {
    const s = await apiAddStudent(form)
    s.avatar_color = avatarColor(s.student_name)
    dispatch({ type: 'ADD_STUDENT', v: s })
    dispatch({ type: 'ACTIVE_SID', id: s.student_id })
  }
  const onDelStudent = async (id) => {
    await apiDelStudent(id)
    dispatch({ type: 'DEL_STUDENT', id })
    setEvals(p => { const n = { ...p }; delete n[id]; return n })
  }

  const ev = state.evaluation
  const { activeTab } = state

  return (
    <div className="min-h-screen w-full antialiased overflow-x-hidden"
      style={{ background: P.bg, color: P.text, fontFamily: 'Inter,sans-serif' }}>

      <div className="w-full max-w-[1600px] mx-auto px-4 py-4 md:px-6 md:py-6 flex flex-col gap-5">

        {/* ── Header (ONLY title + badge, nothing else) ── */}
        <header className="text-center pt-2 pb-1">
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 300, color: P.text,
            letterSpacing: -0.5, fontFamily: 'Inter,sans-serif' }}>
            Student Management System
          </h1>
          <ConnectionBadge connected={state.connected} ping={state.health?.ping_ms} />
        </header>

        {/* Roster strip */}
        <RosterStrip
          students={state.students} activeId={state.activeStudentId}
          onSelect={id => dispatch({ type: 'ACTIVE_SID', id })}
          onOpenRoster={() => dispatch({ type: 'ROSTER_OPEN', v: true })} />

        {/* Tab bar */}
        <TabBar active={activeTab} onChange={tab => dispatch({ type: 'TAB', v: tab })} />

        {/* ── Responsive grid: single-col mobile → xl:grid-cols-12 desktop ── */}
        <div className="flex flex-col xl:grid xl:grid-cols-12 gap-4 xl:gap-5 xl:items-start">

          {/* ── Col 1 (xl:col-span-4): Hero Gauge + rings + bento ── */}
          <div className="xl:col-span-4 flex flex-col gap-3">
            <div className="hero-gauge-wrap">
              <HeroGauge
                score={ev?.composite_score}
                letter={ev?.letter_grade}
                ci={ev?.conformal_interval}
                loading={state.evaluating}
              />
            </div>
            <ScoreRings ev={ev} />
            <BentoCards
              students={state.students} evaluations={evaluations}
              onOpenRoster={() => dispatch({ type: 'ROSTER_OPEN', v: true })} />
          </div>

          {/* ── Col 2 (xl:col-span-5): Tab content ── */}
          <div className="xl:col-span-5 flex flex-col gap-3">
            {activeTab === 'academic' && (
              <AcademicControls state={state} dispatch={dispatch} />
            )}
            {activeTab === 'attendance' && (
              <ExamEligibilityPanel attendance={state.attendance} dispatch={dispatch} />
            )}
            {activeTab === 'calendar' && (
              <TwelveMonthCalendar calendarSlots={state.calendarSlots} />
            )}
            {activeTab === 'ocr' && (
              <OCRDropzone dispatch={dispatch} />
            )}
          </div>

          {/* ── Col 3 (xl:col-span-3): Prescription ── */}
          <div className="xl:col-span-3 flex flex-col gap-3">
            <FiveTrackPrescription state={state} dispatch={dispatch} />
          </div>
        </div>

        {/* ── Footer ── */}
        <footer className="safe-bottom" style={{ textAlign: 'center', paddingTop: 32, paddingBottom: 8,
          fontSize: 10, color: '#1A2035' }}>
          SMS-MH-AVIE FINAL-PROD · UCI DOI 10.24432/C5MC89 · Outputs are advisory only
        </footer>
      </div>

      {/* ── Genie Roster Drawer ── */}
      <RosterDrawer
        open={state.rosterOpen}
        students={state.students}
        activeId={state.activeStudentId}
        evaluations={evaluations}
        onClose={() => dispatch({ type: 'ROSTER_OPEN', v: false })}
        onSelect={id => dispatch({ type: 'ACTIVE_SID', id })}
        onAdd={onAddStudent}
        onDelete={onDelStudent}
      />
    </div>
  )
}
