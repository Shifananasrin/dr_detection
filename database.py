"""
Database module for Diabetic Retinopathy Clinical Decision Support System
Uses SQLite for portability.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = "dr_system.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_db()
    cur = conn.cursor()

    # Users table (patients and doctors)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            email       TEXT    UNIQUE NOT NULL,
            password    TEXT    NOT NULL,
            role        TEXT    NOT NULL CHECK(role IN ('patient','doctor')),
            specialty   TEXT,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # Patient clinical profiles
    cur.execute("""
        CREATE TABLE IF NOT EXISTS patient_profiles (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id            INTEGER UNIQUE NOT NULL REFERENCES users(id),
            date_of_birth      TEXT,
            gender             TEXT,
            diabetes_type      TEXT,
            diabetes_duration  REAL,
            hba1c              REAL,
            systolic_bp        INTEGER,
            diastolic_bp       INTEGER,
            cholesterol        REAL,
            bmi                REAL,
            smoking            TEXT,
            insulin_use        INTEGER DEFAULT 0,
            notes              TEXT,
            updated_at         TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # Scan / Prediction records
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id       INTEGER NOT NULL REFERENCES users(id),
            image_path       TEXT    NOT NULL,
            image_url        TEXT    NOT NULL,
            uploaded_at      TEXT    NOT NULL DEFAULT (datetime('now')),

            -- AI predictions
            ai_class_id      INTEGER,
            ai_class_name    TEXT,
            ai_confidence    REAL,
            ai_probs         TEXT,   -- JSON string
            gradcam_b64      TEXT,
            prob_chart_b64   TEXT,
            risk_score       REAL,
            risk_level       TEXT,
            risk_factors     TEXT,   -- JSON string

            -- Doctor validation
            status           TEXT NOT NULL DEFAULT 'pending'
                             CHECK(status IN ('pending','reviewed','approved','modified')),
            doctor_id        INTEGER REFERENCES users(id),
            doctor_class_id  INTEGER,
            doctor_class_name TEXT,
            doctor_notes     TEXT,
            treatment        TEXT,
            followup_weeks   INTEGER,
            verified_at      TEXT
        )
    """)

    # Notifications / messages
    cur.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL REFERENCES users(id),
            message     TEXT    NOT NULL,
            is_read     INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()

    # Seed demo accounts
    _seed_demo_data()


def _seed_demo_data():
    """Insert demo patient and doctor if the DB is fresh."""
    from werkzeug.security import generate_password_hash

    conn = get_db()
    cur = conn.cursor()

    # Check if already seeded
    if cur.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0:
        conn.close()
        return

    # Demo doctor
    cur.execute("""
        INSERT INTO users (name, email, password, role, specialty)
        VALUES (?,?,?,?,?)
    """, ("Dr. Sarah Mitchell", "doctor@retina.ai",
          generate_password_hash("doctor123"), "doctor", "Ophthalmology"))

    # Demo patient
    cur.execute("""
        INSERT INTO users (name, email, password, role)
        VALUES (?,?,?,?)
    """, ("John Patient", "patient@retina.ai",
          generate_password_hash("patient123"), "patient"))

    patient_id = cur.lastrowid
    cur.execute("""
        INSERT INTO patient_profiles
            (user_id, date_of_birth, gender, diabetes_type, diabetes_duration,
             hba1c, systolic_bp, diastolic_bp, cholesterol, bmi, smoking, insulin_use)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (patient_id, "1978-04-12", "Male", "Type 2", 8,
          7.8, 138, 85, 5.2, 27.4, "Never", 0))

    conn.commit()
    conn.close()


# ─── RISK SCORE ───────────────────────────────────────────────────────────────

RISK_WEIGHTS = {
    "hba1c":             {"low": 7.0, "high": 9.0, "max_pts": 25},
    "diabetes_duration": {"low": 5,   "high": 15,  "max_pts": 20},
    "systolic_bp":       {"low": 130, "high": 160, "max_pts": 15},
    "cholesterol":       {"low": 5.0, "high": 7.0, "max_pts": 10},
    "bmi":               {"low": 25,  "high": 35,  "max_pts": 10},
    "smoking":           {"Never": 0, "Former": 5, "Current": 10},
    "insulin_use":       {True: 5, False: 0},
    "ai_grade":          {0: 0, 1: 10, 2: 25, 3: 40, 4: 55},
}


def compute_risk(profile: dict, ai_class_id: int) -> dict:
    """Compute a 0-100 risk score and return level + contributing factors."""
    score = 0
    factors = []

    def scale(val, low, high, max_pts):
        if val is None:
            return 0
        if val <= low:
            return 0
        if val >= high:
            return max_pts
        return round(max_pts * (val - low) / (high - low), 1)

    # HbA1c
    if profile.get("hba1c"):
        pts = scale(profile["hba1c"], 7.0, 9.0, 25)
        score += pts
        if pts > 8:
            factors.append(f"High HbA1c ({profile['hba1c']}%)")

    # Diabetes duration
    if profile.get("diabetes_duration"):
        pts = scale(profile["diabetes_duration"], 5, 15, 20)
        score += pts
        if pts > 8:
            factors.append(f"Long diabetes duration ({int(profile['diabetes_duration'])} yrs)")

    # Blood pressure
    if profile.get("systolic_bp"):
        pts = scale(profile["systolic_bp"], 130, 160, 15)
        score += pts
        if pts > 5:
            factors.append(f"Elevated BP ({profile['systolic_bp']}/{profile.get('diastolic_bp','-')})")

    # Cholesterol
    if profile.get("cholesterol"):
        pts = scale(profile["cholesterol"], 5.0, 7.0, 10)
        score += pts
        if pts > 3:
            factors.append(f"High cholesterol ({profile['cholesterol']} mmol/L)")

    # BMI
    if profile.get("bmi"):
        pts = scale(profile["bmi"], 25, 35, 10)
        score += pts
        if pts > 3:
            factors.append(f"Elevated BMI ({profile['bmi']})")

    # Smoking
    smoking_map = {"Never": 0, "Former": 5, "Current": 10}
    smoking_pts = smoking_map.get(profile.get("smoking", "Never"), 0)
    score += smoking_pts
    if smoking_pts >= 5:
        factors.append(f"Smoking status: {profile.get('smoking')}")

    # Insulin use
    if profile.get("insulin_use"):
        score += 5

    # AI grade (dominant factor)
    grade_pts = {0: 0, 1: 10, 2: 25, 3: 40, 4: 55}.get(ai_class_id, 0)
    score += grade_pts

    # Cap at 100
    score = min(round(score), 100)

    if score < 20:
        level = "Low"
    elif score < 45:
        level = "Moderate"
    elif score < 70:
        level = "High"
    else:
        level = "Very High"

    return {"score": score, "level": level, "factors": factors}


# ─── CRUD HELPERS ─────────────────────────────────────────────────────────────

def get_user_by_email(email: str):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    return user


def get_user_by_id(uid: int):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    return user


def get_patient_profile(user_id: int):
    conn = get_db()
    p = conn.execute("SELECT * FROM patient_profiles WHERE user_id=?", (user_id,)).fetchone()
    conn.close()
    return p


def get_patient_scans(patient_id: int):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM scans WHERE patient_id=? ORDER BY uploaded_at DESC",
        (patient_id,)
    ).fetchall()
    conn.close()
    return rows


def get_all_pending_scans():
    conn = get_db()
    rows = conn.execute("""
        SELECT s.*, u.name as patient_name, u.email as patient_email
        FROM scans s
        JOIN users u ON s.patient_id = u.id
        WHERE s.status='pending'
        ORDER BY s.uploaded_at ASC
    """).fetchall()
    conn.close()
    return rows


def get_all_reviewed_scans():
    conn = get_db()
    rows = conn.execute("""
        SELECT s.*, u.name as patient_name
        FROM scans s
        JOIN users u ON s.patient_id = u.id
        WHERE s.status != 'pending'
        ORDER BY s.verified_at DESC
        LIMIT 50
    """).fetchall()
    conn.close()
    return rows


def get_scan_by_id(scan_id: int):
    conn = get_db()
    row = conn.execute("""
        SELECT s.*,
               u.name as patient_name, u.email as patient_email,
               pp.date_of_birth, pp.gender, pp.diabetes_type,
               pp.diabetes_duration, pp.hba1c, pp.systolic_bp,
               pp.diastolic_bp, pp.cholesterol, pp.bmi,
               pp.smoking, pp.insulin_use,
               d.name as doctor_name, d.specialty as doctor_specialty
        FROM scans s
        JOIN users u  ON s.patient_id = u.id
        LEFT JOIN patient_profiles pp ON u.id = pp.user_id
        LEFT JOIN users d ON s.doctor_id = d.id
        WHERE s.id=?
    """, (scan_id,)).fetchone()
    conn.close()
    return row


def save_scan(patient_id, image_path, image_url, ai_result, risk):
    import json
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO scans
            (patient_id, image_path, image_url,
             ai_class_id, ai_class_name, ai_confidence, ai_probs,
             gradcam_b64, prob_chart_b64,
             risk_score, risk_level, risk_factors)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        patient_id, image_path, image_url,
        ai_result["class_id"], ai_result["class_name"],
        ai_result["confidence"], json.dumps(ai_result["probs"]),
        ai_result["gradcam_b64"], ai_result["prob_chart_b64"],
        risk["score"], risk["level"], json.dumps(risk["factors"]),
    ))
    scan_id = cur.lastrowid

    # Notify doctors
    doctors = conn.execute("SELECT id FROM users WHERE role='doctor'").fetchall()
    for doc in doctors:
        conn.execute("""
            INSERT INTO notifications (user_id, message)
            VALUES (?, ?)
        """, (doc["id"], f"New scan #{scan_id} awaiting review"))

    conn.commit()
    conn.close()
    return scan_id


def save_doctor_review(scan_id, doctor_id, data: dict):
    conn = get_db()
    conn.execute("""
        UPDATE scans SET
            status           = ?,
            doctor_id        = ?,
            doctor_class_id  = ?,
            doctor_class_name= ?,
            doctor_notes     = ?,
            treatment        = ?,
            followup_weeks   = ?,
            verified_at      = datetime('now')
        WHERE id=?
    """, (
        data["status"], doctor_id,
        data["doctor_class_id"], data["doctor_class_name"],
        data["doctor_notes"], data["treatment"],
        data.get("followup_weeks"), scan_id
    ))

    # Get patient_id for notification
    scan = conn.execute("SELECT patient_id FROM scans WHERE id=?", (scan_id,)).fetchone()
    if scan:
        add_notification(scan["patient_id"], f"Your scan #{scan_id} has been reviewed by a doctor.")

    conn.commit()
    conn.close()


def mark_notifications_read(user_id):
    conn = get_db()
    conn.execute("UPDATE notifications SET is_read=1 WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


def get_unread_count(user_id):
    conn = get_db()
    n = conn.execute(
        "SELECT COUNT(*) FROM notifications WHERE user_id=? AND is_read=0",
        (user_id,)
    ).fetchone()[0]
    conn.close()
    return n


def add_notification(user_id, message):
    """Adds a notification for a user. If an unread notification with the same message exists, updates its timestamp."""
    conn = get_db()
    try:
        # Check for existing unread duplicate
        existing = conn.execute(
            "SELECT id FROM notifications WHERE user_id=? AND message=? AND is_read=0",
            (user_id, message)
        ).fetchone()
        
        if existing:
            conn.execute(
                "UPDATE notifications SET created_at=datetime('now') WHERE id=?",
                (existing[0],)
            )
        else:
            conn.execute(
                "INSERT INTO notifications (user_id, message) VALUES (?, ?)",
                (user_id, message)
            )
        conn.commit()
    finally:
        conn.close()


def get_notifications(user_id, limit=20, only_unread=False):
    conn = get_db()
    query = "SELECT * FROM notifications WHERE user_id=? "
    if only_unread:
        query += " AND is_read=0 "
    query += " ORDER BY created_at DESC LIMIT ?"
    rows = conn.execute(query, (user_id, limit)).fetchall()
    conn.close()
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — APPOINTMENT & FOLLOW-UP MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def init_appointments():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS appointments (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id      INTEGER NOT NULL REFERENCES users(id),
            doctor_id       INTEGER REFERENCES users(id),
            scan_id         INTEGER REFERENCES scans(id),
            appointment_date TEXT NOT NULL,
            appointment_time TEXT NOT NULL,
            type            TEXT NOT NULL DEFAULT 'Follow-up'
                        CHECK(type IN ('Initial Consultation','Follow-up','Laser Therapy',
                                       'Injection Session','Post-treatment Review','Emergency')),
            status          TEXT NOT NULL DEFAULT 'scheduled'
                        CHECK(status IN ('scheduled','confirmed','completed','cancelled','no-show')),
            patient_notes   TEXT,
            doctor_notes    TEXT,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS appointment_slots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id   INTEGER NOT NULL REFERENCES users(id),
            slot_date   TEXT NOT NULL,
            slot_time   TEXT NOT NULL,
            is_booked   INTEGER NOT NULL DEFAULT 0,
            UNIQUE(doctor_id, slot_date, slot_time)
        );
    """)
    conn.commit()
    conn.close()


def get_patient_appointments(patient_id):
    conn = get_db()
    rows = conn.execute("""
        SELECT a.*, u.name as doctor_name, u.specialty as doctor_specialty
        FROM appointments a
        LEFT JOIN users u ON a.doctor_id = u.id
        WHERE a.patient_id=?
        ORDER BY a.appointment_date DESC, a.appointment_time DESC
    """, (patient_id,)).fetchall()
    conn.close()
    return rows


def get_doctor_appointments(doctor_id, date_filter=None):
    conn = get_db()
    if date_filter:
        rows = conn.execute("""
            SELECT a.*, u.name as patient_name, u.email as patient_email
            FROM appointments a
            JOIN users u ON a.patient_id = u.id
            WHERE a.doctor_id=? AND a.appointment_date=?
            ORDER BY a.appointment_time
        """, (doctor_id, date_filter)).fetchall()
    else:
        rows = conn.execute("""
            SELECT a.*, u.name as patient_name, u.email as patient_email
            FROM appointments a
            JOIN users u ON a.patient_id = u.id
            WHERE a.doctor_id=?
            ORDER BY a.appointment_date DESC, a.appointment_time
        """, (doctor_id,)).fetchall()
    conn.close()
    return rows


def get_appointment_by_id(appt_id):
    conn = get_db()
    row = conn.execute("""
        SELECT a.*,
               p.name as patient_name, p.email as patient_email,
               d.name as doctor_name, d.specialty as doctor_specialty
        FROM appointments a
        JOIN users p ON a.patient_id = p.id
        LEFT JOIN users d ON a.doctor_id = d.id
        WHERE a.id=?
    """, (appt_id,)).fetchone()
    conn.close()
    return row


def book_appointment(patient_id, data: dict):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO appointments
            (patient_id, doctor_id, scan_id, appointment_date, appointment_time,
             type, status, patient_notes)
        VALUES (?,?,?,?,?,?,?,?)
    """, (
        patient_id, data.get("doctor_id") or None,
        data.get("scan_id") or None,
        data["appointment_date"], data["appointment_time"],
        data.get("type", "Follow-up"), "scheduled",
        data.get("patient_notes", ""),
    ))
    appt_id = cur.lastrowid

    # Notify the assigned doctor
    if data.get("doctor_id"):
        add_notification(data["doctor_id"], 
                         f"New appointment #{appt_id} booked for {data['appointment_date']} at {data['appointment_time']}")
    conn.commit()
    conn.close()
    return appt_id


def update_appointment(appt_id, data: dict):
    conn = get_db()
    conn.execute("""
        UPDATE appointments SET
            status=?, doctor_notes=?, updated_at=datetime('now')
        WHERE id=?
    """, (data.get("status"), data.get("doctor_notes", ""), appt_id))

    # Notify patient on status change
    appt = conn.execute("SELECT patient_id, status FROM appointments WHERE id=?", (appt_id,)).fetchone()
    if appt:
        add_notification(appt["patient_id"], f"Appointment #{appt_id} status updated to: {data.get('status')}")
    conn.commit()
    conn.close()


def get_all_doctors():
    conn = get_db()
    rows = conn.execute("SELECT * FROM users WHERE role='doctor' ORDER BY name").fetchall()
    conn.close()
    return rows


def get_appointment_stats(doctor_id=None):
    conn = get_db()
    if doctor_id:
        base = "WHERE doctor_id=?"
        args = (doctor_id,)
    else:
        base = ""
        args = ()
    stats = {}
    for status in ["scheduled", "confirmed", "completed", "cancelled", "no-show"]:
        q = f"SELECT COUNT(*) FROM appointments {base} {'AND' if base else 'WHERE'} status=?"
        stats[status] = conn.execute(q, args + (status,)).fetchone()[0]
    stats["total"] = sum(stats.values())
    conn.close()
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — MEDICAL REPORT & DOCUMENT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def init_documents():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id   INTEGER NOT NULL REFERENCES users(id),
            scan_id      INTEGER REFERENCES scans(id),
            uploaded_by  INTEGER NOT NULL REFERENCES users(id),
            filename     TEXT NOT NULL,
            file_path    TEXT NOT NULL,
            file_url     TEXT NOT NULL,
            doc_type     TEXT NOT NULL DEFAULT 'Other'
                     CHECK(doc_type IN ('Lab Report','Prescription','Referral Letter',
                                        'Previous Scan','Insurance','AI Report','Other')),
            description  TEXT,
            file_size    INTEGER,
            uploaded_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS generated_reports (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id   INTEGER NOT NULL REFERENCES users(id),
            scan_id      INTEGER REFERENCES scans(id),
            doctor_id    INTEGER REFERENCES users(id),
            report_type  TEXT NOT NULL DEFAULT 'Diagnostic Report',
            file_path    TEXT NOT NULL,
            file_url     TEXT NOT NULL,
            generated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


def get_patient_documents(patient_id):
    conn = get_db()
    rows = conn.execute("""
        SELECT d.*, u.name as uploader_name
        FROM documents d
        JOIN users u ON d.uploaded_by = u.id
        WHERE d.patient_id=?
        ORDER BY d.uploaded_at DESC
    """, (patient_id,)).fetchall()
    conn.close()
    return rows


def get_document_by_id(doc_id):
    conn = get_db()
    row = conn.execute("""
        SELECT d.*, u.name as patient_name
        FROM documents d
        JOIN users p ON d.patient_id = p.id
        JOIN users u ON d.uploaded_by = u.id
        WHERE d.id=?
    """, (doc_id,)).fetchone()
    conn.close()
    return row


def save_document(patient_id, uploaded_by, scan_id, file_path, file_url,
                  filename, doc_type, description, file_size):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO documents
            (patient_id, uploaded_by, scan_id, filename, file_path, file_url,
             doc_type, description, file_size)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (patient_id, uploaded_by, scan_id or None,
          filename, file_path, file_url, doc_type, description, file_size))
    doc_id = cur.lastrowid
    conn.commit()
    conn.close()
    return doc_id


def delete_document(doc_id):
    conn = get_db()
    doc = conn.execute("SELECT file_path FROM documents WHERE id=?", (doc_id,)).fetchone()
    conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    conn.commit()
    conn.close()
    return doc["file_path"] if doc else None


def save_generated_report(patient_id, scan_id, doctor_id, report_type, file_path, file_url):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO generated_reports
            (patient_id, scan_id, doctor_id, report_type, file_path, file_url)
        VALUES (?,?,?,?,?,?)
    """, (patient_id, scan_id, doctor_id, report_type, file_path, file_url))
    rid = cur.lastrowid
    conn.commit()
    conn.close()
    return rid


def get_patient_reports(patient_id):
    conn = get_db()
    rows = conn.execute("""
        SELECT r.*, d.name as doctor_name
        FROM generated_reports r
        LEFT JOIN users d ON r.doctor_id = d.id
        WHERE r.patient_id=?
        ORDER BY r.generated_at DESC
    """, (patient_id,)).fetchall()
    conn.close()
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 7 — DOCTOR SCHEDULE & WORKLOAD MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def init_schedules():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS doctor_availability (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id   INTEGER NOT NULL REFERENCES users(id),
            day_of_week INTEGER NOT NULL,   -- 0=Mon … 6=Sun
            start_time  TEXT NOT NULL,
            end_time    TEXT NOT NULL,
            max_patients INTEGER NOT NULL DEFAULT 10,
            is_active   INTEGER NOT NULL DEFAULT 1,
            UNIQUE(doctor_id, day_of_week)
        );
    """)
    conn.commit()
    conn.close()


def get_doctor_availability(doctor_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM doctor_availability WHERE doctor_id=? ORDER BY day_of_week",
        (doctor_id,)
    ).fetchall()
    conn.close()
    return rows


def save_doctor_availability(doctor_id, schedules: list):
    """schedules = [{day_of_week, start_time, end_time, max_patients, is_active}]"""
    conn = get_db()
    for s in schedules:
        conn.execute("""
            INSERT INTO doctor_availability
                (doctor_id, day_of_week, start_time, end_time, max_patients, is_active)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(doctor_id, day_of_week) DO UPDATE SET
                start_time=excluded.start_time,
                end_time=excluded.end_time,
                max_patients=excluded.max_patients,
                is_active=excluded.is_active
        """, (doctor_id, s["day_of_week"], s["start_time"],
              s["end_time"], s.get("max_patients", 10), s.get("is_active", 1)))
    conn.commit()
    conn.close()


def get_doctor_workload_stats(doctor_id):
    conn = get_db()
    stats = {}

    # Scan review stats
    stats["total_reviewed"] = conn.execute(
        "SELECT COUNT(*) FROM scans WHERE doctor_id=?", (doctor_id,)
    ).fetchone()[0]
    stats["approved"] = conn.execute(
        "SELECT COUNT(*) FROM scans WHERE doctor_id=? AND status='approved'", (doctor_id,)
    ).fetchone()[0]
    stats["modified"] = conn.execute(
        "SELECT COUNT(*) FROM scans WHERE doctor_id=? AND status='modified'", (doctor_id,)
    ).fetchone()[0]
    stats["pending_assigned"] = conn.execute(
        "SELECT COUNT(*) FROM scans WHERE status='pending'"
    ).fetchone()[0]

    # Appointment stats
    stats["appt_total"] = conn.execute(
        "SELECT COUNT(*) FROM appointments WHERE doctor_id=?", (doctor_id,)
    ).fetchone()[0]
    stats["appt_upcoming"] = conn.execute(
        "SELECT COUNT(*) FROM appointments WHERE doctor_id=? AND status IN ('scheduled','confirmed') AND appointment_date >= date('now')",
        (doctor_id,)
    ).fetchone()[0]
    stats["appt_today"] = conn.execute(
        "SELECT COUNT(*) FROM appointments WHERE doctor_id=? AND appointment_date=date('now')",
        (doctor_id,)
    ).fetchone()[0]

    # Reviews per day — full 7-day window with 0s for missing days
    raw_reviews = conn.execute("""
        SELECT DATE(verified_at) as day, COUNT(*) as cnt
        FROM scans WHERE doctor_id=? AND verified_at IS NOT NULL
          AND verified_at >= date('now', '-6 days')
        GROUP BY DATE(verified_at) ORDER BY day
    """, (doctor_id,)).fetchall()
    # Build full 7-day list so chart never has gaps
    from datetime import date as _date, timedelta as _td
    review_map = {row["day"]: row["cnt"] for row in raw_reviews}
    daily = []
    for offset in range(6, -1, -1):
        d = (_date.today() - _td(days=offset)).isoformat()
        day_label = (_date.today() - _td(days=offset)).strftime("%a")
        daily.append({"day": day_label, "date": d, "cnt": review_map.get(d, 0)})
    stats["daily_reviews"] = daily

    # Average turnaround hours (from uploaded_at to verified_at)
    avg = conn.execute("""
        SELECT AVG((JULIANDAY(verified_at) - JULIANDAY(uploaded_at)) * 24)
        FROM scans WHERE doctor_id=? AND verified_at IS NOT NULL
    """, (doctor_id,)).fetchone()[0]
    stats["avg_turnaround_hrs"] = round(avg, 1) if avg else None

    # Grade distribution of reviewed scans
    stats["grade_dist"] = conn.execute("""
        SELECT doctor_class_id as grade, COUNT(*) as cnt
        FROM scans WHERE doctor_id=? AND doctor_class_id IS NOT NULL
        GROUP BY doctor_class_id ORDER BY doctor_class_id
    """, (doctor_id,)).fetchall()

    conn.close()
    return stats


def get_all_doctors_workload():
    """For admin-style overview of all doctor loads."""
    conn = get_db()
    rows = conn.execute("""
        SELECT u.id, u.name, u.specialty,
               COUNT(DISTINCT s.id)  as total_reviews,
               COUNT(DISTINCT a.id)  as total_appointments,
               SUM(CASE WHEN s.status='pending' THEN 1 ELSE 0 END) as pending_scans
        FROM users u
        LEFT JOIN scans s ON s.doctor_id = u.id
        LEFT JOIN appointments a ON a.doctor_id = u.id
        WHERE u.role='doctor'
        GROUP BY u.id
    """).fetchall()
    conn.close()
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 10 — POPULATION HEALTH ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

def get_population_analytics():
    conn = get_db()
    data = {}

    # DR grade distribution
    data["grade_dist"] = conn.execute("""
        SELECT ai_class_name, COUNT(*) as cnt
        FROM scans WHERE ai_class_id IS NOT NULL
        GROUP BY ai_class_id ORDER BY ai_class_id
    """).fetchall()

    # Risk level distribution
    data["risk_dist"] = conn.execute("""
        SELECT risk_level, COUNT(*) as cnt
        FROM scans WHERE risk_level IS NOT NULL
        GROUP BY risk_level
    """).fetchall()

    # Monthly scan volume (last 12 months)
    data["monthly_scans"] = conn.execute("""
        SELECT strftime('%Y-%m', uploaded_at) as month, COUNT(*) as cnt
        FROM scans
        WHERE uploaded_at >= datetime('now', '-12 months')
        GROUP BY month ORDER BY month
    """).fetchall()

    # Gender distribution from patient profiles
    data["gender_dist"] = conn.execute("""
        SELECT pp.gender, COUNT(*) as cnt
        FROM patient_profiles pp
        WHERE pp.gender IS NOT NULL
        GROUP BY pp.gender
    """).fetchall()

    # Diabetes type distribution
    data["diabetes_dist"] = conn.execute("""
        SELECT diabetes_type, COUNT(*) as cnt
        FROM patient_profiles WHERE diabetes_type IS NOT NULL
        GROUP BY diabetes_type
    """).fetchall()

    # Avg HbA1c by DR grade
    data["hba1c_by_grade"] = conn.execute("""
        SELECT s.ai_class_id, s.ai_class_name, ROUND(AVG(pp.hba1c),2) as avg_hba1c
        FROM scans s
        JOIN patient_profiles pp ON s.patient_id = pp.user_id
        WHERE pp.hba1c IS NOT NULL AND s.ai_class_id IS NOT NULL
        GROUP BY s.ai_class_id ORDER BY s.ai_class_id
    """).fetchall()

    # High-risk patients count
    data["high_risk_count"] = conn.execute("""
        SELECT COUNT(DISTINCT patient_id) FROM scans
        WHERE risk_level IN ('High','Very High')
    """).fetchone()[0]

    # Total patients, scans, doctors
    data["total_patients"] = conn.execute(
        "SELECT COUNT(*) FROM users WHERE role='patient'"
    ).fetchone()[0]
    data["total_scans"] = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    data["total_doctors"] = conn.execute(
        "SELECT COUNT(*) FROM users WHERE role='doctor'"
    ).fetchone()[0]
    data["pending_reviews"] = conn.execute(
        "SELECT COUNT(*) FROM scans WHERE status='pending'"
    ).fetchone()[0]

    # DR severity trend by month
    data["severity_trend"] = conn.execute("""
        SELECT strftime('%Y-%m', uploaded_at) as month,
               ROUND(AVG(ai_class_id),2) as avg_grade
        FROM scans WHERE ai_class_id IS NOT NULL
          AND uploaded_at >= datetime('now', '-12 months')
        GROUP BY month ORDER BY month
    """).fetchall()

    # Smoking vs DR severity
    data["smoking_dr"] = conn.execute("""
        SELECT pp.smoking, ROUND(AVG(s.ai_class_id),2) as avg_grade, COUNT(*) as cnt
        FROM scans s
        JOIN patient_profiles pp ON s.patient_id = pp.user_id
        WHERE pp.smoking IS NOT NULL AND s.ai_class_id IS NOT NULL
        GROUP BY pp.smoking
    """).fetchall()

    # Appointment stats
    data["appt_stats"] = conn.execute("""
        SELECT status, COUNT(*) as cnt FROM appointments GROUP BY status
    """).fetchall()

    # Doctor-reviewed scans count
    data["reviewed_scans"] = conn.execute(
        "SELECT COUNT(*) FROM scans WHERE status IN ('reviewed','approved','modified')"
    ).fetchone()[0]

    # Average risk score across all scans
    avg = conn.execute(
        "SELECT AVG(risk_score) FROM scans WHERE risk_score IS NOT NULL"
    ).fetchone()[0]
    data["avg_risk"] = round(avg, 1) if avg else 0

    # Recent scans with patient name for activity table
    data["recent_scans"] = conn.execute("""
        SELECT s.id, u.name as patient_name, s.uploaded_at,
               s.ai_class_id, s.ai_class_name, s.risk_score,
               s.risk_level, s.status
        FROM scans s
        JOIN users u ON s.patient_id = u.id
        ORDER BY s.uploaded_at DESC LIMIT 20
    """).fetchall()

    conn.close()
    return data
