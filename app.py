"""
Diabetic Retinopathy Clinical Decision Support System
Flask Web Application — Full Three-Module Build
"""

import os
import uuid
import json
import functools
from datetime import datetime

from flask import (Flask, render_template, request, redirect, url_for,
                   session, jsonify, flash)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False
    print("[WARN] TensorFlow not installed. AI prediction disabled.")

try:
    from gradcam import predict_and_explain
    GRADCAM_AVAILABLE = True
except ImportError:
    predict_and_explain = None
    GRADCAM_AVAILABLE = False
    print("[WARN] Grad-CAM module not available.")

import database as db

app = Flask(__name__)
app.secret_key = "dr-system-secret-key-2024-retina"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXT   = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MODEL_PATH    = "model/dr_model.keras"
CLASS_NAMES   = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static/results", exist_ok=True)

db.init_db()

model = None
def load_model():
    global model
    if not TF_AVAILABLE:
        print("[WARN] TensorFlow not available — AI prediction disabled.")
        return
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[INFO] Model loaded successfully.")
    else:
        print("[WARN] Model not found.")
try:
    load_model()
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

# ─── AUTH DECORATORS ─────────────────────────────────────────────────────────

def login_required(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper

def doctor_required(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        if session.get("role") != "doctor":
            flash("Access restricted to medical professionals.", "danger")
            return redirect(url_for("patient_dashboard"))
        return f(*args, **kwargs)
    return wrapper

def allowed_file(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def current_user():
    if "user_id" not in session:
        return None
    return db.get_user_by_id(session["user_id"])

@app.context_processor
def inject_globals():
    try:
        user = current_user()
        notif_count = db.get_unread_count(user["id"]) if user else 0
    except Exception:
        user = None
        notif_count = 0
    return dict(current_user=user, notif_count=notif_count, model_loaded=(model is not None))

# ─── AUTH ROUTES ─────────────────────────────────────────────────────────────

@app.context_processor
def inject_unread_count():
    count = 0
    if session.get("user_id"):
        count = db.get_unread_count(session["user_id"])
    return dict(unread_count=count)

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("doctor_dashboard") if session.get("role") == "doctor"
                        else url_for("patient_dashboard"))
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        user = db.get_user_by_email(email)
        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["role"]    = user["role"]
            session["name"]    = user["name"]
            return redirect(url_for("doctor_dashboard") if user["role"] == "doctor"
                            else url_for("patient_dashboard"))
        error = "Invalid email or password."
    return render_template("login.html", error=error)

@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        role     = request.form.get("role", "patient")
        specialty = request.form.get("specialty", "")
        if not name or not email or not password:
            error = "All fields are required."
        elif db.get_user_by_email(email):
            error = "An account with this email already exists."
        else:
            conn = db.get_db()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO users (name, email, password, role, specialty)
                VALUES (?,?,?,?,?)
            """, (name, email, generate_password_hash(password), role, specialty))
            uid = cur.lastrowid
            if role == "patient":
                cur.execute("INSERT INTO patient_profiles (user_id) VALUES (?)", (uid,))
            conn.commit(); conn.close()
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
    return render_template("register.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ─── PATIENT MODULE ───────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def patient_dashboard():
    if session.get("role") == "doctor":
        return redirect(url_for("doctor_dashboard"))
    uid     = session["user_id"]
    profile = db.get_patient_profile(uid)
    scans   = db.get_patient_scans(uid)
    progression = [
        {"date": s["uploaded_at"][:10], "grade": s["ai_class_id"],
         "risk": s["risk_score"], "id": s["id"]}
        for s in reversed(list(scans)) if s["ai_class_id"] is not None
    ]
    notifs = db.get_notifications(uid, limit=5, only_unread=True)
    unread_count = db.get_unread_count(uid)
    return render_template("patient_dashboard.html",
                           profile=profile, scans=scans,
                           progression=json.dumps(progression),
                           notifications=notifs, unread_count=unread_count)

@app.route("/profile", methods=["GET", "POST"])
@login_required
def patient_profile():
    if session.get("role") == "doctor":
        return redirect(url_for("index"))
    uid = session["user_id"]
    conn = db.get_db()
    if request.method == "POST":
        f = request.form
        conn.execute("""
            INSERT INTO patient_profiles
                (user_id, date_of_birth, gender, diabetes_type, diabetes_duration,
                 hba1c, systolic_bp, diastolic_bp, cholesterol, bmi,
                 smoking, insulin_use, notes, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))
            ON CONFLICT(user_id) DO UPDATE SET
                date_of_birth=excluded.date_of_birth, gender=excluded.gender,
                diabetes_type=excluded.diabetes_type,
                diabetes_duration=excluded.diabetes_duration,
                hba1c=excluded.hba1c, systolic_bp=excluded.systolic_bp,
                diastolic_bp=excluded.diastolic_bp, cholesterol=excluded.cholesterol,
                bmi=excluded.bmi, smoking=excluded.smoking,
                insulin_use=excluded.insulin_use, notes=excluded.notes,
                updated_at=datetime('now')
        """, (
            uid, f.get("date_of_birth"), f.get("gender"),
            f.get("diabetes_type"), f.get("diabetes_duration") or None,
            f.get("hba1c") or None, f.get("systolic_bp") or None,
            f.get("diastolic_bp") or None, f.get("cholesterol") or None,
            f.get("bmi") or None, f.get("smoking"),
            1 if f.get("insulin_use") else 0, f.get("notes"),
        ))
        conn.commit(); conn.close()
        flash("Medical profile updated successfully.", "success")
        return redirect(url_for("patient_dashboard"))
    profile = conn.execute("SELECT * FROM patient_profiles WHERE user_id=?", (uid,)).fetchone()
    conn.close()
    return render_template("patient_profile.html", profile=profile)

@app.route("/scan/new", methods=["GET", "POST"])
@login_required
def new_scan():
    if session.get("role") == "doctor":
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        if model is None:
            error = "AI model not loaded. Please train the model first."
        elif "file" not in request.files or request.files["file"].filename == "":
            error = "Please upload a retinal fundus image."
        else:
            file = request.files["file"]
            if not allowed_file(file.filename):
                error = "Unsupported file type. Use PNG, JPG, JPEG, BMP, or TIFF."
            else:
                ext      = file.filename.rsplit('.', 1)[1].lower()
                filename = f"{uuid.uuid4().hex}.{ext}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                try:
                    ai_result = predict_and_explain(model, filepath)
                    ai_result["image_url"] = f"/static/uploads/{filename}"
                    uid     = session["user_id"]
                    profile = db.get_patient_profile(uid)
                    prof_dict = dict(profile) if profile else {}
                    risk = db.compute_risk(prof_dict, ai_result["class_id"])
                    scan_id = db.save_scan(uid, filepath, f"/static/uploads/{filename}", ai_result, risk)
                    return redirect(url_for("scan_report", scan_id=scan_id))
                except Exception as e:
                    error = f"Analysis failed: {str(e)}"
    profile = db.get_patient_profile(session["user_id"])
    return render_template("new_scan.html", error=error, has_profile=profile is not None)

@app.route("/scan/<int:scan_id>")
@login_required
def scan_report(scan_id):
    scan = db.get_scan_by_id(scan_id)
    if not scan:
        flash("Scan not found.", "danger"); return redirect(url_for("patient_dashboard"))
    uid = session["user_id"]
    if session.get("role") == "patient" and scan["patient_id"] != uid:
        flash("Access denied.", "danger"); return redirect(url_for("patient_dashboard"))
    probs = json.loads(scan["ai_probs"]) if scan["ai_probs"] else {}
    risk_factors = json.loads(scan["risk_factors"]) if scan["risk_factors"] else []
    patient_scans = db.get_patient_scans(scan["patient_id"])
    progression = [
        {"date": s["uploaded_at"][:10], "grade": s["ai_class_id"], "id": s["id"]}
        for s in reversed(list(patient_scans)) if s["ai_class_id"] is not None
    ]
    return render_template("scan_report.html", scan=scan, probs=probs,
                           risk_factors=risk_factors, progression=json.dumps(progression))

# ─── DOCTOR MODULE ────────────────────────────────────────────────────────────

@app.route("/doctor")
@doctor_required
def doctor_dashboard():
    pending  = db.get_all_pending_scans()
    reviewed = db.get_all_reviewed_scans()
    notifs   = db.get_notifications(session["user_id"], limit=5, only_unread=True)
    unread_count = db.get_unread_count(session["user_id"])
    return render_template("doctor_dashboard.html",
                           pending=pending, reviewed=reviewed, 
                           notifications=notifs, unread_count=unread_count)

@app.route("/doctor/scan/<int:scan_id>", methods=["GET", "POST"])
@doctor_required
def doctor_review(scan_id):
    scan = db.get_scan_by_id(scan_id)
    if not scan:
        flash("Scan not found.", "danger"); return redirect(url_for("doctor_dashboard"))
    if request.method == "POST":
        f = request.form
        doc_class = int(f.get("doctor_class_id", scan["ai_class_id"] or 0))
        db.save_doctor_review(scan_id, session["user_id"], {
            "status":           f.get("status", "reviewed"),
            "doctor_class_id":  doc_class,
            "doctor_class_name": CLASS_NAMES[doc_class],
            "doctor_notes":     f.get("doctor_notes", ""),
            "treatment":        f.get("treatment", ""),
            "followup_weeks":   f.get("followup_weeks") or None,
        })
        flash(f"Scan #{scan_id} review submitted successfully.", "success")
        return redirect(url_for("doctor_dashboard"))
    probs = json.loads(scan["ai_probs"]) if scan["ai_probs"] else {}
    risk_factors = json.loads(scan["risk_factors"]) if scan["risk_factors"] else []
    patient_scans = db.get_patient_scans(scan["patient_id"])
    progression = [
        {"date": s["uploaded_at"][:10], "grade": s["ai_class_id"], "id": s["id"]}
        for s in reversed(list(patient_scans)) if s["ai_class_id"] is not None
    ]
    return render_template("doctor_review.html", scan=scan, probs=probs,
                           risk_factors=risk_factors, class_names=CLASS_NAMES,
                           progression=json.dumps(progression))

@app.route("/doctor/patient/<int:patient_id>")
@doctor_required
def doctor_patient_history(patient_id):
    patient = db.get_user_by_id(patient_id)
    if not patient:
        flash("Patient not found.", "danger"); return redirect(url_for("doctor_dashboard"))
    profile = db.get_patient_profile(patient_id)
    scans   = db.get_patient_scans(patient_id)
    progression = [
        {"date": s["uploaded_at"][:10], "grade": s["ai_class_id"],
         "risk": s["risk_score"], "id": s["id"]}
        for s in reversed(list(scans)) if s["ai_class_id"] is not None
    ]
    return render_template("patient_history.html",
                           patient=patient, profile=profile,
                           scans=scans, progression=json.dumps(progression))

# ─── MISC ─────────────────────────────────────────────────────────────────────

@app.route("/graphs")
def graphs():
    graph_files = {
        "Training Curves":   "results/training_curves.png",
        "Confusion Matrix":  "results/confusion_matrix.png",
        "Per-Class Metrics": "results/per_class_metrics.png",
        "Class Distribution":"results/class_distribution.png",
    }
    available = {k: v for k, v in graph_files.items() if os.path.exists(f"static/{v}")}
    return render_template("graphs.html", graphs=available)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/notifications")
@login_required
def notifications():
    uid = session["user_id"]
    notifs = db.get_notifications(uid, limit=50)
    db.mark_notifications_read(uid)
    return render_template("notifications.html", notifications=notifs)

@app.route("/notifications/clear", methods=["POST"])
@login_required
def clear_notifications():
    uid = session["user_id"]
    conn = db.get_db()
    conn.execute("DELETE FROM notifications WHERE user_id=?", (uid,))
    conn.commit()
    conn.close()
    flash("Notifications cleared.", "info")
    return redirect(url_for("notifications"))

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file and allowed_file(file.filename):
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        result = predict_and_explain(model, filepath)
        return jsonify({k: v for k, v in result.items() if k not in ["gradcam_b64","prob_chart_b64"]})
    return jsonify({"error": "Invalid file type"}), 400

# ═══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL MODULE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

import os as _os
_os.makedirs("static/documents", exist_ok=True)
_os.makedirs("static/reports",   exist_ok=True)

db.init_appointments()
db.init_documents()
db.init_schedules()

from report_generator import generate_diagnostic_report

ALLOWED_DOC_EXT = {'pdf', 'png', 'jpg', 'jpeg', 'docx', 'txt', 'bmp', 'tiff'}

def allowed_doc(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in ALLOWED_DOC_EXT


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — APPOINTMENT & FOLLOW-UP MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/appointments")
@login_required
def appointments():
    uid = session["user_id"]
    if session["role"] == "doctor":
        appts = db.get_doctor_appointments(uid)
        stats = db.get_appointment_stats(uid)
        doctors = []
    else:
        appts = db.get_patient_appointments(uid)
        stats = db.get_appointment_stats()
        doctors = db.get_all_doctors()
    return render_template("appointments.html",
                           appointments=appts, stats=stats, doctors=doctors)


@app.route("/appointments/book", methods=["GET", "POST"])
@login_required
def book_appointment():
    if session["role"] == "doctor":
        flash("Doctors cannot book appointments through this form.", "warning")
        return redirect(url_for("appointments"))

    uid = session["user_id"]
    error = None
    if request.method == "POST":
        f = request.form
        if not f.get("appointment_date") or not f.get("appointment_time"):
            error = "Date and time are required."
        else:
            appt_id = db.book_appointment(uid, {
                "doctor_id":        f.get("doctor_id") or None,
                "scan_id":          f.get("scan_id")   or None,
                "appointment_date": f["appointment_date"],
                "appointment_time": f["appointment_time"],
                "type":             f.get("type", "Follow-up"),
                "patient_notes":    f.get("patient_notes", ""),
            })
            flash(f"Appointment #{appt_id} booked successfully!", "success")
            return redirect(url_for("appointments"))

    doctors = db.get_all_doctors()
    scans   = db.get_patient_scans(uid)
    from datetime import date as _date
    return render_template("book_appointment.html",
                           doctors=doctors, scans=scans, error=error,
                           today=_date.today().isoformat())


@app.route("/appointments/<int:appt_id>", methods=["GET", "POST"])
@login_required
def appointment_detail(appt_id):
    appt = db.get_appointment_by_id(appt_id)
    if not appt:
        flash("Appointment not found.", "danger")
        return redirect(url_for("appointments"))

    uid = session["user_id"]
    if (session["role"] == "patient" and appt["patient_id"] != uid) or \
       (session["role"] == "doctor"  and appt["doctor_id"]  != uid):
        flash("Access denied.", "danger")
        return redirect(url_for("appointments"))

    if request.method == "POST":
        f = request.form
        db.update_appointment(appt_id, {
            "status":       f.get("status", appt["status"]),
            "doctor_notes": f.get("doctor_notes", ""),
        })
        flash("Appointment updated.", "success")
        return redirect(url_for("appointments"))

    return render_template("appointment_detail.html", appt=appt)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — MEDICAL REPORT & DOCUMENT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/documents")
@login_required
def documents():
    uid = session["user_id"]
    if session["role"] == "doctor":
        # Doctors see all documents (via patient history)
        flash("Select a patient to view their documents.", "info")
        return redirect(url_for("doctor_dashboard"))
    docs    = db.get_patient_documents(uid)
    reports = db.get_patient_reports(uid)
    return render_template("documents.html", documents=docs, reports=reports)


@app.route("/documents/upload", methods=["GET", "POST"])
@login_required
def upload_document():
    uid   = session["user_id"]
    error = None

    # Doctors upload on behalf of a patient
    target_patient_id = uid
    if session["role"] == "doctor":
        target_patient_id = request.args.get("patient_id") or \
                            request.form.get("patient_id")
        if not target_patient_id:
            flash("Specify a patient_id to upload documents.", "warning")
            return redirect(url_for("doctor_dashboard"))
        target_patient_id = int(target_patient_id)

    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            error = "Please select a file."
        else:
            file = request.files["file"]
            if not allowed_doc(file.filename):
                error = "Unsupported file type."
            else:
                ext       = file.filename.rsplit(".", 1)[1].lower()
                safe_name = secure_filename(file.filename)
                filename  = f"{uuid.uuid4().hex}_{safe_name}"
                filepath  = os.path.join("static/documents", filename)
                file.save(filepath)
                size = os.path.getsize(filepath)

                db.save_document(
                    patient_id   = target_patient_id,
                    uploaded_by  = uid,
                    scan_id      = request.form.get("scan_id") or None,
                    file_path    = filepath,
                    file_url     = f"/static/documents/{filename}",
                    filename     = safe_name,
                    doc_type     = request.form.get("doc_type", "Other"),
                    description  = request.form.get("description", ""),
                    file_size    = size,
                )
                flash("Document uploaded successfully.", "success")
                if session["role"] == "doctor":
                    return redirect(url_for("doctor_patient_history",
                                            patient_id=target_patient_id))
                return redirect(url_for("documents"))

    scans = db.get_patient_scans(target_patient_id)
    return render_template("upload_document.html",
                           error=error, scans=scans,
                           target_patient_id=target_patient_id)


@app.route("/documents/<int:doc_id>/delete", methods=["POST"])
@login_required
def delete_document(doc_id):
    doc = db.get_document_by_id(doc_id)
    if not doc:
        flash("Document not found.", "danger")
    else:
        uid = session["user_id"]
        if session["role"] == "patient" and doc["patient_id"] != uid:
            flash("Access denied.", "danger")
        else:
            fp = db.delete_document(doc_id)
            if fp and os.path.exists(fp):
                os.remove(fp)
            flash("Document deleted.", "success")
    return redirect(url_for("documents"))


@app.route("/documents/generate/<int:scan_id>")
@login_required
def generate_report(scan_id):
    import json as _json
    scan = db.get_scan_by_id(scan_id)
    if not scan:
        flash("Scan not found.", "danger")
        return redirect(url_for("documents"))

    uid = session["user_id"]
    if session["role"] == "patient" and scan["patient_id"] != uid:
        flash("Access denied.", "danger")
        return redirect(url_for("documents"))

    patient = db.get_user_by_id(scan["patient_id"])
    profile = db.get_patient_profile(scan["patient_id"])
    probs   = _json.loads(scan["ai_probs"]) if scan["ai_probs"] else {}
    risk_f  = _json.loads(scan["risk_factors"]) if scan["risk_factors"] else []
    prof_d  = dict(profile) if profile else {}
    scan_d  = dict(scan)
    scan_d["doctor_name"] = scan_d.get("doctor_name")

    fp, fu = generate_diagnostic_report(scan_d, dict(patient), prof_d, probs, risk_f)
    if fp:
        doctor_id = scan["doctor_id"] or (uid if session["role"] == "doctor" else None)
        db.save_generated_report(scan["patient_id"], scan_id, doctor_id,
                                  "Diagnostic Report", fp, fu)
        flash("PDF report generated successfully.", "success")
        from flask import send_file
        return send_file(fp, as_attachment=True,
                         download_name=f"RetinaAI_Report_Scan{scan_id}.pdf")
    else:
        flash("PDF generation failed. reportlab may not be installed.", "danger")
    return redirect(url_for("scan_report", scan_id=scan_id))


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 7 — DOCTOR SCHEDULE & WORKLOAD MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/doctor/schedule", methods=["GET", "POST"])
@doctor_required
def doctor_schedule():
    uid = session["user_id"]
    DAY_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    if request.method == "POST":
        schedules = []
        for i in range(7):
            is_active = 1 if request.form.get(f"active_{i}") else 0
            st = request.form.get(f"start_{i}", "09:00")
            et = request.form.get(f"end_{i}",   "17:00")
            mp = request.form.get(f"max_{i}", "10")
            schedules.append({
                "day_of_week":   i,
                "start_time":    st,
                "end_time":      et,
                "max_patients":  int(mp) if mp.isdigit() else 10,
                "is_active":     is_active,
            })
        db.save_doctor_availability(uid, schedules)
        flash("Schedule updated successfully.", "success")
        return redirect(url_for("doctor_schedule"))

    availability = {row["day_of_week"]: dict(row)
                    for row in db.get_doctor_availability(uid)}
    workload = db.get_doctor_workload_stats(uid)
    all_doctors_load = db.get_all_doctors_workload()

    # Today's appointments
    from datetime import date as _date
    today_str = _date.today().isoformat()
    today_appts = db.get_doctor_appointments(uid, date_filter=today_str)

    import json as _json
    try:
        daily_reviews = _json.dumps([
            {"day": r["day"] if isinstance(r, dict) else r[0],
             "cnt": r["cnt"] if isinstance(r, dict) else r[1]}
            for r in (workload.get("daily_reviews") or [])
        ])
    except Exception:
        daily_reviews = "[]"
    try:
        grade_dist = _json.dumps([
            {"grade": r["grade"] if isinstance(r, dict) else r[0],
             "cnt":   r["cnt"]   if isinstance(r, dict) else r[1]}
            for r in (workload.get("grade_dist") or [])
        ])
    except Exception:
        grade_dist = "[]"

    return render_template("doctor_schedule.html",
                           availability=availability,
                           day_names=DAY_NAMES,
                           workload=workload,
                           daily_reviews=daily_reviews,
                           grade_dist=grade_dist,
                           all_doctors_load=all_doctors_load,
                           today_appts=today_appts,
                           today=today_str)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 10 — POPULATION HEALTH ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/analytics")
@login_required
def analytics():
    import json as _json

    def to_json(rows, key1, key2):
        try:
            return _json.dumps([{key1: r[0], key2: r[1]} for r in rows])
        except Exception:
            return "[]"

    def safe_list(rows, builder):
        try:
            return _json.dumps([builder(r) for r in (rows or [])])
        except Exception:
            return "[]"

    try:
        data = db.get_population_analytics()
    except Exception as e:
        print(f"[ERROR] analytics DB error: {e}")
        data = {}

    # Ensure ALL expected keys exist with safe defaults
    data.setdefault("grade_dist",     [])
    data.setdefault("risk_dist",      [])
    data.setdefault("monthly_scans",  [])
    data.setdefault("gender_dist",    [])
    data.setdefault("diabetes_dist",  [])
    data.setdefault("severity_trend", [])
    data.setdefault("smoking_dr",     [])
    data.setdefault("hba1c_by_grade", [])
    data.setdefault("appt_stats",     [])
    data.setdefault("total_scans",    0)
    data.setdefault("total_patients", 0)
    data.setdefault("total_doctors",  0)
    data.setdefault("pending_reviews",0)
    data.setdefault("high_risk_count",0)
    data.setdefault("reviewed_scans", 0)
    data.setdefault("avg_risk",       0)
    data.setdefault("recent_scans",   [])

    try:
        chart_data = {
            "grade_dist":     to_json(data["grade_dist"],    "label", "cnt"),
            "risk_dist":      to_json(data["risk_dist"],     "label", "cnt"),
            "monthly_scans":  to_json(data["monthly_scans"], "month", "cnt"),
            "gender_dist":    to_json(data["gender_dist"],   "label", "cnt"),
            "diabetes_dist":  to_json(data["diabetes_dist"], "label", "cnt"),
            "severity_trend": to_json(data["severity_trend"],"month", "avg"),
            "smoking_dr":     safe_list(data["smoking_dr"],
                                lambda r: {"label": r[0], "avg": r[1], "cnt": r[2]}),
            "hba1c_by_grade": safe_list(data["hba1c_by_grade"],
                                lambda r: {"grade": r[1], "hba1c": r[2]}),
            "appt_stats":     to_json(data["appt_stats"],   "label", "cnt"),
        }
        chart_json = _json.dumps(chart_data)
    except Exception as e:
        print(f"[ERROR] analytics chart_data build error: {e}")
        chart_json = "{}"

    return render_template("analytics.html",
                           data=data, chart_data=chart_json)

# ─── ERROR HANDLERS ──────────────────────────────────────────────────────────
@app.errorhandler(404)
def page_not_found(e):
    try:
        return render_template("404.html", code=404,
            message="The page you're looking for doesn't exist."), 404
    except Exception:
        return "<h1>404 Not Found</h1><p><a href='/'>Go Home</a></p>", 404

@app.errorhandler(403)
def forbidden(e):
    try:
        return render_template("404.html", code=403,
            message="You don't have permission to access this page."), 403
    except Exception:
        return "<h1>403 Forbidden</h1><p><a href='/'>Go Home</a></p>", 403

@app.errorhandler(500)
def server_error(e):
    import traceback
    print(f"[ERROR 500] {traceback.format_exc()}")
    try:
        return render_template("404.html", code=500,
            message="An internal error occurred. Please try again."), 500
    except Exception:
        return "<h1>500 Internal Server Error</h1><p><a href='/'>Go Home</a></p>", 500

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    debug = "--debug" in sys.argv or True   # default debug on for dev
    port  = int(sys.argv[sys.argv.index("--port")+1]) if "--port" in sys.argv else 5000
    print(f"[INFO] Starting RetinaAI on http://0.0.0.0:{port}  debug={debug}")
    print(f"[INFO] Open http://localhost:{port} in your browser")
    app.run(debug=debug, host="0.0.0.0", port=port, use_reloader=debug)
