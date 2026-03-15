"""
app_final_v2_27zones.py — Water Management Analytics | Flask Backend v3
Role-based auth: admin (upload Excel, manage data) | student (view + download)
27-zone dataset · ~211 MLD · PMC ESR 2024-25 (IIT Bombay)
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, create_access_token,
    jwt_required, get_jwt_identity
)
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3, pickle, os, logging, json, warnings, io
import numpy as np
import pandas as pd
from datetime import timedelta, datetime

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["JWT_SECRET_KEY"]           = os.environ.get("JWT_SECRET_KEY", "water-mgmt-analytics-2026-secret-key")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=30)
app.config["MAX_CONTENT_LENGTH"]       = 50 * 1024 * 1024   # 50 MB upload limit

CORS(app, origins="*")
jwt = JWTManager(app)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── JSON encoder for numpy types ────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.ndarray):   return obj.tolist()
        if isinstance(obj, pd.Timestamp): return str(obj.date())
        return super().default(obj)

app.json_encoder = NumpyEncoder

# ── PATHS ────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE, "users.db")
UPLOADS_DIR = os.path.join(BASE, "uploaded_excel")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ── SQLITE — users + uploaded_files tables ───────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    # Users table now has role column
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT    NOT NULL,
            email    TEXT    NOT NULL UNIQUE,
            password TEXT    NOT NULL,
            role     TEXT    NOT NULL DEFAULT 'student',
            created  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Track uploaded Excel files
    conn.execute("""
        CREATE TABLE IF NOT EXISTS uploaded_files (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            filename     TEXT    NOT NULL,
            original_name TEXT   NOT NULL,
            uploaded_by  TEXT    NOT NULL,
            rows         INTEGER DEFAULT 0,
            file_size    INTEGER DEFAULT 0,
            description  TEXT    DEFAULT '',
            uploaded_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_user_by_email(email):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    u = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    return dict(u) if u else None

def create_user(name, email, password, role):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO users (name,email,password,role) VALUES (?,?,?,?)",
            (name, email, generate_password_hash(password), role)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id,name,email,role,created FROM users ORDER BY created DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def save_file_record(filename, original_name, uploaded_by, rows, file_size, description):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO uploaded_files (filename,original_name,uploaded_by,rows,file_size,description) VALUES (?,?,?,?,?,?)",
        (filename, original_name, uploaded_by, rows, file_size, description)
    )
    conn.commit()
    conn.close()

def get_all_file_records():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM uploaded_files ORDER BY uploaded_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_file_record(file_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    r = conn.execute("SELECT * FROM uploaded_files WHERE id=?", (file_id,)).fetchone()
    conn.close()
    return dict(r) if r else None

def delete_file_record(file_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM uploaded_files WHERE id=?", (file_id,))
    conn.commit()
    conn.close()

# ── LOAD ML MODELS ───────────────────────────────────────
try:
    with open(os.path.join(BASE, "city_models.pkl"), "rb") as f:
        CITY_BUNDLE = pickle.load(f)
    with open(os.path.join(BASE, "ward_models.pkl"), "rb") as f:
        WARD_BUNDLE = {int(k): v for k, v in pickle.load(f).items()}
    log.info("✅ ML Models loaded")
except Exception as e:
    log.error(f"❌ Model load failed: {e}")
    raise

WARDS       = sorted(WARD_BUNDLE.keys())
LEAKAGE_PCT = 0.0952   # Official PMC figure

# ── PREDICTION HELPERS ───────────────────────────────────
def _city_forecast(days):
    results      = []
    supply_hist  = list(CITY_BUNDLE['Water_Supplied_MLD']['last_values'])
    consume_hist = list(CITY_BUNDLE['Water_Consumed_MLD']['last_values'])
    supply_model  = CITY_BUNDLE['Water_Supplied_MLD']['model']
    consume_model = CITY_BUNDLE['Water_Consumed_MLD']['model']

    for d in range(days):
        future = datetime.now() + timedelta(days=d + 1)
        sv = np.array(supply_hist[-14:], dtype=float)
        s_feats = np.array([
            *sv[::-1][:14],
            float(sv[-1]-sv[-2]) if len(sv)>1 else 0,
            float(sv[-1]-sv[-8]) if len(sv)>7 else 0,
            float(sv[-7:].mean()), float(sv.mean()),
            float(sv[-7:].std()) if sv[-7:].std()>0 else 0.001,
            float(future.timetuple().tm_yday),
            float(future.weekday()), float(future.month)
        ]).reshape(1,-1)
        pred_supply = max(float(supply_model.predict(s_feats)[0]), 50.0)

        cv = np.array(consume_hist[-14:], dtype=float)
        c_feats = np.array([
            *cv[::-1][:14],
            float(cv[-1]-cv[-2]) if len(cv)>1 else 0,
            float(cv[-1]-cv[-8]) if len(cv)>7 else 0,
            float(cv[-7:].mean()), float(cv.mean()),
            float(cv[-7:].std()) if cv[-7:].std()>0 else 0.001,
            float(future.timetuple().tm_yday),
            float(future.weekday()), float(future.month)
        ]).reshape(1,-1)
        pred_consume = max(float(consume_model.predict(c_feats)[0]), 40.0)
        pred_consume = min(pred_consume, pred_supply * 0.95)
        pred_leakage = pred_supply - pred_consume
        leak_pct     = pred_leakage / pred_supply * 100

        results.append({
            "Date":                       future.strftime("%Y-%m-%d"),
            "Day":                        d + 1,
            "Predicted_Supply_MLD":       round(pred_supply, 3),
            "Predicted_Consumption_MLD":  round(pred_consume, 3),
            "Predicted_Leakage_MLD":      round(pred_leakage, 3),
            "Leakage_Percentage":         round(leak_pct, 2),
        })
        supply_hist.append(pred_supply)
        consume_hist.append(pred_consume)
    return results

def _ward_forecast(ward_no, days):
    wb       = WARD_BUNDLE[ward_no]
    model    = wb['model']
    hist     = list(wb['last_values'])
    leak_pct = wb.get('leakage_pct', LEAKAGE_PCT)
    results  = []

    for d in range(days):
        future = datetime.now() + timedelta(days=d + 1)
        v = np.array(hist[-14:], dtype=float)
        feats = np.array([
            *v[::-1][:14],
            float(v[-1]-v[-2]) if len(v)>1 else 0,
            float(v[-1]-v[-8]) if len(v)>7 else 0,
            float(v[-7:].mean()), float(v.mean()),
            float(v[-7:].std()) if v[-7:].std()>0 else 0.001,
            float(future.timetuple().tm_yday),
            float(future.weekday()), float(future.month)
        ]).reshape(1,-1)
        pred_supply  = max(float(model.predict(feats)[0]), 0.1)
        pred_consume = round(pred_supply * (1 - leak_pct), 4)
        pred_leakage = round(pred_supply - pred_consume, 4)

        results.append({
            "Date":                      future.strftime("%Y-%m-%d"),
            "Day":                       d + 1,
            "Ward_No":                   ward_no,
            "Ward_Name":                 wb['ward_name'],
            "Predicted_Supply_MLD":      round(pred_supply, 4),
            "Predicted_Consumption_MLD": round(pred_consume, 4),
            "Predicted_Leakage_MLD":     round(pred_leakage, 4),
            "Leakage_Percentage":        round(leak_pct * 100, 2),
        })
        hist.append(pred_supply)
    return results

# Pre-compute 30-day caches on startup
log.info("Building forecast cache...")
_CACHE_CITY_30  = pd.DataFrame(_city_forecast(30))
_cache_ward_rows = []
for _wno in WARDS:
    _cache_ward_rows.extend(_ward_forecast(_wno, 30))
_CACHE_WARD_30 = pd.DataFrame(_cache_ward_rows)
log.info("✅ Cache ready")

# ── ROLE GUARD ───────────────────────────────────────────
def require_admin(fn):
    """Decorator: ensures the JWT user has role='admin'."""
    from functools import wraps
    @wraps(fn)
    @jwt_required()
    def wrapper(*args, **kwargs):
        email = get_jwt_identity()
        user  = get_user_by_email(email)
        if not user or user.get("role") != "admin":
            return jsonify({"error": "Admin access required"}), 403
        return fn(*args, **kwargs)
    return wrapper

# ════════════════════════════════════════════════════════
# AUTH ROUTES
# ════════════════════════════════════════════════════════

@app.route("/auth/register", methods=["POST"])
def register():
    d     = request.get_json()
    name  = (d.get("name") or "").strip()
    email = (d.get("email") or "").strip().lower()
    pwd   = (d.get("password") or "")
    role  = (d.get("role") or "student").strip().lower()

    if not all([name, email, pwd]):
        return jsonify({"error": "All fields are required"}), 400
    if len(pwd) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    if "@" not in email:
        return jsonify({"error": "Invalid email address"}), 400
    if role not in ("admin", "student"):
        return jsonify({"error": "Role must be 'admin' or 'student'"}), 400

    if not create_user(name, email, pwd, role):
        return jsonify({"error": "Email already registered"}), 409

    token = create_access_token(identity=email)
    return jsonify({
        "message": "Account created successfully!",
        "token":   token,
        "name":    name,
        "email":   email,
        "role":    role,
    }), 201


@app.route("/auth/login", methods=["POST"])
def login():
    d     = request.get_json()
    email = (d.get("email") or "").strip().lower()
    pwd   = (d.get("password") or "")
    user  = get_user_by_email(email)

    if not user or not check_password_hash(user["password"], pwd):
        return jsonify({"error": "Invalid email or password"}), 401

    token = create_access_token(identity=email)
    return jsonify({
        "token": token,
        "name":  user["name"],
        "email": email,
        "role":  user["role"],
    }), 200


@app.route("/auth/me", methods=["GET"])
@jwt_required()
def me():
    email = get_jwt_identity()
    user  = get_user_by_email(email)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({
        "email": email,
        "name":  user["name"],
        "role":  user["role"],
    }), 200


# ── Admin: list all users ────────────────────────────────
@app.route("/admin/users", methods=["GET"])
@require_admin
def admin_list_users():
    return jsonify(get_all_users())


# ════════════════════════════════════════════════════════
# EXCEL UPLOAD (Admin only)
# ════════════════════════════════════════════════════════

ALLOWED_EXTENSIONS = {".xlsx", ".xls", ".csv"}

@app.route("/admin/upload_excel", methods=["POST"])
@require_admin
def upload_excel():
    email = get_jwt_identity()

    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    f    = request.files["file"]
    desc = request.form.get("description", "")

    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Only .xlsx / .xls / .csv files are allowed"}), 400

    # Save with timestamp-based unique name
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{ts}_{f.filename.replace(' ', '_')}"
    save_path = os.path.join(UPLOADS_DIR, safe_name)
    f.save(save_path)

    # Read file to get row count + validate
    try:
        if ext == ".csv":
            df = pd.read_csv(save_path)
        else:
            df = pd.read_excel(save_path)
        rows      = len(df)
        file_size = os.path.getsize(save_path)
        cols      = list(df.columns)
    except Exception as e:
        os.remove(save_path)
        return jsonify({"error": f"Could not read file: {str(e)}"}), 400

    save_file_record(safe_name, f.filename, email, rows, file_size, desc)

    return jsonify({
        "message":    "File uploaded successfully",
        "filename":   safe_name,
        "rows":       rows,
        "columns":    cols,
        "file_size":  file_size,
        "uploaded_by": email,
    }), 201


@app.route("/admin/files", methods=["GET"])
@jwt_required()
def list_files():
    """Both admin and student can list files (students download, admin manage)."""
    records = get_all_file_records()
    return jsonify(records)


@app.route("/admin/files/<int:file_id>", methods=["DELETE"])
@require_admin
def delete_file(file_id):
    rec = get_file_record(file_id)
    if not rec:
        return jsonify({"error": "File not found"}), 404
    path = os.path.join(UPLOADS_DIR, rec["filename"])
    if os.path.exists(path):
        os.remove(path)
    delete_file_record(file_id)
    return jsonify({"message": "File deleted successfully"})


@app.route("/admin/files/<int:file_id>/preview", methods=["GET"])
@jwt_required()
def preview_file(file_id):
    """Return first 50 rows of uploaded file as JSON — for both roles."""
    rec = get_file_record(file_id)
    if not rec:
        return jsonify({"error": "File not found"}), 404
    path = os.path.join(UPLOADS_DIR, rec["filename"])
    if not os.path.exists(path):
        return jsonify({"error": "File missing from storage"}), 404
    try:
        ext = os.path.splitext(rec["filename"])[1].lower()
        df  = pd.read_csv(path) if ext == ".csv" else pd.read_excel(path)
        df  = df.head(50).fillna("")
        return jsonify({
            "columns": list(df.columns),
            "rows":    df.to_dict(orient="records"),
            "total_rows": rec["rows"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/files/<int:file_id>/download", methods=["GET"])
@jwt_required()
def download_file(file_id):
    """Download uploaded Excel file — available to both admin and student."""
    rec = get_file_record(file_id)
    if not rec:
        return jsonify({"error": "File not found"}), 404
    path = os.path.join(UPLOADS_DIR, rec["filename"])
    if not os.path.exists(path):
        return jsonify({"error": "File missing from storage"}), 404
    return send_file(
        path,
        as_attachment=True,
        download_name=rec["original_name"],
    )


# ── Download the BUILT-IN Excel datasets ────────────────
@app.route("/download/historical_excel", methods=["GET"])
@jwt_required()
def download_historical_excel():
    """Generate and download complete historical dataset as Excel."""
    try:
        hist = CITY_BUNDLE['city_history']
        df   = pd.DataFrame(hist)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime("%Y-%m-%d")

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="City_Historical_Data")
        buf.seek(0)

        return send_file(
            buf,
            as_attachment=True,
            download_name="WMA_Historical_City_Data.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download/ward_excel/<int:ward_no>", methods=["GET"])
@jwt_required()
def download_ward_excel(ward_no):
    """Download historical data for a specific zone as Excel."""
    if ward_no not in WARD_BUNDLE:
        return jsonify({"error": f"Zone {ward_no} not found"}), 404
    try:
        wb   = WARD_BUNDLE[ward_no]
        vals = wb['last_values']
        dates = pd.date_range(end='2026-02-28', periods=len(vals), freq='D')
        rows  = [{
            "Date":                dt.strftime("%Y-%m-%d"),
            "Ward_No":             ward_no,
            "Ward_Name":           wb['ward_name'],
            "Water_Supplied_MLD":  round(v, 4),
            "Water_Consumed_MLD":  round(v*(1-LEAKAGE_PCT), 4),
            "Leakage_MLD":         round(v*LEAKAGE_PCT, 4),
            "Leakage_Percentage":  round(LEAKAGE_PCT*100, 2),
        } for dt, v in zip(dates, vals)]

        df  = pd.DataFrame(rows)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=f"Zone_{ward_no}_{wb['ward_name'][:15]}")
        buf.seek(0)

        return send_file(
            buf,
            as_attachment=True,
            download_name=f"WMA_Zone{ward_no}_{wb['ward_name'].replace(' ','_')}.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════
# ANALYTICS ROUTES  (JWT required for all)
# ════════════════════════════════════════════════════════

@app.route("/")
def index():
    return jsonify({
        "status":       "Water Management Analytics API v3",
        "zones":        len(WARDS),
        "target_mld":   211,
        "roles":        ["admin", "student"],
    })


@app.route("/summary")
@jwt_required()
def summary():
    try:
        hist     = CITY_BUNDLE['city_history']
        city_df  = pd.DataFrame(hist)
        avg_sup  = float(city_df['Water_Supplied_MLD'].mean())
        avg_leak = float(city_df['Leakage_MLD'].mean())
        worst_wno = max(WARD_BUNDLE.items(), key=lambda x: x[1].get('rmse', 0))[0]
        return jsonify({
            "Last_Data_Date":           "2026-02-28",
            "Avg_Daily_Supply_MLD":     round(avg_sup, 2),
            "Avg_Daily_Leakage_MLD":    round(avg_leak, 3),
            "Avg_Leakage_Percentage":   round(LEAKAGE_PCT * 100, 2),
            "Highest_Leakage_Ward_No":  int(worst_wno),
            "Highest_Leakage_Ward":     WARD_BUNDLE[worst_wno]['ward_name'],
            "Total_Zones":              len(WARDS),
            "Data_Coverage_MLD":        "~211 MLD (PMC ESR 2024-25)",
            "Model_Accuracy": {
                "City_Supply_R2":      round(float(CITY_BUNDLE['Water_Supplied_MLD']['r2']), 4),
                "City_Consumption_R2": round(float(CITY_BUNDLE['Water_Consumed_MLD']['r2']), 4),
                "City_Supply_RMSE":    round(float(CITY_BUNDLE['Water_Supplied_MLD']['rmse']), 4),
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/city/<int:days>")
@jwt_required()
def predict_city(days):
    try:
        if days < 1 or days > 365:
            return jsonify({"error": "days must be between 1 and 365"}), 400
        if days == 30:
            return jsonify(_CACHE_CITY_30.to_dict(orient="records"))
        return jsonify(_city_forecast(days))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/ward/<int:ward_no>/<int:days>")
@jwt_required()
def predict_ward(ward_no, days):
    try:
        if ward_no not in WARD_BUNDLE:
            return jsonify({"error": f"Zone {ward_no} not found"}), 404
        if days < 1 or days > 365:
            return jsonify({"error": "days must be between 1 and 365"}), 400
        if days == 30:
            result = _CACHE_WARD_30[_CACHE_WARD_30["Ward_No"] == ward_no].to_dict(orient="records")
            return jsonify(result)
        return jsonify(_ward_forecast(ward_no, days))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/alerts/scan")
@jwt_required()
def alerts_scan():
    try:
        threshold = request.args.get("threshold", 9.52, type=float)
        days      = max(1, min(request.args.get("days", 30, type=int), 365))
        results   = []
        for wno in WARDS:
            data      = _ward_forecast(wno, days)
            leak_pcts = [r["Leakage_Percentage"] for r in data]
            avg_pct   = sum(leak_pcts) / len(leak_pcts)
            max_pct   = max(leak_pcts)
            avg_mld   = sum(r["Predicted_Leakage_MLD"] for r in data) / len(data)
            days_exc  = sum(1 for p in leak_pcts if p >= threshold)
            if   avg_pct >= 20: level = "CRITICAL"
            elif avg_pct >= 15: level = "HIGH"
            elif avg_pct >= threshold: level = "MODERATE"
            else:                level = "NORMAL"
            peak_idx = leak_pcts.index(max_pct)
            results.append({
                "Ward_No":         wno,
                "Ward_Name":       WARD_BUNDLE[wno]["ward_name"],
                "Level":           level,
                "Avg_Leakage_Pct": round(avg_pct, 2),
                "Max_Leakage_Pct": round(max_pct, 2),
                "Avg_Leakage_MLD": round(avg_mld, 3),
                "Days_Exceeding":  days_exc,
                "Peak_Date":       data[peak_idx]["Date"],
                "Start_Date":      data[0]["Date"],
                "End_Date":        data[-1]["Date"],
            })
        lo = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "NORMAL": 3}
        results.sort(key=lambda x: (lo[x["Level"]], -x["Avg_Leakage_Pct"]))
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/wards")
@jwt_required()
def wards():
    return jsonify([{"Ward_No": int(k), "Ward_Name": v["ward_name"]}
                    for k, v in sorted(WARD_BUNDLE.items())])


@app.route("/metrics")
@jwt_required()
def metrics():
    return jsonify([{
        "Ward_No":     int(wno),
        "Ward_Name":   wb["ward_name"],
        "Supply_R2":   round(float(wb["r2"]), 4),
        "Supply_RMSE": round(float(wb["rmse"]), 4),
    } for wno, wb in sorted(WARD_BUNDLE.items())])


@app.route("/historical/city")
@jwt_required()
def historical_city():
    try:
        hist = CITY_BUNDLE['city_history']
        df   = pd.DataFrame(hist)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime("%Y-%m-%d")
        df['Leakage_Percentage'] = (df['Leakage_MLD'] / df['Water_Supplied_MLD'] * 100).round(2)
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/historical/ward/<int:ward_no>")
@jwt_required()
def historical_ward(ward_no):
    if ward_no not in WARD_BUNDLE:
        return jsonify({"error": f"Zone {ward_no} not found"}), 404
    wb    = WARD_BUNDLE[ward_no]
    vals  = wb['last_values']
    dates = pd.date_range(end='2026-02-28', periods=len(vals), freq='D')
    rows  = [{
        "Date":                dt.strftime("%Y-%m-%d"),
        "Ward_No":             ward_no,
        "Ward_Name":           wb['ward_name'],
        "Water_Supplied_MLD":  round(v, 4),
        "Water_Consumed_MLD":  round(v*(1-LEAKAGE_PCT), 4),
        "Leakage_MLD":         round(v*LEAKAGE_PCT, 4),
        "Leakage_Percentage":  round(LEAKAGE_PCT*100, 2),
    } for dt, v in zip(dates, vals)]
    return jsonify(rows)


@app.route("/predict/all_wards/<string:date_str>")
@jwt_required()
def all_wards_on_date(date_str):
    try:
        rows = _CACHE_WARD_30[_CACHE_WARD_30["Date"] == date_str].to_dict(orient="records")
        if not rows:
            return jsonify({"error": f"No data for {date_str}"}), 404
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found"}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
