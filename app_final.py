"""
app_final.py — Panvel Water Analytics | Flask Backend with Authentication
Simple username/password login stored in SQLite.

Run:  python app_final.py
Requires: city_models.pkl + ward_models.pkl
Install:  pip install flask flask-cors flask-jwt-extended werkzeug
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, create_access_token,
    jwt_required, get_jwt_identity
)
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3, pickle, os, logging, json
import numpy as np
import pandas as pd
from datetime import timedelta

# ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JWT_SECRET_KEY"]           = os.environ.get("JWT_SECRET_KEY", "panvel-water-analytics-secret-key-2026-strong")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=30)

CORS(app, origins="*")
jwt = JWTManager(app)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# NUMPY JSON ENCODER
# ──────────────────────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.ndarray):   return obj.tolist()
        if isinstance(obj, pd.Timestamp): return str(obj.date())
        return super().default(obj)

app.json_encoder = NumpyEncoder

# ──────────────────────────────────────────────────────────────
# SQLITE — USER DATABASE
# ──────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "users.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT    NOT NULL,
            email    TEXT    NOT NULL UNIQUE,
            password TEXT    NOT NULL,
            created  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_user_by_email(email):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    return dict(user) if user else None

def create_user(name, email, password):
    hashed = generate_password_hash(password)
    conn   = sqlite3.connect(DB_PATH)
    try:
        conn.execute("INSERT INTO users (name, email, password) VALUES (?,?,?)",
                     (name, email, hashed))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# ──────────────────────────────────────────────────────────────
# LOAD ML MODELS
# ──────────────────────────────────────────────────────────────
try:
    with open(os.path.join(BASE, "city_models.pkl"), "rb") as f:
        CITY_BUNDLE = pickle.load(f)
    CITY_MODELS = CITY_BUNDLE["models"]
    CITY_DF     = CITY_BUNDLE["city_df"]

    with open(os.path.join(BASE, "ward_models.pkl"), "rb") as f:
        WARD_MODELS = {int(k): v for k, v in pickle.load(f).items()}

    log.info("✅ ML Models loaded")
except FileNotFoundError as e:
    log.error(f"❌ {e} — Run Prediction_Model_v3.ipynb first.")
    raise

WARDS = sorted(WARD_MODELS.keys())

# ──────────────────────────────────────────────────────────────
# FORECAST HELPERS
# ──────────────────────────────────────────────────────────────
def _build_row(raw_hist, diff_hist, future_date, day_num):
    rh = np.array(raw_hist, dtype=float)
    dh = np.array(diff_hist, dtype=float)
    def _get(arr, n, fb): return float(arr[-n]) if len(arr) >= n else float(fb)
    def _mean(arr, n):    return float(np.mean(arr[-n:])) if len(arr) >= n else float(arr[-1])
    m, dom, dow = future_date.month, future_date.day, future_date.dayofweek
    return np.array([[
        float(rh[-1]), _get(rh,2,rh[-1]), _get(rh,3,rh[-1]), _get(rh,7,rh[-1]),
        float(dh[-1]), _get(dh,2,dh[-1]), _get(dh,3,dh[-1]),
        _mean(dh,3), _mean(dh,7), _mean(dh,14),
        _mean(rh,3), _mean(rh,7),
        float(m), float(dom), float(dow), float(day_num),
        float(dom == 1),
        np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12),
        np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),
    ]])

def _forecast_series(bundle, days, day_num_base, start_from_date=None, seed_raw=None, seed_diff=None):
    """Rolling forecast — supports unlimited days by chaining predictions."""
    model     = bundle["model"]
    scaler    = bundle["scaler"]
    lv        = bundle["last_vals"]
    raw_hist  = list(seed_raw)  if seed_raw  is not None else [float(x) for x in lv["raw"]]
    diff_hist = list(seed_diff) if seed_diff is not None else [float(x) for x in lv["diff"]]
    last_date = start_from_date if start_from_date is not None else lv["last_date"]
    dn = int(day_num_base)
    preds = []
    for i in range(days):
        fut   = last_date + timedelta(days=i+1)
        row_s = scaler.transform(_build_row(raw_hist, diff_hist, fut, dn))
        delta = float(model.predict(row_s)[0])
        val   = max(0.0, raw_hist[-1] + delta)
        preds.append(val)
        raw_hist.append(val); diff_hist.append(delta); dn += 1
    return preds, raw_hist, diff_hist

def _city_forecast(days):
    """Supports any number of days via rolling prediction."""
    dn        = int((CITY_DF["Date"].max() - CITY_DF["Date"].min()).days)
    last_date = CITY_DF["Date"].max()
    s_rmse = float(CITY_MODELS["Water_Supplied_MLD"]["metrics"]["RMSE"])
    c_rmse = float(CITY_MODELS["Water_Consumed_MLD"]["metrics"]["RMSE"])
    sp, _, _  = _forecast_series(CITY_MODELS["Water_Supplied_MLD"],  days, dn)
    cp, _, _  = _forecast_series(CITY_MODELS["Water_Consumed_MLD"],  days, dn)
    lp, _, _  = _forecast_series(CITY_MODELS["Leakage_MLD"],          days, dn)
    out = []
    for i in range(days):
        s = max(0.0, sp[i]); c = max(0.0, min(cp[i], s)); l = max(0.0, lp[i])
        out.append({
            "Date":                      str((last_date + timedelta(days=i+1)).date()),
            "Predicted_Supply_MLD":      round(s, 3),
            "Supply_Lower":              round(max(0.0, s-1.96*s_rmse), 3),
            "Supply_Upper":              round(s+1.96*s_rmse, 3),
            "Predicted_Consumption_MLD": round(c, 3),
            "Consumption_Lower":         round(max(0.0, c-1.96*c_rmse), 3),
            "Consumption_Upper":         round(c+1.96*c_rmse, 3),
            "Predicted_Leakage_MLD":     round(l, 3),
            "Leakage_Percentage":        round(l/s*100, 2) if s > 0 else 0.0,
        })
    return out

def _ward_forecast(ward_no, days):
    """Supports any number of days via rolling prediction."""
    wm        = WARD_MODELS[int(ward_no)]
    hist_df   = wm["history"]
    dn        = int((hist_df["Date"].max() - hist_df["Date"].min()).days)
    last_date = hist_df["Date"].max()
    s_rmse = float(wm["supply"]["metrics"]["RMSE"])
    c_rmse = float(wm["consumption"]["metrics"]["RMSE"])
    sp, _, _ = _forecast_series(wm["supply"],      days, dn)
    cp, _, _ = _forecast_series(wm["consumption"], days, dn)
    lp, _, _ = _forecast_series(wm["leakage"],     days, dn)
    out = []
    for i in range(days):
        s = max(0.0, sp[i]); c = max(0.0, min(cp[i], s)); l = max(0.0, lp[i])
        out.append({
            "Date":                      str((last_date + timedelta(days=i+1)).date()),
            "Ward_No":                   int(ward_no),
            "Ward_Name":                 wm["ward_name"],
            "Predicted_Supply_MLD":      round(s, 3),
            "Supply_Lower":              round(max(0.0, s-1.96*s_rmse), 3),
            "Supply_Upper":              round(s+1.96*s_rmse, 3),
            "Predicted_Consumption_MLD": round(c, 3),
            "Consumption_Lower":         round(max(0.0, c-1.96*c_rmse), 3),
            "Consumption_Upper":         round(c+1.96*c_rmse, 3),
            "Predicted_Leakage_MLD":     round(l, 3),
            "Leakage_Percentage":        round(l/s*100, 2) if s > 0 else 0.0,
        })
    return out

# ── Startup cache ──────────────────────────────────────────────
log.info("Building forecast cache...")
_CACHE_CITY_30 = pd.DataFrame(_city_forecast(30))
_CACHE_WARD_30 = pd.DataFrame([p for wno in WARDS for p in _ward_forecast(wno, 30)])
log.info("✅ Cache ready")


# ══════════════════════════════════════════════════════════════
# AUTH ROUTES  (no JWT required)
# ══════════════════════════════════════════════════════════════

@app.route("/auth/register", methods=["POST"])
def register():
    data     = request.get_json()
    name     = data.get("name", "").strip()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not name or not email or not password:
        return jsonify({"error": "Name, email and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    if "@" not in email:
        return jsonify({"error": "Invalid email address"}), 400

    if not create_user(name, email, password):
        return jsonify({"error": "Email already registered. Please login."}), 409

    token = create_access_token(identity=email)
    return jsonify({"message": "Account created!", "token": token, "name": name, "email": email}), 201


@app.route("/auth/login", methods=["POST"])
def login():
    data     = request.get_json()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    user = get_user_by_email(email)
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid email or password"}), 401

    token = create_access_token(identity=email)
    return jsonify({"token": token, "name": user["name"], "email": email}), 200


@app.route("/auth/me", methods=["GET"])
@jwt_required()
def me():
    email = get_jwt_identity()
    user  = get_user_by_email(email)
    return jsonify({"email": email, "name": user["name"] if user else ""}), 200


# ══════════════════════════════════════════════════════════════
# PROTECTED DATA ROUTES  (JWT required)
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return jsonify({"status": "Panvel Water Analytics API", "version": "final"})


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
        if ward_no not in WARD_MODELS:
            return jsonify({"error": f"Ward {ward_no} not found"}), 404
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
    """Scan all wards for leakage alerts — returns triggered wards with level."""
    try:
        threshold = request.args.get("threshold", 10, type=float)
        days      = request.args.get("days", 30, type=int)
        days      = max(1, min(days, 365))
        results   = []
        for wno in WARDS:
            data = _ward_forecast(wno, days)
            leak_pcts = [r["Leakage_Percentage"] for r in data]
            avg_pct   = sum(leak_pcts) / len(leak_pcts)
            max_pct   = max(leak_pcts)
            avg_mld   = sum(r["Predicted_Leakage_MLD"] for r in data) / len(data)
            days_exc  = sum(1 for p in leak_pcts if p >= threshold)
            # Determine level
            if avg_pct >= 20:   level = "CRITICAL"
            elif avg_pct >= 15: level = "HIGH"
            elif avg_pct >= threshold: level = "MODERATE"
            else:               level = "NORMAL"
            # Find peak day
            peak_idx  = leak_pcts.index(max_pct)
            peak_date = data[peak_idx]["Date"]
            results.append({
                "Ward_No":      wno,
                "Ward_Name":    WARD_MODELS[wno]["ward_name"],
                "Level":        level,
                "Avg_Leakage_Pct":  round(avg_pct, 2),
                "Max_Leakage_Pct":  round(max_pct, 2),
                "Avg_Leakage_MLD":  round(avg_mld, 3),
                "Days_Exceeding":   days_exc,
                "Peak_Date":        peak_date,
                "Start_Date":       data[0]["Date"],
                "End_Date":         data[-1]["Date"],
                "Daily_Pct":        [round(p,2) for p in leak_pcts],
            })
        # Sort: CRITICAL first, then HIGH, MODERATE, NORMAL
        level_order = {"CRITICAL":0, "HIGH":1, "MODERATE":2, "NORMAL":3}
        results.sort(key=lambda x: (level_order[x["Level"]], -x["Avg_Leakage_Pct"]))
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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


@app.route("/wards")
@jwt_required()
def wards():
    return jsonify([{"Ward_No": int(k), "Ward_Name": v["ward_name"]}
                    for k, v in sorted(WARD_MODELS.items())])


@app.route("/summary")
@jwt_required()
def summary():
    try:
        cdf   = CITY_DF
        top7  = _CACHE_WARD_30[_CACHE_WARD_30["Date"].isin(_CACHE_WARD_30["Date"].unique()[:7])]
        worst = top7.groupby(["Ward_No","Ward_Name"])["Predicted_Leakage_MLD"].mean().idxmax()
        return jsonify({
            "Last_Data_Date":          str(cdf["Date"].max().date()),
            "Avg_Daily_Supply_MLD":    round(float(cdf["Water_Supplied_MLD"].mean()), 3),
            "Avg_Daily_Leakage_MLD":   round(float(cdf["Leakage_MLD"].mean()), 3),
            "Avg_Leakage_Percentage":  round(float((cdf["Leakage_MLD"]/cdf["Water_Supplied_MLD"]*100).mean()), 2),
            "Highest_Leakage_Ward_No": int(worst[0]),
            "Highest_Leakage_Ward":    str(worst[1]),
            "Total_Wards":             len(WARD_MODELS),
            "Model_Accuracy": {
                "City_Supply_R2":      float(CITY_MODELS["Water_Supplied_MLD"]["metrics"]["R2"]),
                "City_Consumption_R2": float(CITY_MODELS["Water_Consumed_MLD"]["metrics"]["R2"]),
                "City_Leakage_R2":     float(CITY_MODELS["Leakage_MLD"]["metrics"]["R2"]),
                "City_Supply_MAE":     float(CITY_MODELS["Water_Supplied_MLD"]["metrics"]["MAE"]),
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/metrics")
@jwt_required()
def metrics():
    return jsonify([{
        "Ward_No":        int(wno),
        "Ward_Name":      wm["ward_name"],
        "Supply_R2":      float(wm["supply"]["metrics"]["R2"]),
        "Supply_MAE":     float(wm["supply"]["metrics"]["MAE"]),
        "Consumption_R2": float(wm["consumption"]["metrics"]["R2"]),
        "Leakage_R2":     float(wm["leakage"]["metrics"]["R2"]),
    } for wno, wm in sorted(WARD_MODELS.items())])


@app.route("/historical/city")
@jwt_required()
def historical_city():
    cdf = CITY_DF.copy()
    cdf["Date"] = cdf["Date"].dt.strftime("%Y-%m-%d")
    cdf["Leakage_Percentage"] = (cdf["Leakage_MLD"]/cdf["Water_Supplied_MLD"]*100).round(2)
    return jsonify(cdf.to_dict(orient="records"))


@app.route("/historical/ward/<int:ward_no>")
@jwt_required()
def historical_ward(ward_no):
    if ward_no not in WARD_MODELS:
        return jsonify({"error": f"Ward {ward_no} not found"}), 404
    h = WARD_MODELS[ward_no]["history"].copy()
    h["Date"] = h["Date"].dt.strftime("%Y-%m-%d")
    h["Ward_Name"] = WARD_MODELS[ward_no]["ward_name"]
    return jsonify(h.to_dict(orient="records"))


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found"}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
