"""
app_final.py — Panvel Water Analytics | Flask Backend v2
27-zone dataset (20 PMC wards + 7 CIDCO/MIDC/Village zones) = ~211 MLD
Matches PMC Environmental Status Report 2024-25 (IIT Bombay)
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3, pickle, os, logging, json, warnings
import numpy as np
import pandas as pd
from datetime import timedelta, datetime

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["JWT_SECRET_KEY"]           = os.environ.get("JWT_SECRET_KEY", "panvel-water-2026-viva-secret-key-strong")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=30)

CORS(app, origins="*")
jwt = JWTManager(app)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.ndarray):   return obj.tolist()
        if isinstance(obj, pd.Timestamp): return str(obj.date())
        return super().default(obj)

app.json_encoder = NumpyEncoder

# ── SQLITE USER DB ──────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "users.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            created DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit(); conn.close()

init_db()

def get_user_by_email(email):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    u = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    return dict(u) if u else None

def create_user(name, email, password):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("INSERT INTO users (name,email,password) VALUES (?,?,?)",
                     (name, email, generate_password_hash(password)))
        conn.commit(); return True
    except sqlite3.IntegrityError: return False
    finally: conn.close()

# ── LOAD MODELS ─────────────────────────────────────────────
try:
    with open(os.path.join(BASE, "city_models.pkl"), "rb") as f:
        CITY_BUNDLE = pickle.load(f)
    with open(os.path.join(BASE, "ward_models.pkl"), "rb") as f:
        WARD_BUNDLE = {int(k): v for k, v in pickle.load(f).items()}
    log.info("✅ ML Models loaded")
except Exception as e:
    log.error(f"❌ {e}")
    raise

WARDS = sorted(WARD_BUNDLE.keys())
LEAKAGE_PCT = 0.0952  # Official PMC value

FEATURE_COLS = ([f'lag_{i}' for i in range(1,15)] +
    ['diff_1','diff_7','rolling_mean_7','rolling_mean_14','rolling_std_7',
     'day_of_year','day_of_week','month'])

def _make_features_for_pred(last_values):
    """Build feature vector from last 14 values for rolling prediction."""
    v = np.array(last_values, dtype=float)
    if len(v) < 14: v = np.pad(v, (14-len(v), 0), 'edge')
    v = v[-14:]
    diff1 = float(v[-1] - v[-2])
    diff7 = float(v[-1] - v[-8]) if len(v) >= 8 else diff1
    rm7  = float(v[-7:].mean())
    rm14 = float(v.mean())
    rstd = float(v[-7:].std()) if v[-7:].std() > 0 else 0.001
    today = datetime.now()
    row = (list(v[::-1][:14]) +   # lags 1-14
           [diff1, diff7, rm7, rm14, rstd,
            float(today.timetuple().tm_yday),
            float(today.weekday()),
            float(today.month)])
    return np.array(row).reshape(1, -1)

def _city_forecast(days):
    results = []
    supply_hist  = list(CITY_BUNDLE['Water_Supplied_MLD']['last_values'])
    consume_hist = list(CITY_BUNDLE['Water_Consumed_MLD']['last_values'])
    supply_model  = CITY_BUNDLE['Water_Supplied_MLD']['model']
    consume_model = CITY_BUNDLE['Water_Consumed_MLD']['model']

    for d in range(days):
        future = datetime.now() + timedelta(days=d+1)
        # Build features
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
        leak_pct = (pred_leakage / pred_supply * 100)

        results.append({
            "Date": future.strftime("%Y-%m-%d"),
            "Day": d+1,
            "Predicted_Supply_MLD":       round(pred_supply, 3),
            "Predicted_Consumption_MLD":  round(pred_consume, 3),
            "Predicted_Leakage_MLD":      round(pred_leakage, 3),
            "Leakage_Percentage":         round(leak_pct, 2),
        })
        supply_hist.append(pred_supply)
        consume_hist.append(pred_consume)

    return results

def _ward_forecast(ward_no, days):
    wb = WARD_BUNDLE[ward_no]
    model = wb['model']
    hist  = list(wb['last_values'])
    leak_pct = wb.get('leakage_pct', LEAKAGE_PCT)
    results = []

    for d in range(days):
        future = datetime.now() + timedelta(days=d+1)
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
        pred_supply = max(float(model.predict(feats)[0]), 0.1)
        pred_consume = round(pred_supply * (1 - leak_pct), 4)
        pred_leakage = round(pred_supply - pred_consume, 4)

        results.append({
            "Date":                      future.strftime("%Y-%m-%d"),
            "Day":                       d+1,
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
_CACHE_CITY_30 = pd.DataFrame(_city_forecast(30))
_cache_ward_rows = []
for _wno in WARDS:
    _cache_ward_rows.extend(_ward_forecast(_wno, 30))
_CACHE_WARD_30 = pd.DataFrame(_cache_ward_rows)
log.info("✅ Cache ready")

# ── AUTH ROUTES ─────────────────────────────────────────────
@app.route("/auth/register", methods=["POST"])
def register():
    d = request.get_json()
    name  = d.get("name","").strip()
    email = d.get("email","").strip().lower()
    pwd   = d.get("password","")
    if not all([name, email, pwd]):
        return jsonify({"error": "All fields required"}), 400
    if len(pwd) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    if "@" not in email:
        return jsonify({"error": "Invalid email address"}), 400
    if not create_user(name, email, pwd):
        return jsonify({"error": "Email already registered"}), 409
    token = create_access_token(identity=email)
    return jsonify({"message": "Account created!", "token": token, "name": name, "email": email}), 201

@app.route("/auth/login", methods=["POST"])
def login():
    d = request.get_json()
    email = d.get("email","").strip().lower()
    pwd   = d.get("password","")
    user  = get_user_by_email(email)
    if not user or not check_password_hash(user["password"], pwd):
        return jsonify({"error": "Invalid email or password"}), 401
    token = create_access_token(identity=email)
    return jsonify({"token": token, "name": user["name"], "email": email}), 200

@app.route("/auth/me", methods=["GET"])
@jwt_required()
def me():
    email = get_jwt_identity()
    user  = get_user_by_email(email)
    return jsonify({"email": email, "name": user["name"] if user else ""}), 200

# ── DATA ROUTES ─────────────────────────────────────────────
@app.route("/")
def index():
    return jsonify({"status": "Panvel Water Analytics API v2", "zones": len(WARDS), "target_mld": 211})

@app.route("/summary")
@jwt_required()
def summary():
    try:
        hist = CITY_BUNDLE['city_history']
        city_df = pd.DataFrame(hist)
        avg_supply = float(city_df['Water_Supplied_MLD'].mean())
        avg_leak   = float(city_df['Leakage_MLD'].mean())
        worst_wno  = max(WARD_BUNDLE.items(), key=lambda x: x[1].get('rmse',0))[0]
        return jsonify({
            "Last_Data_Date":         "2026-02-28",
            "Avg_Daily_Supply_MLD":   round(avg_supply, 2),
            "Avg_Daily_Leakage_MLD":  round(avg_leak, 3),
            "Avg_Leakage_Percentage": round(LEAKAGE_PCT * 100, 2),
            "Highest_Leakage_Ward_No": int(worst_wno),
            "Highest_Leakage_Ward":   WARD_BUNDLE[worst_wno]['ward_name'],
            "Total_Zones":            len(WARDS),
            "Total_Wards":            len(WARDS),
            "Data_Coverage_MLD":      "~211 MLD (aligned with PMC ESR 2024-25)",
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
            return jsonify({"error": "days 1-365"}), 400
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
            return jsonify({"error": "days 1-365"}), 400
        if days == 30:
            result = _CACHE_WARD_30[_CACHE_WARD_30["Ward_No"]==ward_no].to_dict(orient="records")
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
            data     = _ward_forecast(wno, days)
            leak_pcts = [r["Leakage_Percentage"] for r in data]
            avg_pct  = sum(leak_pcts)/len(leak_pcts)
            max_pct  = max(leak_pcts)
            avg_mld  = sum(r["Predicted_Leakage_MLD"] for r in data)/len(data)
            days_exc = sum(1 for p in leak_pcts if p >= threshold)
            if   avg_pct >= 20: level = "CRITICAL"
            elif avg_pct >= 15: level = "HIGH"
            elif avg_pct >= threshold: level = "MODERATE"
            else:               level = "NORMAL"
            peak_idx  = leak_pcts.index(max_pct)
            results.append({
                "Ward_No":          wno,
                "Ward_Name":        WARD_BUNDLE[wno]["ward_name"],
                "Level":            level,
                "Avg_Leakage_Pct":  round(avg_pct, 2),
                "Max_Leakage_Pct":  round(max_pct, 2),
                "Avg_Leakage_MLD":  round(avg_mld, 3),
                "Days_Exceeding":   days_exc,
                "Peak_Date":        data[peak_idx]["Date"],
                "Start_Date":       data[0]["Date"],
                "End_Date":         data[-1]["Date"],
            })
        lo = {"CRITICAL":0,"HIGH":1,"MODERATE":2,"NORMAL":3}
        results.sort(key=lambda x:(lo[x["Level"]],-x["Avg_Leakage_Pct"]))
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/wards")
@jwt_required()
def wards():
    return jsonify([{"Ward_No": int(k), "Ward_Name": v["ward_name"]}
                    for k,v in sorted(WARD_BUNDLE.items())])

@app.route("/metrics")
@jwt_required()
def metrics():
    return jsonify([{
        "Ward_No":   int(wno),
        "Ward_Name": wb["ward_name"],
        "Supply_R2": round(float(wb["r2"]),4),
        "Supply_RMSE": round(float(wb["rmse"]),4),
    } for wno,wb in sorted(WARD_BUNDLE.items())])

@app.route("/historical/city")
@jwt_required()
def historical_city():
    try:
        hist = CITY_BUNDLE['city_history']
        df = pd.DataFrame(hist)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime("%Y-%m-%d")
        df['Leakage_Percentage'] = (df['Leakage_MLD']/df['Water_Supplied_MLD']*100).round(2)
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/historical/ward/<int:ward_no>")
@jwt_required()
def historical_ward(ward_no):
    if ward_no not in WARD_BUNDLE:
        return jsonify({"error": f"Zone {ward_no} not found"}), 404
    wb = WARD_BUNDLE[ward_no]
    vals = wb['last_values']
    # Generate dates going back from Feb 28 2026
    dates = pd.date_range(end='2026-02-28', periods=len(vals), freq='D')
    rows = []
    for dt, v in zip(dates, vals):
        rows.append({
            "Date": dt.strftime("%Y-%m-%d"),
            "Ward_No": ward_no,
            "Ward_Name": wb['ward_name'],
            "Water_Supplied_MLD": round(v,4),
            "Water_Consumed_MLD": round(v*(1-LEAKAGE_PCT),4),
            "Leakage_MLD": round(v*LEAKAGE_PCT,4),
            "Leakage_Percentage": round(LEAKAGE_PCT*100,2),
        })
    return jsonify(rows)

@app.route("/predict/all_wards/<string:date_str>")
@jwt_required()
def all_wards_on_date(date_str):
    try:
        rows = _CACHE_WARD_30[_CACHE_WARD_30["Date"]==date_str].to_dict(orient="records")
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
