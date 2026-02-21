"""
app.py v3 — Panvel Smart Water Analytics | Flask Backend
FIX: All numpy int64/float64 values explicitly cast to Python int/float
     so Flask's JSON encoder never crashes.

Run:  python app_v3.py
Requires: city_models.pkl + ward_models.pkl  (from Prediction_Model_v3.ipynb)
"""

from flask import Flask, jsonify
from flask_cors import CORS
import pickle, os, logging, json
import numpy as np
import pandas as pd
from datetime import timedelta

# ──────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Custom JSON encoder: converts numpy types → native Python ──
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, pd.Timestamp): return str(obj.date())
        return super().default(obj)

app.json_encoder = NumpyEncoder

# ──────────────────────────────────────────────────────────────
# LOAD MODELS
# ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE, "city_models.pkl"), "rb") as f:
        CITY_BUNDLE = pickle.load(f)
    CITY_MODELS = CITY_BUNDLE["models"]
    CITY_DF     = CITY_BUNDLE["city_df"]

    with open(os.path.join(BASE, "ward_models.pkl"), "rb") as f:
        WARD_MODELS_RAW = pickle.load(f)

    # Convert numpy int64 keys → plain Python int  (THE KEY FIX)
    WARD_MODELS = {int(k): v for k, v in WARD_MODELS_RAW.items()}

    log.info("✅ Models loaded")
except FileNotFoundError as e:
    log.error(f"❌ {e}  — Run Prediction_Model_v3.ipynb first.")
    raise

WARDS = sorted(WARD_MODELS.keys())   # now plain Python ints

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


def _forecast_series(bundle, days, day_num_base):
    model     = bundle["model"]
    scaler    = bundle["scaler"]
    lv        = bundle["last_vals"]
    raw_hist  = [float(x) for x in lv["raw"]]
    diff_hist = [float(x) for x in lv["diff"]]
    last_date = lv["last_date"]
    dn = int(day_num_base)

    preds = []
    for i in range(days):
        fut   = last_date + timedelta(days=i+1)
        row_s = scaler.transform(_build_row(raw_hist, diff_hist, fut, dn))
        delta = float(model.predict(row_s)[0])
        val   = max(0.0, raw_hist[-1] + delta)
        preds.append(val)
        raw_hist.append(val)
        diff_hist.append(delta)
        dn += 1
    return preds


def _city_forecast(days):
    dn        = int((CITY_DF["Date"].max() - CITY_DF["Date"].min()).days)
    last_date = CITY_DF["Date"].max()
    sp = _forecast_series(CITY_MODELS["Water_Supplied_MLD"], days, dn)
    cp = _forecast_series(CITY_MODELS["Water_Consumed_MLD"], days, dn)
    lp = _forecast_series(CITY_MODELS["Leakage_MLD"],        days, dn)
    s_rmse = float(CITY_MODELS["Water_Supplied_MLD"]["metrics"]["RMSE"])
    c_rmse = float(CITY_MODELS["Water_Consumed_MLD"]["metrics"]["RMSE"])

    out = []
    for i in range(days):
        s = max(0.0, sp[i]); c = max(0.0, min(cp[i], s)); l = max(0.0, lp[i])
        out.append({
            "Date":                      str((last_date + timedelta(days=i+1)).date()),
            "Predicted_Supply_MLD":      round(s, 3),
            "Supply_Lower":              round(max(0.0, s - 1.96*s_rmse), 3),
            "Supply_Upper":              round(s + 1.96*s_rmse, 3),
            "Predicted_Consumption_MLD": round(c, 3),
            "Consumption_Lower":         round(max(0.0, c - 1.96*c_rmse), 3),
            "Consumption_Upper":         round(c + 1.96*c_rmse, 3),
            "Predicted_Leakage_MLD":     round(l, 3),
            "Leakage_Percentage":        round(l/s*100, 2) if s > 0 else 0.0,
        })
    return out


def _ward_forecast(ward_no, days):
    wm        = WARD_MODELS[int(ward_no)]
    hist_df   = wm["history"]
    dn        = int((hist_df["Date"].max() - hist_df["Date"].min()).days)
    last_date = hist_df["Date"].max()
    sp = _forecast_series(wm["supply"],      days, dn)
    cp = _forecast_series(wm["consumption"], days, dn)
    lp = _forecast_series(wm["leakage"],     days, dn)
    s_rmse = float(wm["supply"]["metrics"]["RMSE"])
    c_rmse = float(wm["consumption"]["metrics"]["RMSE"])

    out = []
    for i in range(days):
        s = max(0.0, sp[i]); c = max(0.0, min(cp[i], s)); l = max(0.0, lp[i])
        out.append({
            "Date":                      str((last_date + timedelta(days=i+1)).date()),
            "Ward_No":                   int(ward_no),
            "Ward_Name":                 wm["ward_name"],
            "Predicted_Supply_MLD":      round(s, 3),
            "Supply_Lower":              round(max(0.0, s - 1.96*s_rmse), 3),
            "Supply_Upper":              round(s + 1.96*s_rmse, 3),
            "Predicted_Consumption_MLD": round(c, 3),
            "Consumption_Lower":         round(max(0.0, c - 1.96*c_rmse), 3),
            "Consumption_Upper":         round(c + 1.96*c_rmse, 3),
            "Predicted_Leakage_MLD":     round(l, 3),
            "Leakage_Percentage":        round(l/s*100, 2) if s > 0 else 0.0,
        })
    return out


# ──────────────────────────────────────────────────────────────
# STARTUP CACHE
# ──────────────────────────────────────────────────────────────
log.info("Pre-computing 30-day forecast cache...")
_CACHE_CITY_30 = pd.DataFrame(_city_forecast(30))
_CACHE_WARD_30 = pd.DataFrame([p for wno in WARDS for p in _ward_forecast(wno, 30)])
log.info("✅ Cache ready")


# ──────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return jsonify({"status": "Panvel Water Analytics API v3", "wards": len(WARDS)})


@app.route("/predict/city/<int:days>")
def predict_city(days):
    try:
        if days not in [1, 7, 30]:
            return jsonify({"error": "days must be 1, 7 or 30"}), 400
        if days == 30:
            return jsonify(_CACHE_CITY_30.to_dict(orient="records"))
        return jsonify(_city_forecast(days))
    except Exception as e:
        log.exception("predict_city error")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/ward/<int:ward_no>/<int:days>")
def predict_ward(ward_no, days):
    try:
        if ward_no not in WARD_MODELS:
            return jsonify({"error": f"Ward {ward_no} not found"}), 404
        if days not in [1, 7, 30]:
            return jsonify({"error": "days must be 1, 7 or 30"}), 400
        if days == 30:
            result = _CACHE_WARD_30[_CACHE_WARD_30["Ward_No"] == ward_no].to_dict(orient="records")
            return jsonify(result)
        return jsonify(_ward_forecast(ward_no, days))
    except Exception as e:
        log.exception("predict_ward error")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/all_wards/<string:date_str>")
def all_wards_on_date(date_str):
    try:
        rows = _CACHE_WARD_30[_CACHE_WARD_30["Date"] == date_str].to_dict(orient="records")
        if not rows:
            return jsonify({"error": f"No data for {date_str}. Must be within next 30 days."}), 404
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/wards")
def wards():
    try:
        return jsonify([
            {"Ward_No": int(k), "Ward_Name": v["ward_name"]}
            for k, v in sorted(WARD_MODELS.items())
        ])
    except Exception as e:
        log.exception("wards error")
        return jsonify({"error": str(e)}), 500


@app.route("/summary")
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
                "City_Supply_CV_R2":   float(CITY_MODELS["Water_Supplied_MLD"]["metrics"]["CV_R2"]),
                "City_Supply_MAE":     float(CITY_MODELS["Water_Supplied_MLD"]["metrics"]["MAE"]),
            },
        })
    except Exception as e:
        log.exception("summary error")
        return jsonify({"error": str(e)}), 500


@app.route("/metrics")
def metrics():
    try:
        return jsonify([{
            "Ward_No":        int(wno),
            "Ward_Name":      wm["ward_name"],
            "Supply_R2":      float(wm["supply"]["metrics"]["R2"]),
            "Supply_MAE":     float(wm["supply"]["metrics"]["MAE"]),
            "Supply_CV_R2":   float(wm["supply"]["metrics"]["CV_R2"]),
            "Consumption_R2": float(wm["consumption"]["metrics"]["R2"]),
            "Leakage_R2":     float(wm["leakage"]["metrics"]["R2"]),
        } for wno, wm in sorted(WARD_MODELS.items())])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/historical/city")
def historical_city():
    try:
        cdf = CITY_DF.copy()
        cdf["Date"] = cdf["Date"].dt.strftime("%Y-%m-%d")
        cdf["Leakage_Percentage"] = (cdf["Leakage_MLD"]/cdf["Water_Supplied_MLD"]*100).round(2)
        return jsonify(cdf.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/historical/ward/<int:ward_no>")
def historical_ward(ward_no):
    try:
        if ward_no not in WARD_MODELS:
            return jsonify({"error": f"Ward {ward_no} not found"}), 404
        h = WARD_MODELS[ward_no]["history"].copy()
        h["Date"] = h["Date"].dt.strftime("%Y-%m-%d")
        h["Ward_Name"] = WARD_MODELS[ward_no]["ward_name"]
        return jsonify(h.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found", "routes": [
        "/", "/wards", "/summary", "/metrics",
        "/predict/city/<1|7|30>",
        "/predict/ward/<ward_no>/<1|7|30>",
        "/predict/all_wards/<YYYY-MM-DD>",
        "/historical/city", "/historical/ward/<ward_no>",
    ]}), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
