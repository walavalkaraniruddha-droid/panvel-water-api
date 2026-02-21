from flask import Flask, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
from datetime import timedelta
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

# ======================================
# LOAD MODELS
# ======================================

with open("supply_model.pkl", "rb") as f:
    supply_model = pickle.load(f)

with open("consumption_model.pkl", "rb") as f:
    consumption_model = pickle.load(f)

print("✅ Models Loaded Successfully")

# ======================================
# MODEL ACCURACY
# ======================================

MODEL_ACCURACY = {
    "Supply_R2": 0.9763,
    "Consumption_R2": 0.9792
}

# ======================================
# CONNECT TO MONGODB
# ======================================

client = MongoClient("mongodb://localhost:27017/")
db = client["panvel_water_db"]
collection = db["water_analytics"]

data = list(collection.find({}, {"_id": 0}))
original_df = pd.DataFrame(data)

original_df["Date"] = pd.to_datetime(original_df["Date"])

print("✅ Raw Ward-Level Data Loaded")

# ======================================
# CITY LEVEL DATA
# ======================================

city_df = original_df.groupby("Date").agg({
    "Water_Supplied_MLD": "sum",
    "Water_Consumed_MLD": "sum"
}).reset_index()

city_df = city_df.sort_values("Date")

print("✅ City-Level Data Ready")

# ======================================
# FORECAST FUNCTION
# ======================================

def forecast_days(dataframe, days=7):

    df = dataframe.copy()
    predictions = []

    for _ in range(days):

        last_row = df.iloc[-1]
        next_date = last_row["Date"] + timedelta(days=1)

        features_supply = pd.DataFrame([{
            "Supply_Lag1": df.iloc[-1]["Water_Supplied_MLD"],
            "Supply_Lag2": df.iloc[-2]["Water_Supplied_MLD"],
            "Supply_Lag3": df.iloc[-3]["Water_Supplied_MLD"],
            "Supply_Roll7": df["Water_Supplied_MLD"].tail(7).mean(),
            "Month": next_date.month,
            "DayOfWeek": next_date.dayofweek
        }])

        features_cons = pd.DataFrame([{
            "Cons_Lag1": df.iloc[-1]["Water_Consumed_MLD"],
            "Cons_Lag2": df.iloc[-2]["Water_Consumed_MLD"],
            "Cons_Lag3": df.iloc[-3]["Water_Consumed_MLD"],
            "Cons_Roll7": df["Water_Consumed_MLD"].tail(7).mean(),
            "Month": next_date.month,
            "DayOfWeek": next_date.dayofweek
        }])

        pred_supply = float(supply_model.predict(features_supply)[0])
        pred_cons = float(consumption_model.predict(features_cons)[0])
        pred_leakage = pred_supply - pred_cons

        predictions.append({
            "Date": str(next_date.date()),
            "Predicted_Supply_MLD": round(pred_supply, 3),
            "Predicted_Consumption_MLD": round(pred_cons, 3),
            "Predicted_Leakage_MLD": round(pred_leakage, 3)
        })

        new_row = {
            "Date": next_date,
            "Water_Supplied_MLD": pred_supply,
            "Water_Consumed_MLD": pred_cons
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return predictions

# ======================================
# ROUTES
# ======================================

@app.route("/predict/<int:days>", methods=["GET"])
def predict(days):

    if days not in [1, 7, 30]:
        return jsonify({"error": "Only 1, 7 or 30 days allowed."})

    result = forecast_days(city_df, days)
    return jsonify(result)


@app.route("/model_accuracy", methods=["GET"])
def model_accuracy():
    return jsonify(MODEL_ACCURACY)

@app.route("/area_comparison/<string:selected_date>", methods=["GET"])
def area_comparison(selected_date):

    date_obj = pd.to_datetime(selected_date)

    # 1️⃣ Try historical data first
    filtered_df = original_df[original_df["Date"] == date_obj]

    if not filtered_df.empty:

        area_df = filtered_df.groupby("Ward_Name").agg({
            "Water_Supplied_MLD": "sum",
            "Water_Consumed_MLD": "sum"
        }).reset_index()

        area_df["Leakage_MLD"] = (
            area_df["Water_Supplied_MLD"] -
            area_df["Water_Consumed_MLD"]
        ).round(2)

        return jsonify(area_df.to_dict(orient="records"))

    # 2️⃣ If future date → distribute based on latest ratio

    latest_date = original_df["Date"].max()
    latest_df = original_df[original_df["Date"] == latest_date]

    area_df = latest_df.groupby("Ward_Name").agg({
        "Water_Supplied_MLD": "sum",
        "Water_Consumed_MLD": "sum"
    }).reset_index()

    area_df["Leakage_MLD"] = (
        area_df["Water_Supplied_MLD"] -
        area_df["Water_Consumed_MLD"]
    )

    total_leakage = area_df["Leakage_MLD"].sum()

    # Get predicted city leakage for selected date
    city_predictions = forecast_days(city_df, 30)
    city_pred_df = pd.DataFrame(city_predictions)

    row = city_pred_df[city_pred_df["Date"] == selected_date]

    if row.empty:
        return jsonify([])

    predicted_city_leakage = float(row["Predicted_Leakage_MLD"].values[0])

    # Apply proportional distribution
    area_df["Leakage_MLD"] = (
        (area_df["Leakage_MLD"] / total_leakage)
        * predicted_city_leakage
    ).round(2)

    return jsonify(area_df[["Ward_Name", "Leakage_MLD"]]
                   .to_dict(orient="records"))


# ======================================
# RUN
# ======================================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
