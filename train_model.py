import pandas as pd
import numpy as np
import pickle
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

print("Connecting to MongoDB...")

client = MongoClient("mongodb://localhost:27017/")
db = client["panvel_water_db"]
collection = db["water_analytics"]

data = list(collection.find())

if len(data) == 0:
    print("❌ No data found in MongoDB")
    exit()

df = pd.DataFrame(data)
df = df.drop(columns=["_id"], errors="ignore")

print("✅ Data Loaded Successfully")
print("Total Rows:", len(df))

# ==============================
# CONVERT DATE
# ==============================
df["Date"] = pd.to_datetime(df["Date"])

# ==============================
# GROUP BY DATE (CITY LEVEL)
# ==============================
city_df = df.groupby("Date").agg({
    "Water_Supplied_MLD": "sum",
    "Water_Consumed_MLD": "sum"
}).reset_index()

city_df = city_df.sort_values("Date")

print("✅ Converted to City-Level Daily Data")
print("Total Days:", len(city_df))

# ==============================
# FEATURE ENGINEERING
# ==============================

city_df["Supply_Lag1"] = city_df["Water_Supplied_MLD"].shift(1)
city_df["Supply_Lag2"] = city_df["Water_Supplied_MLD"].shift(2)
city_df["Supply_Lag3"] = city_df["Water_Supplied_MLD"].shift(3)

city_df["Cons_Lag1"] = city_df["Water_Consumed_MLD"].shift(1)
city_df["Cons_Lag2"] = city_df["Water_Consumed_MLD"].shift(2)
city_df["Cons_Lag3"] = city_df["Water_Consumed_MLD"].shift(3)

city_df["Supply_Roll7"] = city_df["Water_Supplied_MLD"].rolling(7).mean()
city_df["Cons_Roll7"] = city_df["Water_Consumed_MLD"].rolling(7).mean()

city_df["Month"] = city_df["Date"].dt.month
city_df["DayOfWeek"] = city_df["Date"].dt.dayofweek

city_df = city_df.dropna()

print("✅ Feature Engineering Completed")

# ==============================
# FEATURES
# ==============================

supply_features = [
    "Supply_Lag1", "Supply_Lag2", "Supply_Lag3",
    "Supply_Roll7", "Month", "DayOfWeek"
]

cons_features = [
    "Cons_Lag1", "Cons_Lag2", "Cons_Lag3",
    "Cons_Roll7", "Month", "DayOfWeek"
]

X_supply = city_df[supply_features]
y_supply = city_df["Water_Supplied_MLD"]

X_cons = city_df[cons_features]
y_cons = city_df["Water_Consumed_MLD"]

# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_supply, y_supply, test_size=0.2, shuffle=False
)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cons, y_cons, test_size=0.2, shuffle=False
)

# ==============================
# RANDOM FOREST
# ==============================

supply_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

cons_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

supply_model.fit(X_train_s, y_train_s)
cons_model.fit(X_train_c, y_train_c)

print("✅ Models Trained Successfully")

# ==============================
# EVALUATION
# ==============================

s_pred = supply_model.predict(X_test_s)
c_pred = cons_model.predict(X_test_c)

print("\n📊 MODEL PERFORMANCE")
print("----------------------------")
print("Supply R2:", round(r2_score(y_test_s, s_pred), 4))
print("Supply RMSE:", round(np.sqrt(mean_squared_error(y_test_s, s_pred)), 4))
print()
print("Consumption R2:", round(r2_score(y_test_c, c_pred), 4))
print("Consumption RMSE:", round(np.sqrt(mean_squared_error(y_test_c, c_pred)), 4))

# ==============================
# SAVE MODELS
# ==============================

with open("supply_model.pkl", "wb") as f:
    pickle.dump(supply_model, f)

with open("consumption_model.pkl", "wb") as f:
    pickle.dump(cons_model, f)

print("\n✅ Models Saved Successfully!")
print("\n🎉 TRAINING COMPLETE")
