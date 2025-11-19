
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score, 
    mean_absolute_percentage_error
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ===========================================
# LOAD DATASET
# ===========================================
df = pd.read_csv("housing.csv")   

print("===== 5 Data Teratas =====")
print(df.head())

print("\n===== Info Dataset =====")
print(df.info())

print("\n===== Cek Missing Value =====")
print(df.isnull().sum())


# ===========================================
# HANDLE MISSING VALUE
# ===========================================
# data kosong → isi median
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

# ===========================================
# ENCODING KATEGORI 'ocean_proximity'
# ===========================================
df = pd.get_dummies(df, drop_first=True)

# ===========================================
# PISAHKAN FITUR & TARGET
# ===========================================
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]


# ===========================================
# SPLIT DATA TRAIN & TEST
# ===========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nJumlah data train:", len(X_train))
print("Jumlah data test :", len(X_test))

# ===========================================
# MODEL 1 — XGBOOST REGRESSOR
# ===========================================
print("\n==============================")
print("Training XGBoost Regressor")
print("==============================")

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb.fit(X_train, y_train)

# Prediksi
y_pred_xgb = xgb.predict(X_test)

# Evaluasi
mae_xgb  = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb  = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
mape_xgb = mean_absolute_percentage_error(y_test, y_pred_xgb)
r2_xgb   = r2_score(y_test, y_pred_xgb)

print("\n===== Evaluasi XGBoost =====")
print("MAE  :", mae_xgb)
print("MSE  :", mse_xgb)
print("RMSE :", rmse_xgb)
print("MAPE :", mape_xgb)
print("R²   :", r2_xgb)


# ===========================================
# MODEL 2 — LIGHTGBM REGRESSOR
# ===========================================
print("\n==============================")
print("Training LightGBM Regressor")
print("==============================")

lgbm = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=-1,
    random_state=42
)

lgbm.fit(X_train, y_train)

# Prediksi
y_pred_lgbm = lgbm.predict(X_test)

# Evaluasi
mae_lgbm  = mean_absolute_error(y_test, y_pred_lgbm)
mse_lgbm  = mean_squared_error(y_test, y_pred_lgbm)
rmse_lgbm = np.sqrt(mse_lgbm)
mape_lgbm = mean_absolute_percentage_error(y_test, y_pred_lgbm)
r2_lgbm   = r2_score(y_test, y_pred_lgbm)

print("\n===== Evaluasi LightGBM =====")
print("MAE  :", mae_lgbm)
print("MSE  :", mse_lgbm)
print("RMSE :", rmse_lgbm)
print("MAPE :", mape_lgbm)
print("R²   :", r2_lgbm)


# ===========================================
# VISUALISASI PREDIKSI
# ===========================================
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.5)
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.title("XGBoost — Actual vs Predicted")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lgbm, alpha=0.5)
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.title("LightGBM — Actual vs Predicted")
plt.grid(True)
plt.show()
