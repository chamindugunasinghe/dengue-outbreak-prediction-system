import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dengue_data.csv"
MODEL_PATH = BASE_DIR / "models" / "dengue_weekly_cases_regressor.pkl"


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)

    required_cols = [
        "District",
        "Year",
        "Week",
        "Month",
        "Number_of_Cases",
        "Avg Max Temp (°C)",
        "Avg Min Temp (°C)",
        "Total Precipitation (mm)",
        "Avg Wind Speed (km/h)",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].copy()
    for col in [
        "Year",
        "Week",
        "Month",
        "Number_of_Cases",
        "Avg Max Temp (°C)",
        "Avg Min Temp (°C)",
        "Total Precipitation (mm)",
        "Avg Wind Speed (km/h)",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["District", "Year", "Week", "Month", "Number_of_Cases"])
    df = df[df["Number_of_Cases"] >= 0]

    df["Year"] = df["Year"].astype(int)
    df["Week"] = df["Week"].astype(int)
    df["Month"] = df["Month"].astype(int)

    df = df.sort_values(["District", "Year", "Week"]).reset_index(drop=True)

    districts = sorted(df["District"].dropna().unique())
    district_encoding = {district: idx for idx, district in enumerate(districts)}
    df["District_Encoded"] = df["District"].map(district_encoding).astype(int)

    district_month_mean = (
        df.groupby(["District", "Month"])["Number_of_Cases"].mean().to_dict()
    )

    df["last_week_cases"] = df.groupby("District")["Number_of_Cases"].shift(1)
    df["last_2_weeks_cases"] = df.groupby("District")["Number_of_Cases"].shift(2)
    df["cases_3week_avg"] = (
        df.groupby("District")["Number_of_Cases"]
        .transform(lambda s: s.rolling(window=3).mean().shift(1))
    )

    global_mean = float(df["Number_of_Cases"].mean())
    df["month_fallback"] = df.apply(
        lambda row: district_month_mean.get((row["District"], row["Month"]), global_mean),
        axis=1,
    )

    df["last_week_cases"] = df["last_week_cases"].fillna(df["month_fallback"])
    df["last_2_weeks_cases"] = df["last_2_weeks_cases"].fillna(df["last_week_cases"])
    df["cases_3week_avg"] = df["cases_3week_avg"].fillna(
        (df["last_week_cases"] + df["last_2_weeks_cases"]) / 2.0
    )
    df = df.drop(columns=["month_fallback"])

    df["temp_range"] = df["Avg Max Temp (°C)"] - df["Avg Min Temp (°C)"]
    df["avg_temp"] = (df["Avg Max Temp (°C)"] + df["Avg Min Temp (°C)"]) / 2.0
    df["rain_temp_interaction"] = df["Total Precipitation (mm)"] * df["avg_temp"]
    df["is_monsoon"] = df["Month"].isin([5, 6, 7, 8, 9, 10]).astype(int)
    df["case_trend"] = df["last_week_cases"] - df["last_2_weeks_cases"]

    feature_cols = [
        "District_Encoded",
        "Month",
        "Week",
        "Avg Max Temp (°C)",
        "Avg Min Temp (°C)",
        "Total Precipitation (mm)",
        "Avg Wind Speed (km/h)",
        "last_week_cases",
        "last_2_weeks_cases",
        "temp_range",
        "avg_temp",
        "rain_temp_interaction",
        "is_monsoon",
        "cases_3week_avg",
        "case_trend",
    ]

    X = df[feature_cols].astype(float)
    y = df["Number_of_Cases"].astype(float)

    return df, X, y, feature_cols, district_encoding


def train_and_evaluate(X, y, years):
    cutoff_year = np.quantile(years, 0.8)
    train_mask = years <= cutoff_year

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[~train_mask]
    y_test = y[~train_mask]

    if len(X_test) < 50:
        split_index = int(0.8 * len(X))
        X_train = X.iloc[:split_index]
        y_train = y.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_test = y.iloc[split_index:]

    y_train_log = np.log1p(y_train)

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(X_train, y_train_log)

    y_pred_log = model.predict(X_test)
    y_pred = np.maximum(0, np.expm1(y_pred_log))

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    mape = float(np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1.0))) * 100)

    metrics = {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 3),
        "mape": round(mape, 3),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    return model, metrics


def main():
    print("Loading and preparing dengue dataset...")
    df, X, y, feature_cols, district_encoding = load_and_prepare_data()

    print(f"Prepared rows: {len(X)}")
    model, metrics = train_and_evaluate(X, y, df["Year"].values)

    bundle = {
        "model": model,
        "feature_names": feature_cols,
        "district_encoding": district_encoding,
        "target_transform": "log1p",
        "metrics": metrics,
        "trained_at": pd.Timestamp.now().isoformat(),
        "model_type": "xgb_regressor_weekly_cases_v1",
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    print(f"Saved regressor bundle: {MODEL_PATH}")
    print("Validation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
