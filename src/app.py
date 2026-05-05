from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import requests
from datetime import datetime, timedelta
import pandas as pd
import json
import sqlite3
import os
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler
from official_data import (
    get_all_latest_official_cases,
    get_latest_official_cases,
    get_official_data_summary,
    init_official_data_tables,
    official_data_is_stale,
    update_official_dengue_data,
)

app = Flask(__name__)

# Database setup for caching
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "forecast_cache.db"
DATA_PATH = BASE_DIR / "data" / "dengue_data.csv"
MODELS_DIR = BASE_DIR / "models"
EVALUATION_REPORT_PATH = BASE_DIR / "results" / "real_world_evaluation.json"
MODEL_VERSION = "real_world_validated_v1"

def init_db():
    """Initialize database for forecast caching"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS forecasts (
            district TEXT PRIMARY KEY,
            forecast_data TEXT,
            lat REAL,
            lng REAL,
            timestamp DATETIME
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS update_log (
            id INTEGER PRIMARY KEY,
            timestamp DATETIME,
            status TEXT,
            districts_updated INTEGER
        )
    ''')
    conn.commit()
    conn.close()

init_db()
init_official_data_tables(DB_PATH)

# Load real-world validated model
try:
    model = joblib.load(str(MODELS_DIR / "dengue_model_real_world_validated.pkl"))
except:
    # Fallback to enhanced/original model if validated model is not available
    try:
        model = joblib.load(str(MODELS_DIR / "dengue_model_enhanced.pkl"))
    except:
        try:
            model = joblib.load(str(MODELS_DIR / "dengue_model.pkl"))
        except:
            model = None

# Load weekly case count regressor bundle (trained model + metadata)
try:
    case_regressor_bundle = joblib.load(str(MODELS_DIR / "dengue_weekly_cases_regressor.pkl"))
except:
    case_regressor_bundle = None


def build_case_feature_context(district, month, seasonal_mean, prev_month_mean=None, next_month_mean=None):
    """Use latest official cases when available, with seasonal history as fallback."""
    prev_month_mean = float(prev_month_mean if prev_month_mean is not None else seasonal_mean)
    next_month_mean = float(next_month_mean if next_month_mean is not None else seasonal_mean)
    seasonal_mean = float(seasonal_mean)

    official_cases = get_latest_official_cases(DB_PATH, district)
    if official_cases:
        current_cases = float(official_cases.get("current_week_cases") or 0)
        previous_cases_raw = official_cases.get("previous_week_cases")
        previous_cases = (
            float(previous_cases_raw)
            if previous_cases_raw is not None
            else prev_month_mean
        )
        cases_3week_avg = (current_cases + previous_cases + seasonal_mean) / 3.0
        case_trend = current_cases - previous_cases

        public_context = {
            "source": "official",
            "source_name": official_cases.get("source"),
            "source_url": official_cases.get("source_url"),
            "report_year": official_cases.get("report_year"),
            "report_week": official_cases.get("report_week"),
            "report_period": official_cases.get("report_period"),
            "current_week_cases": int(round(current_cases)),
            "previous_week_cases": (
                int(round(previous_cases_raw)) if previous_cases_raw is not None else None
            ),
            "cumulative_cases": official_cases.get("cumulative_cases"),
            "case_trend": int(round(case_trend)) if previous_cases_raw is not None else None,
            "pulled_at": official_cases.get("pulled_at"),
        }

        return {
            "last_week_cases": current_cases,
            "last_2_weeks_cases": (current_cases + previous_cases) / 2.0,
            "cases_3week_avg": cases_3week_avg,
            "case_trend": case_trend,
            "public_context": public_context,
        }

    cases_3week_avg = (prev_month_mean + seasonal_mean + next_month_mean) / 3.0
    return {
        "last_week_cases": seasonal_mean,
        "last_2_weeks_cases": (seasonal_mean + prev_month_mean) / 2.0,
        "cases_3week_avg": cases_3week_avg,
        "case_trend": seasonal_mean - prev_month_mean,
        "public_context": {
            "source": "historical",
            "source_name": "Historical seasonal dengue profile",
            "current_week_cases": None,
            "previous_week_cases": None,
            "cumulative_cases": None,
            "seasonal_mean_cases": int(round(seasonal_mean)),
            "report_year": None,
            "report_week": None,
            "report_period": None,
            "source_url": None,
            "pulled_at": None,
        },
    }


def build_confidence_details(probabilities, prediction):
    """Convert model probabilities into user-facing confidence details."""
    probabilities = [float(p) for p in probabilities]
    sorted_probs = sorted(probabilities, reverse=True)
    confidence = int(round(probabilities[prediction] * 100))
    margin = int(round((sorted_probs[0] - sorted_probs[1]) * 100)) if len(sorted_probs) > 1 else confidence

    if confidence >= 75 and margin >= 15:
        label = "High confidence"
        level = "high"
        note = "The model strongly favors this risk level."
    elif confidence >= 55:
        label = "Medium confidence"
        level = "medium"
        note = "The model favors this risk level, but nearby alternatives remain possible."
    else:
        label = "Low confidence"
        level = "low"
        note = "Signals are mixed, so review the probability split and official case context."

    return {
        "confidence": confidence,
        "confidence_label": label,
        "confidence_level": level,
        "confidence_margin": margin,
        "confidence_note": note,
    }


def is_forecast_payload_current(forecasts):
    """Detect old cache records that do not include official data/confidence metadata."""
    if not forecasts or not isinstance(forecasts, list):
        return False

    first = forecasts[0]
    return (
        isinstance(first, dict)
        and "case_context" in first
        and "confidence_label" in first
        and "probabilities" in first
        and first.get("model_version") == MODEL_VERSION
    )


def load_real_world_evaluation_report():
    """Load saved real-world validation metrics for analytics/API display."""
    if not EVALUATION_REPORT_PATH.exists():
        return {
            "available": False,
            "message": "Run evaluate_real_world_model.py to generate real-world metrics.",
        }

    try:
        with open(EVALUATION_REPORT_PATH, "r", encoding="utf-8") as file:
            report = json.load(file)
        report["available"] = True
        return report
    except Exception as error:
        return {
            "available": False,
            "message": f"Could not load evaluation report: {error}",
        }


def _parse_forecast_date_with_year(date_text):
    """Parse UI date format like 'Tue, Mar 10' and attach current year."""
    if not date_text:
        return None
    try:
        dt = datetime.strptime(date_text, "%a, %b %d")
        return dt.replace(year=datetime.now().year)
    except Exception:
        return None


def _build_weekly_case_features(district, forecasts):
    """Build weekly feature vector from 7-day forecast for case count regression."""
    if not forecasts:
        return None, None

    first_date = _parse_forecast_date_with_year(forecasts[0].get("date"))
    month = first_date.month if first_date else datetime.now().month
    week = (first_date.day - 1) // 7 + 1 if first_date else 1

    max_temp_week = float(np.mean([f.get("max_temp", 0) for f in forecasts]))
    min_temp_week = float(np.mean([f.get("min_temp", 0) for f in forecasts]))
    rainfall_week = float(np.sum([f.get("rainfall", 0) for f in forecasts]))
    wind_week = float(np.mean([f.get("wind", 0) for f in forecasts]))

    global_mean = float(CASE_PROFILES.get("global_mean_weekly_cases", 34.75))
    seasonal_mean = float(CASE_PROFILES["district_month_mean"].get((district, month), global_mean))

    prev_month = (month - 2) % 12 + 1
    next_month = month % 12 + 1

    prev_month_mean = float(CASE_PROFILES["district_month_mean"].get((district, prev_month), seasonal_mean))
    next_month_mean = float(CASE_PROFILES["district_month_mean"].get((district, next_month), seasonal_mean))

    case_context = build_case_feature_context(
        district, month, seasonal_mean, prev_month_mean, next_month_mean
    )
    cases_3week_avg = case_context["cases_3week_avg"]
    case_trend = case_context["case_trend"]
    last_week_cases = case_context["last_week_cases"]
    last_2_weeks_cases = case_context["last_2_weeks_cases"]

    temp_range = max_temp_week - min_temp_week
    avg_temp = (max_temp_week + min_temp_week) / 2.0
    rain_temp_interaction = rainfall_week * avg_temp
    is_monsoon = 1 if month in [5, 6, 7, 8, 9, 10] else 0

    district_id = CASE_PROFILES["district_encoding"].get(district, 0)

    features = np.array([[
        district_id,
        month,
        week,
        max_temp_week,
        min_temp_week,
        rainfall_week,
        wind_week,
        last_week_cases,
        last_2_weeks_cases,
        temp_range,
        avg_temp,
        rain_temp_interaction,
        is_monsoon,
        cases_3week_avg,
        case_trend,
    ]], dtype=float)

    context = {
        "month": month,
        "seasonal_mean": seasonal_mean,
        "case_context": case_context["public_context"],
        "dominant_risk": int(max(set([f.get("risk_score", 0) for f in forecasts]), key=[f.get("risk_score", 0) for f in forecasts].count))
    }
    return features, context


def predict_weekly_cases_for_ministry(district, forecasts):
    """Predict weekly district case count for ministry planning.

    Uses direct regression model when available, with robust seasonal fallback.
    Returns dict with point estimate and uncertainty bounds.
    """
    if not forecasts:
        return {
            "weekly_predicted_cases": 0,
            "lower_bound": 0,
            "upper_bound": 0,
            "method": "no_data"
        }

    features, context = _build_weekly_case_features(district, forecasts)
    month = context["month"] if context else None
    dominant_risk = context["dominant_risk"] if context else 0
    case_context = context.get("case_context") if context else None

    seasonal_fallback = float(estimate_weekly_cases(district, dominant_risk, month))

    if case_regressor_bundle and features is not None:
        try:
            reg_model = case_regressor_bundle.get("model")
            raw_pred = float(reg_model.predict(features)[0]) if reg_model else seasonal_fallback

            if case_regressor_bundle.get("target_transform") == "log1p":
                raw_pred = float(np.expm1(raw_pred))

            raw_pred = max(0.0, raw_pred)

            seasonal_mean = float(context.get("seasonal_mean", seasonal_fallback))
            official_current_cases = None
            if case_context and case_context.get("source") == "official":
                official_current_cases = float(case_context.get("current_week_cases") or 0)
                blended_pred = 0.65 * raw_pred + 0.20 * seasonal_mean + 0.15 * official_current_cases
            else:
                blended_pred = 0.75 * raw_pred + 0.25 * seasonal_mean

            if dominant_risk == 2:
                blended_pred = max(blended_pred, seasonal_mean * 0.85)
            elif dominant_risk == 0:
                blended_pred = min(blended_pred, max(seasonal_mean * 1.10, blended_pred))

            mae = float(case_regressor_bundle.get("metrics", {}).get("mae", max(5.0, 0.25 * blended_pred)))
            lower = max(0.0, blended_pred - mae)
            upper = blended_pred + mae

            return {
                "weekly_predicted_cases": int(round(blended_pred)),
                "lower_bound": int(round(lower)),
                "upper_bound": int(round(upper)),
                "method": (
                    "official_regression_blended"
                    if official_current_cases is not None
                    else "regression_blended"
                )
            }
        except Exception:
            pass

    if case_context and case_context.get("source") == "official":
        official_current_cases = float(case_context.get("current_week_cases") or 0)
        seasonal_fallback = 0.70 * seasonal_fallback + 0.30 * official_current_cases
        fallback_method = "official_seasonal_risk_fallback"
    else:
        fallback_method = "seasonal_risk_fallback"

    spread = max(4.0, seasonal_fallback * 0.30)
    return {
        "weekly_predicted_cases": int(round(seasonal_fallback)),
        "lower_bound": int(round(max(0.0, seasonal_fallback - spread))),
        "upper_bound": int(round(seasonal_fallback + spread)),
        "method": fallback_method
    }

def load_historical_case_profiles():
    """Load historical data and build realistic weekly case estimation profiles from seasonal patterns."""
    try:
        if not DATA_PATH.exists():
            return {
                "district_encoding": {},
                "global_risk_weekly": {0: 1.08, 1: 6.18, 2: 67.48},
                "district_risk_weekly": {},
                "district_month_risk_weekly": {},
                "district_month_mean": {},
                "district_baseline": {},
                "global_mean_weekly_cases": 34.75
            }

        df = pd.read_csv(DATA_PATH)
        required_cols = {"District", "Number_of_Cases", "Month", "Year", "Week"}
        if not required_cols.issubset(df.columns):
            return {
                "district_encoding": {},
                "global_risk_weekly": {0: 1.08, 1: 6.18, 2: 67.48},
                "district_risk_weekly": {},
                "district_month_risk_weekly": {},
                "district_month_mean": {},
                "district_baseline": {},
                "global_mean_weekly_cases": 34.75
            }

        df = df[["District", "Number_of_Cases", "Month", "Year", "Week"]].copy()
        df["Number_of_Cases"] = pd.to_numeric(df["Number_of_Cases"], errors="coerce").fillna(0).clip(lower=0)
        df["Month"] = pd.to_numeric(df["Month"], errors="coerce").fillna(1).astype(int)
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(0).astype(int)
        df["Week"] = pd.to_numeric(df["Week"], errors="coerce").fillna(0).astype(int)

        # Create risk levels based on actual weekly cases
        df["Risk"] = pd.cut(df["Number_of_Cases"], bins=[-1, 3, 9, float("inf")], labels=[0, 1, 2]).astype(int)

        district_encoding = {
            district: index for index, district in enumerate(sorted(df["District"].dropna().unique()))
        }

        # USE WEEKLY AVERAGES DIRECTLY (not daily conversions)
        # Global risk-to-weekly-cases mapping
        global_risk_weekly = df.groupby("Risk")["Number_of_Cases"].mean().to_dict()
        global_risk_weekly = {
            risk: float(global_risk_weekly.get(risk, 0.0))
            for risk in [0, 1, 2]
        }

        # District-specific risk-to-weekly-cases
        district_risk_weekly = {}
        district_risk_grouped = df.groupby(["District", "Risk"])["Number_of_Cases"].mean()
        for (district, risk), weekly_mean in district_risk_grouped.items():
            district_risk_weekly[(district, int(risk))] = float(weekly_mean)

        # District + Month + Risk -> Weekly cases (SEASONAL PATTERNS)
        district_month_risk_weekly = {}
        district_month_risk_grouped = df.groupby(["District", "Month", "Risk"])["Number_of_Cases"].mean()
        for (district, month, risk), weekly_mean in district_month_risk_grouped.items():
            district_month_risk_weekly[(district, int(month), int(risk))] = float(weekly_mean)

        # District + Month long-term mean (seasonal epidemiological baseline for model features)
        # This is the historical average weekly case count for each district in each month.
        # Using this prevents the dataset tail (recent outbreak spikes) from dominating model input.
        district_month_mean = {}
        for (dist_name, mon), grp in df.groupby(["District", "Month"]):
            district_month_mean[(dist_name, int(mon))] = float(grp["Number_of_Cases"].mean())

        # Baseline features for model inputs
        district_baseline = {}
        sorted_df = df.sort_values(["District", "Year", "Week"])
        for district, group in sorted_df.groupby("District"):
            recent = group.tail(12)
            if recent.empty:
                continue

            baseline_cases = float(recent.tail(4)["Number_of_Cases"].mean())

            if len(recent) >= 8:
                previous_cases = float(recent.iloc[-8:-4]["Number_of_Cases"].mean())
            else:
                previous_cases = baseline_cases

            cases_3week_avg = float(recent.tail(3)["Number_of_Cases"].mean()) if len(recent) >= 3 else baseline_cases
            case_trend = baseline_cases - previous_cases

            district_baseline[district] = {
                "baseline_cases": baseline_cases,
                "previous_cases": previous_cases,
                "cases_3week_avg": cases_3week_avg,
                "case_trend": case_trend
            }

        return {
            "district_encoding": district_encoding,
            "global_risk_weekly": global_risk_weekly,
            "district_risk_weekly": district_risk_weekly,
            "district_month_risk_weekly": district_month_risk_weekly,
            "district_month_mean": district_month_mean,
            "district_baseline": district_baseline,
            "global_mean_weekly_cases": float(df["Number_of_Cases"].mean())
        }

    except Exception as error:
        print(f"Warning: could not load historical profiles: {error}")
        return {
            "district_encoding": {},
            "global_risk_weekly": {0: 1.08, 1: 6.18, 2: 67.48},
            "district_risk_weekly": {},
            "district_month_risk_weekly": {},
            "district_month_mean": {},
            "district_baseline": {},
            "global_mean_weekly_cases": 34.75
        }

CASE_PROFILES = load_historical_case_profiles()

def load_historical_weather_profiles():
    """Build district/month weather defaults for offline forecast fallback."""
    defaults = {
        "max_temp": 30.0,
        "min_temp": 24.0,
        "rainfall": 2.0,
        "wind": 12.0,
    }

    try:
        if not DATA_PATH.exists():
            return {"district_month": {}, "global_month": {}, "defaults": defaults}

        df = pd.read_csv(DATA_PATH)
        required_cols = {
            "District",
            "Month",
            "Avg Max Temp (°C)",
            "Avg Min Temp (°C)",
            "Total Precipitation (mm)",
            "Avg Wind Speed (km/h)",
        }
        if not required_cols.issubset(df.columns):
            return {"district_month": {}, "global_month": {}, "defaults": defaults}

        df = df[list(required_cols)].copy()
        numeric_cols = [
            "Month",
            "Avg Max Temp (°C)",
            "Avg Min Temp (°C)",
            "Total Precipitation (mm)",
            "Avg Wind Speed (km/h)",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["District", "Month"])
        df["Month"] = df["Month"].astype(int)

        district_month = {}
        for (district, month), group in df.groupby(["District", "Month"]):
            district_month[(district, int(month))] = {
                "max_temp": float(group["Avg Max Temp (°C)"].mean()),
                "min_temp": float(group["Avg Min Temp (°C)"].mean()),
                "rainfall": float(group["Total Precipitation (mm)"].mean() / 7.0),
                "wind": float(group["Avg Wind Speed (km/h)"].mean()),
            }

        global_month = {}
        for month, group in df.groupby("Month"):
            global_month[int(month)] = {
                "max_temp": float(group["Avg Max Temp (°C)"].mean()),
                "min_temp": float(group["Avg Min Temp (°C)"].mean()),
                "rainfall": float(group["Total Precipitation (mm)"].mean() / 7.0),
                "wind": float(group["Avg Wind Speed (km/h)"].mean()),
            }

        return {
            "district_month": district_month,
            "global_month": global_month,
            "defaults": defaults,
        }
    except Exception as error:
        print(f"Warning: could not load historical weather profiles: {error}")
        return {"district_month": {}, "global_month": {}, "defaults": defaults}


WEATHER_PROFILES = load_historical_weather_profiles()


def build_local_weather_forecast(district, days=7):
    """Create a 7-day forecast from historical district/month weather averages."""
    daily = {
        "time": [],
        "temperature_2m_max": [],
        "temperature_2m_min": [],
        "precipitation_sum": [],
        "windspeed_10m_max": [],
        "precipitation_probability_max": [],
        "uv_index_max": [],
    }

    today = datetime.now().date()
    for i in range(days):
        forecast_date = today + timedelta(days=i)
        month = forecast_date.month
        profile = (
            WEATHER_PROFILES["district_month"].get((district, month))
            or WEATHER_PROFILES["global_month"].get(month)
            or WEATHER_PROFILES["defaults"]
        )

        temp_wave = ((i % 3) - 1) * 0.6
        rain_factor = 0.7 + ((i * 37) % 6) * 0.12
        rainfall = max(0.0, profile["rainfall"] * rain_factor)
        precip_probability = min(95, int(25 + rainfall * 6))

        daily["time"].append(forecast_date.isoformat())
        daily["temperature_2m_max"].append(round(profile["max_temp"] + temp_wave, 1))
        daily["temperature_2m_min"].append(round(profile["min_temp"] + temp_wave * 0.5, 1))
        daily["precipitation_sum"].append(round(rainfall, 1))
        daily["windspeed_10m_max"].append(round(profile["wind"], 1))
        daily["precipitation_probability_max"].append(precip_probability)
        daily["uv_index_max"].append(9.0 if profile["max_temp"] >= 30 else 7.5)

    return {"daily": daily, "source": "historical_local_fallback"}


def get_case_feature_profile(district):
    """Return deterministic baseline case features for model input."""
    district_profile = CASE_PROFILES["district_baseline"].get(district)
    if district_profile:
        return district_profile

    fallback_weekly = CASE_PROFILES.get("global_mean_weekly_cases", 7.0)
    return {
        "baseline_cases": fallback_weekly,
        "previous_cases": fallback_weekly,
        "cases_3week_avg": fallback_weekly,
        "case_trend": 0.0
    }

def estimate_weekly_cases(district, risk_score, month=None):
    """Estimate weekly cases using seasonal patterns from the same month/district/risk in previous years."""
    risk = int(risk_score)

    # Priority 1: Use district + month + risk (most specific seasonal pattern)
    if month is not None:
        month_key = (district, int(month), risk)
        if month_key in CASE_PROFILES["district_month_risk_weekly"]:
            return max(0.0, CASE_PROFILES["district_month_risk_weekly"][month_key])

    # Priority 2: Use district + risk (district-specific, all months)
    district_key = (district, risk)
    if district_key in CASE_PROFILES["district_risk_weekly"]:
        return max(0.0, CASE_PROFILES["district_risk_weekly"][district_key])

    # Priority 3: Global risk average (fallback)
    return max(0.0, CASE_PROFILES["global_risk_weekly"].get(risk, 0.0))

def get_traveler_advice(risk_level):
    """Return actionable travel safety tips based on dengue risk level."""
    if risk_level == 2:
        return {
            "level": "High",
            "color": "#dc3545",
            "headline": "High precautions strongly advised",
            "tips": [
                "Apply DEET or Picaridin repellent every 3–4 hours",
                "Wear long sleeves and long pants — especially at dawn and dusk",
                "Stay in air-conditioned or well-screened accommodation",
                "See a doctor immediately if fever develops within 2 weeks of travel",
                "Eliminate any standing water near where you stay"
            ]
        }
    elif risk_level == 1:
        return {
            "level": "Medium",
            "color": "#ffc107",
            "headline": "Standard mosquito precautions advised",
            "tips": [
                "Use mosquito repellent when outdoors",
                "Wear protective clothing at dawn and dusk",
                "Watch for symptoms: fever, headache, rash, joint or muscle pain",
                "Check that accommodation has proper window screens or nets"
            ]
        }
    else:
        return {
            "level": "Low",
            "color": "#28a745",
            "headline": "Low risk — basic awareness",
            "tips": [
                "Standard repellent use advisable for outdoor activities at dusk/dawn",
                "Monitor local health advisories during your stay"
            ]
        }


def compute_weekly_summary(forecasts, district):
    """Compute a weekly travel risk summary from 7-day forecast data."""
    if not forecasts:
        return None

    risk_counts = {0: 0, 1: 0, 2: 0}
    for f in forecasts:
        risk_counts[f["risk_score"]] += 1

    # Overall trip risk = highest risk level encountered
    if risk_counts[2] > 0:
        overall_risk = 2
        overall_label = "High"
        overall_color = "#dc3545"
    elif risk_counts[1] > 0:
        overall_risk = 1
        overall_label = "Medium"
        overall_color = "#ffc107"
    else:
        overall_risk = 0
        overall_label = "Low"
        overall_color = "#28a745"

    avg_temp = sum(((f["max_temp"] + f["min_temp"]) / 2) for f in forecasts) / 7
    total_rain = sum(f["rainfall"] for f in forecasts)
    case_context = forecasts[0].get("case_context") if forecasts and isinstance(forecasts[0], dict) else None

    if overall_risk == 2:
        if risk_counts[2] >= 5:
            travel_verdict = "Avoid non-essential travel — sustained high risk all week"
        else:
            travel_verdict = "Travel with full precautions — high-risk days present"
    elif overall_risk == 1:
        travel_verdict = "Take precautions — moderate dengue risk this week"
    else:
        travel_verdict = "Relatively safe period — standard precautions sufficient"

    return {
        "overall_risk": overall_risk,
        "overall_label": overall_label,
        "overall_color": overall_color,
        "high_days": risk_counts[2],
        "medium_days": risk_counts[1],
        "low_days": risk_counts[0],
        "avg_temp": round(avg_temp, 1),
        "total_rainfall": round(total_rain, 1),
        "travel_verdict": travel_verdict,
        "case_context": case_context,
        "district": district
    }


def extract_month_from_forecast_date(date_value):
    """Extract month from dashboard forecast date format like 'Tue, Mar 10'."""
    if not date_value:
        return None

    try:
        return datetime.strptime(date_value, "%a, %b %d").month
    except Exception:
        return None

# Districts with their correct coordinates (Sri Lanka) — names match dengue_data.csv exactly
DISTRICTS = {
    "Ampara":        {"lat": 7.2906,  "lng": 81.6724},
    "Anuradhapura":  {"lat": 8.3350,  "lng": 80.4050},
    "Badulla":       {"lat": 6.9897,  "lng": 81.0561},
    "Batticaloa":    {"lat": 7.7172,  "lng": 81.6924},
    "Colombo":       {"lat": 6.9271,  "lng": 79.8612},
    "Galle":         {"lat": 6.0535,  "lng": 80.2207},
    "Gampaha":       {"lat": 7.0890,  "lng": 80.0167},
    "Hambantota":    {"lat": 6.1240,  "lng": 81.1198},
    "Jaffna":        {"lat": 9.6615,  "lng": 80.0255},
    "Kalutara":      {"lat": 6.5844,  "lng": 79.9600},
    "Kandy":         {"lat": 7.2906,  "lng": 80.6337},
    "Kegalle":       {"lat": 7.2530,  "lng": 80.3550},
    "Kilinochchi":   {"lat": 9.3947,  "lng": 80.3930},
    "Kurunegala":    {"lat": 7.4833,  "lng": 80.3631},
    "Mannar":        {"lat": 8.9800,  "lng": 79.9029},
    "Matale":        {"lat": 7.4688,  "lng": 80.6233},
    "Matara":        {"lat": 5.9490,  "lng": 80.5540},
    "Monaragala":    {"lat": 6.8730,  "lng": 81.3483},
    "Mullaitivu":    {"lat": 9.2683,  "lng": 80.8122},
    "NuwaraEliya":   {"lat": 6.9497,  "lng": 80.7891},
    "Polonnaruwa":   {"lat": 7.9408,  "lng": 81.0111},
    "Puttalam":      {"lat": 8.0314,  "lng": 79.8278},
    "Ratnapura":     {"lat": 6.6833,  "lng": 80.4000},
    "Trincomalee":   {"lat": 8.5711,  "lng": 81.2348},
    "Vavuniya":      {"lat": 8.7500,  "lng": 80.4970},
}

# Home page - Landing page
@app.route('/')
def home():
    return render_template("landing.html")

# Prediction module home page
@app.route('/prediction-home')
def prediction_home():
    return render_template("prediction_home.html")

# Dashboard with forecast model
@app.route('/dashboard')
def dashboard():
    return render_template("index.html", districts=json.dumps(DISTRICTS))

# Analytics page
@app.route('/analytics')
def analytics():
    return render_template("analytics.html")

# Ministry Dashboard - All districts overview
@app.route('/ministry-dashboard')
def ministry_dashboard():
    return render_template("ministry_dashboard.html")

# API to get cache statistics
@app.route('/api/cache-stats', methods=['GET'])
def cache_stats():
    """Return cache statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get cached districts count
        c.execute('SELECT COUNT(*) FROM forecasts')
        cached_count = c.fetchone()[0]
        
        # Get last update time
        c.execute('SELECT timestamp FROM update_log ORDER BY timestamp DESC LIMIT 1')
        last_update = c.fetchone()
        
        # Get update history (last 7 days)
        c.execute('''
            SELECT timestamp, status, districts_updated 
            FROM update_log 
            ORDER BY timestamp DESC LIMIT 7
        ''')
        history = c.fetchall()
        conn.close()
        
        return jsonify({
            "cached_districts": cached_count,
            "total_districts": len(DISTRICTS),
            "last_update": last_update[0] if last_update else None,
            "official_data": get_official_data_summary(DB_PATH),
            "update_history": [
                {"time": h[0], "status": h[1], "districts": h[2]} 
                for h in history
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Manual trigger for forecast update (for testing)
@app.route('/api/update-now', methods=['POST'])
def update_now():
    """Manually trigger forecast update"""
    update_all_forecasts()
    return jsonify({"success": True, "message": "Forecast update triggered"})


@app.route('/api/update-official-data', methods=['POST'])
def update_official_data_now():
    """Manually import latest official dengue case reports."""
    result = update_official_dengue_data(DB_PATH, max_reports=1)
    return jsonify(result), (200 if result.get("success") else 400)


@app.route('/api/official-dengue-data', methods=['GET'])
def official_dengue_data():
    """Return the latest imported official district dengue case counts."""
    try:
        return jsonify({
            "success": True,
            "summary": get_official_data_summary(DB_PATH),
            "districts": get_all_latest_official_cases(DB_PATH),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/model-evaluation', methods=['GET'])
def model_evaluation():
    """Return real-world backtest metrics generated from newer unseen data."""
    report = load_real_world_evaluation_report()
    return jsonify({
        "success": bool(report.get("available")),
        "evaluation": report,
    }), (200 if report.get("available") else 404)

def build_ministry_forecast_item(district, coords, forecasts, timestamp, source="cache"):
    """Build one district row for the ministry dashboard."""
    if not forecasts:
        return {
            "district": district,
            "lat": coords['lat'],
            "lng": coords['lng'],
            "weekly_predicted_cases": 0,
            "avg_risk_level": 0,
            "max_risk_level": 0,
            "risk_category": "No Data",
            "trend": "unknown",
            "peak_day": 0,
            "daily_forecasts": [],
            "last_updated": timestamp,
            "case_context": None,
            "source": source
        }

    total_risk_score = sum(f['risk_score'] for f in forecasts)
    avg_risk = total_risk_score / len(forecasts)
    max_risk = max(f['risk_score'] for f in forecasts)

    weekly_case_result = predict_weekly_cases_for_ministry(district, forecasts)
    weekly_cases = weekly_case_result["weekly_predicted_cases"]

    first_half_avg = np.mean([forecasts[i]['risk_score'] for i in range(3)])
    second_half_avg = np.mean([forecasts[i]['risk_score'] for i in range(4, 7)])

    if second_half_avg > first_half_avg + 0.3:
        trend = "increasing"
    elif second_half_avg < first_half_avg - 0.3:
        trend = "decreasing"
    else:
        trend = "stable"

    peak_day = max(range(len(forecasts)), key=lambda i: forecasts[i]['risk_score']) + 1

    return {
        "district": district,
        "lat": coords['lat'],
        "lng": coords['lng'],
        "weekly_predicted_cases": int(round(weekly_cases)),
        "weekly_case_range": {
            "lower": weekly_case_result.get("lower_bound", int(round(weekly_cases))),
            "upper": weekly_case_result.get("upper_bound", int(round(weekly_cases)))
        },
        "prediction_method": weekly_case_result.get("method", "unknown"),
        "avg_risk_level": round(avg_risk, 2),
        "max_risk_level": int(max_risk),
        "risk_category": "Low" if max_risk == 0 else ("Medium" if max_risk == 1 else "High"),
        "trend": trend,
        "peak_day": peak_day,
        "daily_forecasts": forecasts,
        "last_updated": timestamp,
        "case_context": forecasts[0].get("case_context") if forecasts else None,
        "source": source
    }

# API to get all districts forecasts for Ministry Dashboard
@app.route('/api/all-districts-forecast', methods=['GET'])
def all_districts_forecast():
    """Get forecast predictions for all districts"""
    try:
        all_forecasts = []
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        for district, coords in DISTRICTS.items():
            # Try to get cached forecast
            c.execute('SELECT forecast_data, timestamp FROM forecasts WHERE district = ?', (district,))
            result = c.fetchone()
            
            if result:
                forecast_data, timestamp = result
                forecasts = json.loads(forecast_data)
                try:
                    cache_time = datetime.fromisoformat(timestamp)
                except Exception:
                    cache_time = datetime.min

                if is_forecast_payload_current(forecasts) and datetime.now() - cache_time < timedelta(hours=12):
                    all_forecasts.append(build_ministry_forecast_item(
                        district, coords, forecasts, timestamp, source="cache"
                    ))
                    continue

                weather_data = build_local_weather_forecast(district)
                forecasts = generate_forecasts(weather_data, district)
                timestamp = datetime.now().isoformat()
                if forecasts:
                    cache_forecast(district, forecasts, coords['lat'], coords['lng'])
                all_forecasts.append(build_ministry_forecast_item(
                    district, coords, forecasts, timestamp, source="refreshed-cache"
                ))
            else:
                weather_data = build_local_weather_forecast(district)
                forecasts = generate_forecasts(weather_data, district)
                timestamp = datetime.now().isoformat()
                if forecasts:
                    cache_forecast(district, forecasts, coords['lat'], coords['lng'])
                all_forecasts.append(build_ministry_forecast_item(
                    district, coords, forecasts, timestamp, source="historical-local-fallback"
                ))
        
        conn.close()
        
        # Calculate national summary
        total_cases = sum(f['weekly_predicted_cases'] for f in all_forecasts if f['weekly_predicted_cases'] > 0)
        high_risk = [f for f in all_forecasts if f['risk_category'] == 'High']
        medium_risk = [f for f in all_forecasts if f['risk_category'] == 'Medium']
        low_risk = [f for f in all_forecasts if f['risk_category'] == 'Low']
        
        # Priority districts (top 5 by predicted cases)
        priority = sorted(all_forecasts, key=lambda x: x['weekly_predicted_cases'], reverse=True)[:5]
        
        return jsonify({
            "success": True,
            "summary": {
                "total_predicted_cases": total_cases,
                "high_risk_count": len(high_risk),
                "medium_risk_count": len(medium_risk),
                "low_risk_count": len(low_risk),
                "priority_districts": [p['district'] for p in priority],
                "case_prediction_model": (
                    case_regressor_bundle.get("model_type", "seasonal_risk_fallback")
                    if case_regressor_bundle else "seasonal_risk_fallback"
                ),
                "model_mae": (
                    case_regressor_bundle.get("metrics", {}).get("mae")
                    if case_regressor_bundle else None
                ),
                "official_data": get_official_data_summary(DB_PATH)
            },
            "districts": all_forecasts,
            "generated_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# API to get weather forecast and predict dengue risk
@app.route('/api/forecast', methods=['POST'])
def get_forecast():
    try:
        data = request.json
        lat = float(data['lat'])
        lng = float(data['lng'])
        district = data.get('district', 'Unknown')

        # Try to get cached forecast
        cached = get_cached_forecast(district)
        if cached:
            weekly_summary = compute_weekly_summary(cached, district)
            return jsonify({
                "success": True,
                "district": district,
                "forecasts": cached,
                "weekly_summary": weekly_summary,
                "cached": True,
                "cached_at": get_cache_timestamp(district),
                "case_context": weekly_summary.get("case_context") if weekly_summary else None,
                "official_data_summary": get_official_data_summary(DB_PATH)
            })

        # If no cache, fetch from API
        return fetch_and_cache_forecast(lat, lng, district)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

def get_cached_forecast(district):
    """Retrieve forecast from cache if it exists and is current"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT forecast_data, timestamp FROM forecasts WHERE district = ?', (district,))
        result = c.fetchone()
        conn.close()
        
        if result:
            forecast_data, timestamp = result
            # Check if cache is less than 12 hours old
            cache_time = datetime.fromisoformat(timestamp)
            forecasts = json.loads(forecast_data)
            if (
                datetime.now() - cache_time < timedelta(hours=12)
                and is_forecast_payload_current(forecasts)
            ):
                return forecasts
        return None
    except:
        return None

def get_cache_timestamp(district):
    """Get when the forecast was cached"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT timestamp FROM forecasts WHERE district = ?', (district,))
        result = c.fetchone()
        conn.close()
        return result[0] if result else None
    except:
        return None

def fetch_open_meteo_weather(lat, lng, timeout=10):
    """Fetch weather data without inheriting broken system proxy settings."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,precipitation_probability_max,uv_index_max&timezone=auto&forecast_days=7"
    session = requests.Session()
    session.trust_env = False
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    weather_data = response.json()
    if not weather_data.get("daily"):
        raise ValueError("Weather API returned no daily forecast data")
    return weather_data


def fetch_and_cache_forecast(lat, lng, district):
    """Fetch forecast from API and cache it, falling back to local history."""
    try:
        source = "open-meteo"
        fallback_reason = None
        try:
            weather_data = fetch_open_meteo_weather(lat, lng, timeout=10)
        except Exception as error:
            fallback_reason = str(error)
            source = "historical-local-fallback"
            weather_data = build_local_weather_forecast(district)

        forecasts = generate_forecasts(weather_data, district)
        if not forecasts:
            raise ValueError("Could not generate forecasts from available weather data")
        
        # Cache the forecast
        cache_forecast(district, forecasts, lat, lng)

        weekly_summary = compute_weekly_summary(forecasts, district)
        
        return jsonify({
            "success": True,
            "district": district,
            "forecasts": forecasts,
            "weekly_summary": weekly_summary,
            "cached": False,
            "source": source,
            "case_context": weekly_summary.get("case_context") if weekly_summary else None,
            "official_data_summary": get_official_data_summary(DB_PATH),
            "fallback_reason": fallback_reason
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

def generate_forecasts(weather_data, district):
    """Generate 7-day dengue risk forecasts using seasonal epidemiological baselines.

    Key design decisions:
    - Epidemiological features use district's long-term MONTHLY MEAN (not dataset tail).
      This prevents recent outbreak spikes from locking every district into High risk
      regardless of current weather conditions.
    - Plateau smoothing removed: each day reflects the genuine model prediction for
      that day's weather + seasonal context, allowing real day-to-day variation.
    - Precipitation probability and UV index are included for traveler context.
    """
    forecasts = []
    daily = weather_data.get('daily', {})

    if not daily:
        return forecasts

    risk_levels = {0: "Low", 1: "Medium", 2: "High"}
    risk_colors = {0: "#28a745", 1: "#ffc107", 2: "#dc3545"}
    global_mean = CASE_PROFILES.get("global_mean_weekly_cases", 34.75)
    district_id = CASE_PROFILES["district_encoding"].get(district, 0)
    shared_case_context = None

    for i in range(7):
        date = datetime.fromisoformat(daily['time'][i])
        month = date.month
        week = (date.day - 1) // 7 + 1

        max_temp = float(daily['temperature_2m_max'][i] or 28)
        min_temp = float(daily['temperature_2m_min'][i] or 22)
        rainfall = float(daily['precipitation_sum'][i] or 0)
        wind = float(daily['windspeed_10m_max'][i] or 10)

        # Extra display-only fields (may not exist in older cached weather)
        precip_prob_list = daily.get('precipitation_probability_max') or []
        uv_list = daily.get('uv_index_max') or []
        precip_prob = int(precip_prob_list[i]) if i < len(precip_prob_list) and precip_prob_list[i] is not None else 0
        uv_index = float(uv_list[i]) if i < len(uv_list) and uv_list[i] is not None else 5.0

        # ── SEASONAL EPIDEMIOLOGICAL BASELINE ──────────────────────────────────
        # Use the district's historical long-term monthly average as the
        # "recent cases" context fed to XGBoost. This is the best estimate of
        # "what's typically happening in this district this month" and allows
        # the weather features to have meaningful influence on predictions.
        seasonal_mean = CASE_PROFILES["district_month_mean"].get(
            (district, month), global_mean
        )

        # Previous month mean gives a trend direction signal
        prev_month = (month - 2) % 12 + 1
        prev_month_mean = CASE_PROFILES["district_month_mean"].get(
            (district, prev_month), seasonal_mean
        )

        # 3-week average spans adjacent months for smoother seasonality
        next_month = month % 12 + 1
        next_month_mean = CASE_PROFILES["district_month_mean"].get(
            (district, next_month), seasonal_mean
        )
        case_feature_context = build_case_feature_context(
            district, month, seasonal_mean, prev_month_mean, next_month_mean
        )
        shared_case_context = case_feature_context["public_context"]
        cases_3week_avg = case_feature_context["cases_3week_avg"]
        seasonal_case_trend = case_feature_context["case_trend"]
        last_week_cases = case_feature_context["last_week_cases"]
        last_2_weeks_cases = case_feature_context["last_2_weeks_cases"]

        # ── DERIVED FEATURES ────────────────────────────────────────────────────
        temp_range = max_temp - min_temp
        avg_temp = (max_temp + min_temp) / 2.0
        rain_temp_interaction = rainfall * avg_temp
        is_monsoon = 1 if month in [5, 6, 7, 8, 9, 10] else 0

        features = np.array([[
            district_id, month, week,
            max_temp, min_temp, rainfall, wind,
            last_week_cases, last_2_weeks_cases,
            temp_range, avg_temp, rain_temp_interaction,
            is_monsoon, cases_3week_avg, seasonal_case_trend
        ]])

        if model:
            prediction = int(model.predict(features)[0])
            probabilities = model.predict_proba(features)[0]
        else:
            # Fallback heuristic when model is unavailable
            raw = (avg_temp - 24) / 3.0 + rainfall / 30.0 - wind / 40.0
            prediction = max(0, min(2, int(round(raw))))
            probs = [0.6, 0.3, 0.1]
            if prediction == 1:
                probs = [0.2, 0.6, 0.2]
            elif prediction == 2:
                probs = [0.05, 0.2, 0.75]
            probabilities = probs

        explanation = generate_explanation(
            prediction,
            max_temp,
            min_temp,
            rainfall,
            wind,
            month,
            seasonal_mean,
            shared_case_context,
        )
        advice = get_traveler_advice(prediction)
        confidence_details = build_confidence_details(probabilities, prediction)

        forecasts.append({
            "date": date.strftime("%a, %b %d"),
            "day": i + 1,
            "max_temp": round(max_temp, 1),
            "min_temp": round(min_temp, 1),
            "rainfall": round(rainfall, 1),
            "wind": round(wind, 1),
            "precip_probability": precip_prob,
            "uv_index": round(uv_index, 1),
            "risk_level": risk_levels[prediction],
            "risk_color": risk_colors[prediction],
            "risk_score": prediction,
            "model_version": MODEL_VERSION,
            "confidence": confidence_details["confidence"],
            "confidence_label": confidence_details["confidence_label"],
            "confidence_level": confidence_details["confidence_level"],
            "confidence_margin": confidence_details["confidence_margin"],
            "confidence_note": confidence_details["confidence_note"],
            "probabilities": {
                "low": int(probabilities[0] * 100),
                "medium": int(probabilities[1] * 100),
                "high": int(probabilities[2] * 100)
            },
            "case_context": shared_case_context,
            "explanation": explanation,
            "traveler_advice": advice
        })

    return forecasts

def generate_explanation(
    risk_level,
    max_temp,
    min_temp,
    rainfall,
    wind,
    month,
    seasonal_mean=34.75,
    case_context=None,
):
    """Generate an honest multi-factor explanation combining epidemiology and weather.

    Args:
        risk_level: 0/1/2 predicted by the model
        seasonal_mean: historical average weekly cases for this district+month.
        case_context: latest official case context when public WER data is loaded.
    """
    factors = []
    avg_temp = (max_temp + min_temp) / 2

    if case_context and case_context.get("source") == "official":
        current_cases = int(case_context.get("current_week_cases") or 0)
        previous_cases = case_context.get("previous_week_cases")
        report_week = case_context.get("report_week")
        report_year = case_context.get("report_year")

        if current_cases > 60:
            official_impact = "increases"
            official_desc = "Latest official case count is in a high transmission range"
        elif current_cases > 20:
            official_impact = "moderate"
            official_desc = "Latest official case count shows moderate active transmission"
        else:
            official_impact = "decreases"
            official_desc = "Latest official case count is currently low"

        if previous_cases is not None:
            trend = current_cases - int(previous_cases)
            if trend > 0:
                official_desc += f"; cases increased by {trend} from the previous imported week"
            elif trend < 0:
                official_desc += f"; cases decreased by {abs(trend)} from the previous imported week"
            else:
                official_desc += "; cases are stable against the previous imported week"

        factors.append({
            "factor": "Official Cases",
            "value": f"{current_cases} cases",
            "impact": official_impact,
            "icon": "CASE",
            "description": f"{official_desc} (WER week {report_week}, {report_year})"
        })

    # ── 1. SEASONAL EPIDEMIOLOGICAL CONTEXT (primary model driver) ───────────
    if seasonal_mean > 60:
        epi_impact = "increases"
        epi_desc = f"Historical avg {int(seasonal_mean)} cases/week — high-transmission season"
    elif seasonal_mean > 20:
        epi_impact = "moderate"
        epi_desc = f"Historical avg {int(seasonal_mean)} cases/week — moderate transmission season"
    else:
        epi_impact = "decreases"
        epi_desc = f"Historical avg {int(seasonal_mean)} cases/week — low-transmission season"
    factors.append({
        "factor": "Seasonal History",
        "value": f"{int(seasonal_mean)} cases/wk avg",
        "impact": epi_impact,
        "icon": "📊",
        "description": epi_desc
    })

    # ── 2. TEMPERATURE ────────────────────────────────────────────────────────
    if 26 <= avg_temp <= 30:
        factors.append({
            "factor": "Optimal Mosquito Temp",
            "value": f"{avg_temp:.1f}°C",
            "impact": "increases",
            "icon": "🌡️",
            "description": "26–30°C is ideal for Aedes aegypti breeding & biting"
        })
    elif avg_temp > 30:
        factors.append({
            "factor": "Very High Temperature",
            "value": f"{avg_temp:.1f}°C",
            "impact": "moderate",
            "icon": "🌡️",
            "description": "Above 30°C slightly slows mosquito development"
        })
    elif avg_temp < 22:
        factors.append({
            "factor": "Cool Temperature",
            "value": f"{avg_temp:.1f}°C",
            "impact": "decreases",
            "icon": "❄️",
            "description": "Below 22°C significantly reduces mosquito activity"
        })
    elif avg_temp < 26:
        factors.append({
            "factor": "Mild Temperature",
            "value": f"{avg_temp:.1f}°C",
            "impact": "moderate",
            "icon": "🌤️",
            "description": "Below optimal Aedes breeding temperature"
        })

    # ── 3. RAINFALL ────────────────────────────────────────────────────────────
    if rainfall > 50:
        factors.append({
            "factor": "Heavy Rainfall",
            "value": f"{rainfall:.1f}mm",
            "impact": "increases",
            "icon": "💧",
            "description": "Creates abundant standing water and breeding sites"
        })
    elif rainfall >= 10:
        factors.append({
            "factor": "Moderate Rainfall",
            "value": f"{rainfall:.1f}mm",
            "impact": "moderate",
            "icon": "🌧️",
            "description": "May create some mosquito breeding opportunities"
        })
    elif rainfall < 3:
        factors.append({
            "factor": "Dry Conditions",
            "value": f"{rainfall:.1f}mm",
            "impact": "decreases",
            "icon": "☀️",
            "description": "Minimal standing water limits breeding sites"
        })

    # ── 4. WIND ────────────────────────────────────────────────────────────────
    if wind > 20:
        factors.append({
            "factor": "Strong Winds",
            "value": f"{wind:.1f}km/h",
            "impact": "decreases",
            "icon": "💨",
            "description": "High winds disrupt mosquito flight and dispersal"
        })

    # ── 5. MONSOON SEASON ─────────────────────────────────────────────────────
    if month in [5, 6, 10, 11]:
        factors.append({
            "factor": "Monsoon Season",
            "value": "Active",
            "impact": "increases",
            "icon": "🌊",
            "description": "Peak dengue transmission season in Sri Lanka"
        })

    # ── SUMMARY ────────────────────────────────────────────────────────────────
    if risk_level == 2:
        if seasonal_mean > 50:
            summary = "High-risk season with weather conditions supporting transmission"
        else:
            summary = "Current weather conditions support high dengue risk"
    elif risk_level == 1:
        summary = "Mixed conditions present — moderate dengue transmission risk"
    else:
        summary = "Conditions suggest reduced dengue transmission risk this period"

    return {
        "summary": summary,
        "factors": factors
    }

def cache_forecast(district, forecast_data, lat, lng):
    """Store forecast in cache"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO forecasts (district, forecast_data, lat, lng, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (district, json.dumps(forecast_data), lat, lng, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Cache error: {e}")

def update_all_forecasts():
    """Scheduled job to update all district forecasts daily"""
    print(f"[{datetime.now()}] Starting daily forecast update...")
    updated = 0
    try:
        if official_data_is_stale(DB_PATH, max_age_hours=18):
            official_result = update_official_dengue_data(DB_PATH, max_reports=1)
            print(f"Official dengue data update: {official_result}")

        for district, coords in DISTRICTS.items():
            try:
                try:
                    weather_data = fetch_open_meteo_weather(coords['lat'], coords['lng'], timeout=5)
                except Exception as weather_error:
                    print(f"Using local weather fallback for {district}: {weather_error}")
                    weather_data = build_local_weather_forecast(district)

                forecasts = generate_forecasts(weather_data, district)
                if not forecasts:
                    raise ValueError("No forecasts generated")
                cache_forecast(district, forecasts, coords['lat'], coords['lng'])
                updated += 1
            except Exception as e:
                print(f"Error updating {district}: {e}")
        
        # Log the update
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT INTO update_log (timestamp, status, districts_updated)
            VALUES (?, ?, ?)
        ''', (datetime.now().isoformat(), 'success', updated))
        conn.commit()
        conn.close()
        
        print(f"[{datetime.now()}] Daily update complete. Updated {updated} districts.")
    except Exception as e:
        print(f"Daily update error: {e}")

# Initialize scheduler for daily updates
scheduler = BackgroundScheduler()
# Update official case data before forecasts so model features use latest imported cases.
scheduler.add_job(
    lambda: update_official_dengue_data(DB_PATH, max_reports=1),
    'cron',
    hour=1,
    minute=30,
    id='daily_official_dengue_update'
)
# Update forecasts daily at 2 AM
scheduler.add_job(update_all_forecasts, 'cron', hour=2, minute=0, id='daily_forecast_update')
scheduler.start()

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode, host='127.0.0.1', port=5000)
