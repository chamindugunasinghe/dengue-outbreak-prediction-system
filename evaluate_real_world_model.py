import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
TRAIN_DATA_PATH = BASE_DIR / "data" / "dengue_data.csv"
TEST_DATA_PATH = BASE_DIR / "data" / "test_2025_2026_real.csv"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
REPORT_PATH = RESULTS_DIR / "real_world_evaluation.json"
PREDICTIONS_PATH = RESULTS_DIR / "real_world_predictions_2025_2026.csv"
MODEL_PATH = MODELS_DIR / "dengue_model_real_world_validated.pkl"

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}
RISK_THRESHOLDS = {
    "low": "0-3 cases/week",
    "medium": "4-9 cases/week",
    "high": "10+ cases/week",
}


FEATURE_COLS = [
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


def risk_from_cases(series):
    return pd.cut(
        series,
        bins=[-1, 3, 9, float("inf")],
        labels=[0, 1, 2],
    ).astype(int)


def load_data():
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    train_df["split"] = "train"
    test_df["split"] = "test"

    return train_df, test_df


def prepare_features(train_df, test_df):
    required_cols = [
        "District",
        "Number_of_Cases",
        "Year",
        "Week",
        "Month",
        "Avg Max Temp (°C)",
        "Avg Min Temp (°C)",
        "Total Precipitation (mm)",
        "Avg Wind Speed (km/h)",
    ]
    missing = [
        col
        for col in required_cols
        if col not in train_df.columns or col not in test_df.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    train_df = train_df.copy()
    test_df = test_df.copy()

    for df in [train_df, test_df]:
        for col in required_cols:
            if col == "District":
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")

    train_df = train_df.dropna(subset=required_cols)
    test_df = test_df.dropna(subset=required_cols)

    districts = sorted(train_df["District"].dropna().unique())
    district_encoding = {district: idx for idx, district in enumerate(districts)}

    unknown_districts = sorted(set(test_df["District"]) - set(district_encoding))
    if unknown_districts:
        raise ValueError(f"Test data has unseen districts: {unknown_districts}")

    train_df["District_Encoded"] = train_df["District"].map(district_encoding)
    test_df["District_Encoded"] = test_df["District"].map(district_encoding)

    train_month_mean = (
        train_df.groupby(["District", "Month"])["Number_of_Cases"].mean().to_dict()
    )
    global_mean = float(train_df["Number_of_Cases"].mean())

    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = combined.sort_values(["District", "Year", "Week"]).reset_index(drop=True)

    combined["last_week_cases"] = combined.groupby("District")["Number_of_Cases"].shift(1)
    combined["last_2_weeks_cases"] = combined.groupby("District")["Number_of_Cases"].shift(2)
    combined["cases_3week_avg"] = (
        combined.groupby("District")["Number_of_Cases"]
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean().shift(1))
    )

    combined["month_fallback"] = combined.apply(
        lambda row: train_month_mean.get(
            (row["District"], int(row["Month"])), global_mean
        ),
        axis=1,
    )
    combined["last_week_cases"] = combined["last_week_cases"].fillna(
        combined["month_fallback"]
    )
    combined["last_2_weeks_cases"] = combined["last_2_weeks_cases"].fillna(
        combined["last_week_cases"]
    )
    combined["cases_3week_avg"] = combined["cases_3week_avg"].fillna(
        (combined["last_week_cases"] + combined["last_2_weeks_cases"]) / 2.0
    )

    combined["temp_range"] = combined["Avg Max Temp (°C)"] - combined["Avg Min Temp (°C)"]
    combined["avg_temp"] = (
        combined["Avg Max Temp (°C)"] + combined["Avg Min Temp (°C)"]
    ) / 2.0
    combined["rain_temp_interaction"] = (
        combined["Total Precipitation (mm)"] * combined["avg_temp"]
    )
    combined["is_monsoon"] = combined["Month"].isin([5, 6, 7, 8, 9, 10]).astype(int)
    combined["case_trend"] = combined["last_week_cases"] - combined["last_2_weeks_cases"]
    combined["Risk"] = risk_from_cases(combined["Number_of_Cases"])

    train_ready = combined[combined["split"] == "train"].copy()
    test_ready = combined[combined["split"] == "test"].copy()

    return train_ready, test_ready, district_encoding


def train_model(train_df):
    X_train = train_df[FEATURE_COLS].astype(float)
    y_train = train_df["Risk"].astype(int)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=450,
        learning_rate=0.045,
        max_depth=6,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def build_metrics(test_df, predictions, probabilities):
    y_true = test_df["Risk"].astype(int).to_numpy()
    y_pred = np.asarray(predictions).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Low", "Medium", "High"],
        output_dict=True,
        zero_division=0,
    )

    by_class = {}
    for idx, label in RISK_LABELS.items():
        by_class[label.lower()] = {
            "precision": round(float(precision[idx]), 4),
            "recall": round(float(recall[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(support[idx]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    confidence = probabilities[np.arange(len(y_pred)), y_pred]

    test_results = test_df.copy()
    test_results["Predicted_Risk"] = y_pred
    test_results["Correct"] = y_true == y_pred
    district_metrics = []
    for district, group in test_results.groupby("District"):
        district_metrics.append(
            {
                "district": district,
                "accuracy": round(float(group["Correct"].mean()), 4),
                "weeks": int(len(group)),
                "actual_high_weeks": int((group["Risk"] == 2).sum()),
                "missed_high_weeks": int(
                    ((group["Risk"] == 2) & (group["Predicted_Risk"] != 2)).sum()
                ),
            }
        )
    district_metrics.sort(key=lambda item: (item["missed_high_weeks"], -item["accuracy"]), reverse=True)

    high_misses = test_results[
        (test_results["Risk"] == 2) & (test_results["Predicted_Risk"] != 2)
    ].copy()
    high_misses = high_misses.sort_values(
        ["Number_of_Cases", "Year", "Week"],
        ascending=[False, True, True],
    )
    top_missed_high = [
        {
            "district": row["District"],
            "year": int(row["Year"]),
            "week": int(row["Week"]),
            "actual_cases": int(row["Number_of_Cases"]),
            "predicted_risk": RISK_LABELS[int(row["Predicted_Risk"])],
        }
        for _, row in high_misses.head(10).iterrows()
    ]

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "average_confidence": round(float(np.mean(confidence)), 4),
        "average_confidence_when_correct": round(
            float(np.mean(confidence[y_true == y_pred])) if np.any(y_true == y_pred) else 0.0,
            4,
        ),
        "average_confidence_when_wrong": round(
            float(np.mean(confidence[y_true != y_pred])) if np.any(y_true != y_pred) else 0.0,
            4,
        ),
        "high_risk_recall": by_class["high"]["recall"],
        "high_risk_precision": by_class["high"]["precision"],
        "by_class": by_class,
        "confusion_matrix": {
            "labels": ["Low", "Medium", "High"],
            "matrix": cm.astype(int).tolist(),
        },
        "district_metrics": district_metrics,
        "top_missed_high_risk_weeks": top_missed_high,
    }


def save_predictions(test_df, predictions, probabilities):
    output = test_df[
        [
            "District",
            "Number_of_Cases",
            "Week_Start_Date",
            "Year",
            "Week",
            "Week_End_Date",
            "Risk",
        ]
    ].copy()
    output["Actual_Risk_Label"] = output["Risk"].map(RISK_LABELS)
    output["Predicted_Risk"] = predictions
    output["Predicted_Risk_Label"] = output["Predicted_Risk"].map(RISK_LABELS)
    output["Confidence"] = np.round(
        probabilities[np.arange(len(predictions)), predictions] * 100,
        2,
    )
    output["Probability_Low"] = np.round(probabilities[:, 0] * 100, 2)
    output["Probability_Medium"] = np.round(probabilities[:, 1] * 100, 2)
    output["Probability_High"] = np.round(probabilities[:, 2] * 100, 2)
    output["Correct"] = output["Risk"] == output["Predicted_Risk"]
    output = output.drop(columns=["Risk", "Predicted_Risk"])
    output.to_csv(PREDICTIONS_PATH, index=False)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw = load_data()
    train_df, test_df, district_encoding = prepare_features(train_raw, test_raw)

    model = train_model(train_df)
    X_test = test_df[FEATURE_COLS].astype(float)
    predictions = model.predict(X_test).astype(int)
    probabilities = model.predict_proba(X_test)

    metrics = build_metrics(test_df, predictions, probabilities)
    save_predictions(test_df, predictions, probabilities)
    joblib.dump(model, MODEL_PATH)

    report = {
        "generated_at": datetime.now().isoformat(),
        "model_path": str(MODEL_PATH.relative_to(BASE_DIR)),
        "predictions_path": str(PREDICTIONS_PATH.relative_to(BASE_DIR)),
        "train_data": {
            "path": str(TRAIN_DATA_PATH.relative_to(BASE_DIR)),
            "years": [int(train_df["Year"].min()), int(train_df["Year"].max())],
            "rows": int(len(train_df)),
        },
        "test_data": {
            "path": str(TEST_DATA_PATH.relative_to(BASE_DIR)),
            "years": [int(test_df["Year"].min()), int(test_df["Year"].max())],
            "weeks": int(test_df.groupby(["Year", "Week"]).ngroups),
            "rows": int(len(test_df)),
            "districts": int(test_df["District"].nunique()),
        },
        "risk_thresholds": RISK_THRESHOLDS,
        "feature_names": FEATURE_COLS,
        "district_encoding": district_encoding,
        "metrics": metrics,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Balanced accuracy: {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"High-risk recall: {metrics['high_risk_recall'] * 100:.2f}%")
    print(f"High-risk precision: {metrics['high_risk_precision'] * 100:.2f}%")


if __name__ == "__main__":
    main()
