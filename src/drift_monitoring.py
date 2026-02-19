"""
Production-style data, metric, and prediction drift detection 
for Cat Adoption Analysis.
"""

import sys
import pandas as pd
import joblib
from src.features import collapse_rare_levels
from src.drift_features import get_features_drift
from src.drift_metrics import adoption_rate_shift
from src.drift_predictions import predicted_rate_shift


NUMERIC_FEATURES = [
    "age_intake_months"
]

CATEGORICAL_FEATURES = [
    "age_group_intake",
    "intake_type",
    "intake_condition",
    "breed_intake",
    "color_intake",
    "has_name",
    "intake_month"
]

DATA_PATH = "data/processed/aac_processed.csv"
MODEL_PATH = "models/xgb_adoption_model.pkl"

DRIFT_SHARE_THRESHOLD = 0.5 # 50% share of drifted columns threshold
METRIC_SHIFT_THRESHOLD = 0.03  # 3% absolute metric shift threshold
PRED_SHIFT_THRESHOLD = 0.03 # 3% absolute adoption shift threshold


def prepare_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_all = pd.read_csv(DATA_PATH, parse_dates=["datetime_intake"])

    # filter for intakes with an outcome
    df = df_all[df_all["has_outcome"]].copy()
    
    # define name presence
    df["has_name"] = df["name_intake"].notna()
    
    # define intake month
    df["intake_month"] = df["datetime_intake"].dt.month.astype("Int64")
    
    # collapse rare levels 
    df["intake_condition"] = collapse_rare_levels(df["intake_condition"])
    df["breed_intake"] = collapse_rare_levels(df["breed_intake"])
    df["color_intake"] = collapse_rare_levels(df["color_intake"])
    
    # define trailing annual windows for reference data and new data
    reference = df[
        (df["datetime_intake"] >= "2023-03-31") &
        (df["datetime_intake"] < "2024-03-31")
    ]

    new = df[
        (df["datetime_intake"] >= "2024-03-31") &
        (df["datetime_intake"] < "2025-03-31")
    ]
    
    if len(reference) < 100 or len(new) < 100:
        raise ValueError("Insufficient data in monitoring windows.")
    
    return reference, new


def detect_feature_drift(reference: pd.DataFrame, new: pd.DataFrame) -> bool:
    print("\n===== Feature Drift =====")

    drift_share, drifted_columns = get_features_drift(
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        reference,
        new
    )
    
    print(f"Share of drifted columns: {drift_share}")
    if len(drifted_columns) > 0:
        print(f"Drifted columns: {drifted_columns}")
    else: 
        print("No drifted columns.")
    
    feature_drift = False
    if drift_share > DRIFT_SHARE_THRESHOLD:
        feature_drift = True
        print("\nShare of drifted columns exceeds threshold.")
    
    return feature_drift


def detect_metric_drift(reference: pd.DataFrame, new: pd.DataFrame) -> bool:
    print("\n===== Metric Drift (Actual Adoption Rate) =====")

    ref_rate, new_rate, metric_delta, p_value = adoption_rate_shift(reference, new)

    print(f"Reference rate: {ref_rate:.2%}")
    print(f"New rate: {new_rate:.2%}")
    print(f"Delta: {metric_delta:.2%}")
    print(f"p-value: {p_value:.4f}")
    
    metric_drift = False
    if abs(metric_delta) > METRIC_SHIFT_THRESHOLD:
        metric_drift = True
        print("\nAdoption rate shift exceeds threshold.")
        
    return metric_drift


def detect_pred_drift(reference: pd.DataFrame, new: pd.DataFrame) -> bool:
    print("\n===== Predicted Rate Drift =====")
    
    model = joblib.load(MODEL_PATH)
    
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    
    ref_mean, new_mean, pred_delta = predicted_rate_shift(
        model,
        reference,
        new,
        feature_cols
    )

    print(f"Reference predicted rate: {ref_mean:.2%}")
    print(f"New predicted rate: {new_mean:.2%}")
    print(f"Predicted delta: {pred_delta:.2%}")
    
    pred_drift = False
    if abs(pred_delta) > PRED_SHIFT_THRESHOLD:
        pred_drift = True
        print("\nPredicted adoption probability shift exceeds threshold.")
    
    return pred_drift


def main():
    reference, new = prepare_data(DATA_PATH)
    
    feature_drift = detect_feature_drift(reference, new)
    metric_drift = detect_metric_drift(reference, new)
    pred_drift = detect_pred_drift(reference, new)
    
    fail = False
        
    if pred_drift:
        fail = True
    elif metric_drift and feature_drift:
        fail = True
    else:
        fail = False

    if fail:
        print("\nDRIFT DETECTED! Model performance may degrade.")
        sys.exit(1)
    else:
        print("\nNo significant drift.")
        sys.exit(0)


if __name__ == "__main__":
    main()

