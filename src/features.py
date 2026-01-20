"""
Feature engineering for Cat Adoption Analysis.

This script constructs model-ready features derived strictly from
information available at intake time. Outcome-related variables are
excluded from feature selection to prevent outcome leakage.

The script also includes a helper variable for simulation purposes only
(i.e. baseline adoption probability) that is not used for SQL analysis and
supervised modeling.
"""

import pandas as pd
import numpy as np

# a months to seasons
SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}


def build_intake_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build intake-related features for adoption modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical intake-level dataset produced by ETL.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix containing only intake-time variables.
    """

    features = pd.DataFrame(index=df.index)

    # age features
    features["age_intake_months"] = pd.to_numeric(df["age_intake_months"])
    features["is_kitten_at_intake"] = (df["age_group_intake"] == "kitten").astype(int)
    features["age_group_at_intake"] = df["age_group_intake"].fillna("Unknown")

    # intake characteristics
    features["sex_at_intake"] = df["Sex upon Intake"].fillna("Unknown")
    features["intake_type"] = df["Intake Type"].fillna("Unknown")
    features["has_name"] = df["Name_intake"].notna().astype(int)

    # intake time features
    features["intake_month"] = df["datetime_intake"].dt.month.astype("Int64")
    features["intake_season"] = features["intake_month"].map(SEASON_MAP).fillna("Unknown")

    return features


def build_targets(df: pd.DataFrame) -> pd.Series:
    """
    Build target variable for supervised adoption modeling.
    Records without outcomes are excluded.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical intake-level dataset produced by ETL.

    Returns
    -------
    y : pd.Series
        Binary adoption outcome (1 = adopted, 0 = not adopted).
    """

    if "has_outcome" not in df.columns or "is_adopted" not in df.columns:
        raise ValueError("Missing required columns for target feature.")
    y = df.loc[df["has_outcome"], "is_adopted"].astype(int)
    y = y.rename("adopted")
    
    return y


def add_baseline_adoption_prob(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a heuristic baseline adoption probability for A/B simulation.

    NOTE:
    - Simulation-only feature (NOT used for modeling)
    - Constructed from intake-related variables only
    
    Parameters
    ----------
    df : pd.DataFrame
        Canonical intake-level dataset.

    Returns
    -------
    df : pd.DataFrame
        Data frame with an added "baseline_adoption_prob" column.
    """

    p = pd.Series(0.30, index=df.index)  # global mean adoption rate

    # age-based adjustment 
    # kittens are most likely to be adopted, followed by mature adults, 
    # young adults and seniors
    age = df["age_intake_months"]

    p += np.select(
        condlist=[
            age <= 12,
            (age > 12) & (age <= 72),
            (age > 72) & (age <= 120),
            age > 120,
        ],
        choicelist=[
            0.25,   # kitten
            0.10,   # young adult
            0.15,   # mature adult
           -0.05,   # senior
        ],
        default=0.0
    )

    # intake type adjustment
    # owner surrender has the highest adoption rate, followed by abandoned,
    # stray, public assist and euthanasia request
    intake_adjustment = {
        "Owner Surrender": 0.30,
        "Abandoned": 0.20,
        "Public Assist": -0.10,
        "Euthanasia Request": -0.20,
    }

    p += df["intake_type"].map(intake_adjustment).fillna(0.0)

    # name presence adjustment 
    # cat having a name is more likely to be adopted 
    p += np.where(df["has_name"], 0.30, -0.15)

    # avoid probabilities of 0 or 1 
    df["baseline_adoption_prob"] = p.clip(0.01, 0.99)

    return df


def validate_feature_inputs(df: pd.DataFrame):
    """
    Optional validation to ensure required intake-related columns exist
    before feature construction.
    
    Parameters
    ----------
    df : pd.DataFrame
        Canonical intake-level dataset produced by ETL.
    """

    required_columns = [
        "age_intake_months",
        "age_group_intake",
        "Sex upon Intake",
        "Intake Type",
        "Name_intake",
        "datetime_intake"
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {missing}")
