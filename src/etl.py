"""
ETL for Cat Adoption Analysis

This script reads two raw CSV files (intakes and outcomes), normalizes columns, 
computes age in months and length of stay, and writes processed CSV files to 
the processed folder.

NOTE:
Only intake-related variables should be used for modeling.
Outcome-related columns are for analysis only.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re
import argparse

# pattern to parse age column
AGE_PATTERN = re.compile(
    r"(?P<value>[0-9\.]+)\s*(?P<unit>day|days|week|weeks|month|months|year|years)",
    re.I,
)


def load_raw(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    return df


def parse_age_to_months(age_str: str):
    if pd.isna(age_str):
        return np.nan
    # strip spaces from age
    s = str(age_str).strip().lower()
    # parse age to months
    m = AGE_PATTERN.search(s)
    if not m:
        return np.nan
    value = float(m.group("value"))
    unit = m.group("unit")
    if unit.startswith("day"):
        return value/30.0
    elif unit.startswith("week"):
        return value*7.0/30.0
    elif unit.startswith("month"):
        return value
    elif unit.startswith("year"):
        return value*12.0
    return np.nan


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # strip spaces from column names
    df = df.rename(columns={c: c.strip() for c in df.columns})
    return df


def age_group(m):
    if pd.isna(m):
        return "unknown"
    # age group buckets, according to 2021 AAHA/AAFP Feline Life Stage Guidelines
    if m <= 12:
        return "kitten"
    if m <= 72:
        return "young adult"
    if m <= 120:
        return "mature adult"
    return "senior"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    # filter to cats if Animal Type column exists
    if "Animal Type" in df.columns:
        df = df[df["Animal Type"].astype(str).str.strip().str.lower() == "cat"]

    # normalize text columns
    if "Name" in df.columns:
        df["Name"] = df["Name"].fillna("Unknown").astype(str).str.title()
        df["Name"] = df["Name"].astype(str).replace("\\*", "", regex=True).str.strip()

    return df

def process_intakes(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess(raw_df)
    debug = False #True
    
    # parse dates
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df.rename(columns={'DateTime': 'datetime_intake'}, inplace=True)
        # check time zone
        if debug:
            tz = df["datetime_intake"].dt.tz
            if tz:
                print(f"\nTime zone of intakes: {tz}")
            else:
                print("\nIntakes do not specify a time zone.")
        df["datetime_intake"] = df["datetime_intake"].dt.tz_localize("UTC-05:00")
        
    if "MonthYear" in df.columns:
        df["MonthYear"] = pd.to_datetime(df["MonthYear"], format="mixed", dayfirst=False)
        df.rename(columns={'MonthYear': 'month_year_intake'}, inplace=True)
    
    # age to months
    if "Age upon Intake" in df.columns:
        df["age_intake_months"] = df["Age upon Intake"].apply(parse_age_to_months)
 
    # normalize text columns
    if "Sex upon Intake" in df.columns:
        df["Sex upon Intake"] = df["Sex upon Intake"].fillna("Unknown").astype(str).str.title()
        
    # classify into age groups
    if "Age upon Intake (months)" in df.columns:
        df["age_group_intake"] = df["Age upon Intake (months)"].apply(age_group)
    
    return df

def process_outcomes(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess(raw_df)
    debug = False #True
    
    # parse dates
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"], format="ISO8601")
        df.rename(columns={'DateTime': 'datetime_outcome'}, inplace=True)
        # check time zone
        if debug:
            tz = df["datetime_outcome"].dt.tz
            if tz:
                print(f"\nTime zone of outcomes: {tz}")
            else:
                print("\nOutcomes do not specify a time zone.")
        df["datetime_outcome"] = df["datetime_outcome"].dt.tz_convert("UTC-05:00")
        
    if "MonthYear" in df.columns:
        df["MonthYear"] = pd.to_datetime(df["MonthYear"], format="mixed", dayfirst=False)
        df.rename(columns={'MonthYear': 'month_year_outcome'}, inplace=True)
    if "Date of Birth" in df.columns:
        df["Date of Birth"] = pd.to_datetime(df["Date of Birth"])
    
    # age to months
    if "Age upon Outcome" in df.columns:
        df["age_outcome_months"] = df["Age upon Outcome"].apply(parse_age_to_months)
        
    # normalize text columns
    if "Sex upon Outcome" in df.columns:
        df["Sex upon Outcome"] = df["Sex upon Outcome"].fillna("Unknown").astype(str).str.title()
    if "Outcome Type" in df.columns:
        df["Outcome Type"] = df["Outcome Type"].fillna("Unknown").astype(str).str.title()
    if "Outcome Subtype" in df.columns:
        df["Outcome Subtype"] = df["Outcome Subtype"].fillna("").astype(str).str.title()
        
    # classify into age groups
    if "Age upon Outcome (months)" in df.columns:
        df["age_group_outcome"] = df["Age upon Outcome (months)"].apply(age_group)
        
    return df


def validate_join_key(intakes: pd.DataFrame, outcomes: pd.DataFrame, key: str):
    print("--- JOIN KEY VALIDATION ---")

    # row counts
    print(f"Intakes rows:  {len(intakes)}")
    print(f"Outcomes rows: {len(outcomes)}")

    # uniqueness
    intake_unique = intakes[key].nunique()
    outcome_unique = outcomes[key].nunique()

    print(f"\nUnique {key}s:")
    print(f"  Intakes:  {intake_unique}")
    print(f"  Outcomes: {outcome_unique}")

    # duplicates
    dup_intakes = intakes[key].value_counts().gt(1).sum()
    dup_outcomes = outcomes[key].value_counts().gt(1).sum()

    print(f"\nDuplicate {key}s:")
    print(f"  Intakes:  {dup_intakes}")
    print(f"  Outcomes: {dup_outcomes}")

    # coverage
    intakes_only = set(intakes[key]) - set(outcomes[key])
    outcomes_only = set(outcomes[key]) - set(intakes[key])

    print("\nCoverage:")
    print(f"  In intakes but not outcomes: {len(intakes_only)}")
    print(f"  In outcomes but not intakes: {len(outcomes_only)}")
    
    
def inspect_multiplicity(intakes: pd.DataFrame, outcomes: pd.DataFrame, key: str):
    print("\n--- MULTIPLICITY ---")
    intake_counts = intakes[key].value_counts()
    intake_multi = intake_counts[intake_counts > 1]
    
    print(f"\nCount of animals with >1 intake: {len(intake_multi):,}")
    print(f"% of animals with >1 intake: {len(intake_multi) / intake_counts.size:.2%}")
    
    outcome_counts = outcomes[key].value_counts()
    outcome_multi = outcome_counts[outcome_counts > 1]

    print(f"\nCount of animals with >1 outcome: {len(outcome_multi):,}")
    print(f"% of animals with >1 outcome: {len(outcome_multi) / outcome_counts.size:.2%}")
    

def validate_post_join(df: pd.DataFrame, intakes: pd.DataFrame, key: str):
    print("\n--- POST-JOIN VALIDATION ---")

    print(f"\nJoined rows:  {len(df):,}")
    print(f"Intakes rows: {len(intakes):,}")

    if len(df) < len(intakes):
        print("WARNING: Joined table has fewer rows than intakes")

    # outcome coverage
    has_outcome = df["Outcome Type"].notna().mean()
    print(f"\n% with outcome: {has_outcome:.2%}")
    print(f"% without outcome: {1 - has_outcome:.2%}")

    # check for row explosion
    joined_unique = df[key].nunique()
    print(f"\nUnique animals after join: {joined_unique:,}")
    

def build_adoption_episodes(intakes: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs intake -> outcome episodes.

    Each intake is matched to the earliest outcome for the same animal
    that occurs at or after the intake datetime. This handles animals
    with multiple stays.
    """
    # optional validation checks used during development
    debug = False #True
    if debug:
        validate_join_key(intakes, outcomes, key="Animal ID")
        inspect_multiplicity(intakes, outcomes, key="Animal ID")
        
    # pair intakes and outcomes
    intakes_sorted = intakes.sort_values(by="datetime_intake", ignore_index=True)
    outcomes_sorted = outcomes.sort_values(by="datetime_outcome", ignore_index=True)
    
    # match on same Animal ID, and match each intake to the first outcome that 
    # occurs no earlier than that intake
    df = pd.merge_asof(intakes_sorted, 
                       outcomes_sorted, 
                       by="Animal ID", 
                       left_on="datetime_intake",
                       right_on="datetime_outcome",
                       direction="forward",
                       suffixes=["_intake", "_outcome"])
    
    # optional validation checks used during development
    if debug: 
        validate_post_join(df, intakes, key="Animal ID")
        
    # create variables related to outcome and length of stay (days)
    df["is_adopted"] = (df["Outcome Type"] == 'Adoption')
    df["has_outcome"] = df["Outcome Type"].notna()
    df["length_of_stay_days"] = (df["datetime_outcome"] - df["datetime_intake"]).dt.days
    df["invalid_los"] = (df["length_of_stay_days"] < 0)
    
    return df
    

def main(intakes_input_path: str, outcomes_input_path: str, output_dir: str):
    # load data
    df_intakes_raw = load_raw(intakes_input_path)
    df_outcomes_raw = load_raw(outcomes_input_path)
    
    # process data
    df_intakes = process_intakes(df_intakes_raw)
    df_outcomes = process_outcomes(df_outcomes_raw)
    
    # merge intakes and outcomes into one canonical dataset
    df_all = build_adoption_episodes(df_intakes, df_outcomes)
    
    # output final dataset to CSV
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "aac_processed.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"Wrote processed data to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ETL for Cat Adoption Analysis")
    parser.add_argument("--input_intakes", required=True, help="Path to raw intakes CSV")
    parser.add_argument("--input_outcomes", required=True, help="Path to raw outcomes CSV")
    parser.add_argument("--output_dir", required=True, help="Path to write processed file")
    args = parser.parse_args()
    main(args.input_intakes, args.input_outcomes, args.output_dir)
