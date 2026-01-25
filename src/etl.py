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
    # strip spaces from column names and convert to lower case
    df = df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in df.columns})
    return df


def age_group(m) -> str:
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

    # filter to cats if animal_type column exists
    if "animal_type" in df.columns:
        df = df[df["animal_type"].astype(str).str.strip().str.lower() == "cat"]

    # normalize text columns
    if "name" in df.columns:
        df["name"] = df["name"].str.replace("\\*", "", regex=True).str.strip()
        df['name'] = df['name'].replace("Unknown", "")
        df['name'] = df['name'].replace("Unknown-Stray", "")
        df['name'] = df['name'].fillna("").astype(str)
        
    return df

def process_intakes(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess(raw_df)
    debug = False #True
    
    # parse dates
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.rename(columns={"datetime": "datetime_intake"}, inplace=True)
        # check time zone
        if debug:
            tz = df["datetime_intake"].dt.tz
            if tz:
                print(f"\nTime zone of intakes: {tz}")
            else:
                print("\nIntakes do not specify a time zone.")
        df["datetime_intake"] = df["datetime_intake"].dt.tz_localize("UTC-05:00")
        
    if "monthyear" in df.columns:
        df["monthyear"] = pd.to_datetime(df["monthyear"], format="mixed", dayfirst=False)
        df.rename(columns={'monthyear': 'month_year_intake'}, inplace=True)
    
    # age to months
    if "age_upon_intake" in df.columns:
        df["age_intake_months"] = df["age_upon_intake"].apply(parse_age_to_months)
 
    # normalize text columns
    if "sex_upon_intake" in df.columns:
        df["sex_upon_intake"] = df["sex_upon_intake"].replace("Unknown", "")
        
    # classify into age groups
    if "age_intake_months" in df.columns:
        df["age_group_intake"] = df["age_intake_months"].apply(age_group)
    
    return df

def process_outcomes(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess(raw_df)
    debug = False #True
    
    # parse dates
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601")
        df.rename(columns={"datetime": "datetime_outcome"}, inplace=True)
        # check time zone
        if debug:
            tz = df["datetime_outcome"].dt.tz
            if tz:
                print(f"\nTime zone of outcomes: {tz}")
            else:
                print("\nOutcomes do not specify a time zone.")
        df["datetime_outcome"] = df["datetime_outcome"].dt.tz_convert("UTC-05:00")
        
    if "monthyear" in df.columns:
        df["monthyear"] = pd.to_datetime(df["monthyear"], format="mixed", dayfirst=False)
        df.rename(columns={"monthyear": "month_year_outcome"}, inplace=True)
    if "date_of_birth" in df.columns:
        df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])
    
    # age to months
    if "age_upon_outcome" in df.columns:
        df["age_outcome_months"] = df["age_upon_outcome"].apply(parse_age_to_months)
        
    # normalize text columns
    if "sex_upon_outcome" in df.columns:
        df["sex_upon_outcome"] = df["sex_upon_outcome"].replace("Unknown", "")
    if "outcome_type" in df.columns:
        df["outcome_type"] = df["outcome_type"].replace("Unknown", "")
    if "outcome_subtype" in df.columns:
        df["outcome_subtype"] = df["outcome_subtype"].fillna("").astype(str).str.title()
        
    # classify into age groups
    if "age_outcome_months" in df.columns:
        df["age_group_outcome"] = df["age_outcome_months"].apply(age_group)
        
    return df


def validate_join_key(intakes: pd.DataFrame, outcomes: pd.DataFrame, key: str):
    print("\n--- JOIN KEY VALIDATION ---")

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
    has_outcome = df["outcome_type"].notna().mean()
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
        validate_join_key(intakes, outcomes, key="animal_id")
        inspect_multiplicity(intakes, outcomes, key="animal_id")
        
    # pair intakes and outcomes
    intakes_sorted = intakes.sort_values(by="datetime_intake", ignore_index=True)
    outcomes_sorted = outcomes.sort_values(by="datetime_outcome", ignore_index=True)
    
    # match on same animal_id, and match each intake to the first outcome that 
    # occurs no earlier than that intake
    df = pd.merge_asof(intakes_sorted, 
                       outcomes_sorted, 
                       by="animal_id", 
                       left_on="datetime_intake",
                       right_on="datetime_outcome",
                       direction="forward",
                       suffixes=["_intake", "_outcome"])
    
    # optional validation checks used during development
    if debug: 
        validate_post_join(df, intakes, key="animal_id")
        
    # create variables related to outcome and length of stay (days)
    df["is_adopted"] = (df["outcome_type"] == 'Adoption')
    df["has_outcome"] = df["outcome_type"].notna()
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
