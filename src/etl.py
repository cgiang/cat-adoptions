"""
ETL for Cat Adoption Analysis

This script reads two raw CSV files (intakes and outcomes), normalizes columns, 
computes age in months and writes processed CSV files to the processed folder.
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


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    # filter to cats if Animal Type column exists
    if "Animal Type" in df.columns:
        df = df[df["Animal Type"].astype(str).str.strip().str.lower() == "cat"]

    # parse dates
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    if "Date of Birth" in df.columns:
        df["Date of Birth"] = pd.to_datetime(df["Date of Birth"], errors="coerce")

    # age to months
    if "Age upon Intake" in df.columns:
        df["Age upon Intake (months)"] = df["Age upon Intake"].apply(parse_age_to_months)
    elif "Age upon Outcome" in df.columns:
        df["Age upon Outcome (months)"] = df["Age upon Outcome"].apply(parse_age_to_months)


    # normalize text columns
    if "Name" in df.columns:
        df["Name"] = df["Name"].fillna("Unknown").astype(str).str.title()
        df["Name"] = df["Name"].astype(str).replace("\\*", "", regex=True).str.strip()
    if "Sex upon Intake" in df.columns:
        df["Sex upon Intake"] = df["Sex upon Intake"].fillna("Unknown").astype(str).str.title()
    if "Sex upon Outcome" in df.columns:
        df["Sex upon Outcome"] = df["Sex upon Outcome"].fillna("Unknown").astype(str).str.title()
    if "Outcome Type" in df.columns:
        df["Outcome Type"] = df["Outcome Type"].fillna("Unknown").astype(str).str.title()
    if "Outcome Subtype" in df.columns:
        df["Outcome Subtype"] = df["Outcome Subtype"].fillna("").astype(str).str.title()

    # age group buckets, according to 2021 AAHA/AAFP Feline Life Stage Guidelines
    def age_group(m):
        if pd.isna(m):
            return "unknown"
        if m <= 12:
            return "kitten"
        if m <= 72:
            return "young adult"
        if m <= 120:
            return "mature adult"
        return "senior"
    
    # classify into age groups
    if "Age upon Intake (months)" in df.columns:
        df["age_group_intake"] = df["Age upon Intake (months)"].apply(age_group)
    elif "Age upon Outcome (months)" in df.columns:
        df["age_group_outcome"] = df["Age upon Outcome (months)"].apply(age_group)
        

    # simple baseline adoption probability heuristics for A/B simulation
    def baseline_prob(row):
        p = 0.2 # modest base assumption
        if "Age upon Intake (months)" in df.columns:
            age = row.get("Age upon Intake (months)", np.nan)
        elif "Age upon Outcome (months)" in df.columns:
            age = row.get("Age upon Outcome (months)", np.nan)
        if not pd.isna(age):
            if age < 12:
                p += 0.45 # kittens are more likely to be adopted
            elif age < 72:
                p += 0.20 # young adult cats are less likely to be adopted
            elif age < 120:
                p += 0.05 # mature adult cats are less likely to be adopted
            else:
                p -= 0.10 # senior cats are the least likely to be adopted
        
        outcome = str(row.get("Outcome Type", "")).lower()
        if "missing" in outcome or "transfer" in outcome:
            p -= 0.05
        return float(max(min(p, 0.99), 0.01)) # avoid probs of 0 or 1 

    df["baseline_prob"] = df.apply(baseline_prob, axis=1)

    return df


def main(input_path: str, output_path: str):
    df = load_raw(input_path)
    df_processed = preprocess(df)
    df_processed.to_csv(output_path, index=False)
    print(f"Wrote processed data to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ETL for Cat Adoption Analysis")
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--output", required=True, help="Path to write processed file")
    args = parser.parse_args()
    main(args.input, args.output)
