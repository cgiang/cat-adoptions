"""
Production-style prediction drift detection for Cat Adoption Analysis.
"""

import pandas as pd
from sklearn.pipeline import Pipeline


def predicted_rate_shift(
        model: Pipeline, 
        reference: pd.DataFrame, 
        new: pd.DataFrame, 
        feature_cols: list[str] 
    ) -> tuple[float, float, float]:
    
    if len(reference) == 0 or len(new) == 0:
        raise ValueError("Reference and/or new window empty.")
    
    ref_preds = model.predict_proba(reference[feature_cols])[:, 1]
    new_preds = model.predict_proba(new[feature_cols])[:, 1]

    ref_mean = ref_preds.mean()
    new_mean = new_preds.mean()
    delta = new_mean - ref_mean

    return ref_mean, new_mean, delta
