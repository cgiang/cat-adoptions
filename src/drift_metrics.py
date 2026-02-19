"""
Production-style metric drift detection for Cat Adoption Analysis.
"""

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest


def adoption_rate_shift(
        reference: pd.DataFrame, 
        new: pd.DataFrame
    ) -> tuple[float, float, float, float]:
    ref_success = reference["is_adopted"].sum()
    ref_n = len(reference)

    new_success = new["is_adopted"].sum()
    new_n = len(new)

    # p-value is logged for diagnostics but not used for CI failure
    stat, p_value = proportions_ztest(
        [ref_success, new_success],
        [ref_n, new_n]
    )

    ref_rate = ref_success / ref_n
    new_rate = new_success / new_n
    delta = new_rate - ref_rate

    return ref_rate, new_rate, delta, p_value
