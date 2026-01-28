"""
A/B testing and simulation helper functions for Cat Adoption Analysis.
"""

import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import mannwhitneyu

def z_test(df_ab: pd.DataFrame) -> tuple[float, float, tuple[float, float]]:
    """
    Test for 30-day adoption rates based on a normal (z) test for two groups:
    named cats (first group) and unnamed cats (second group).

    Parameters
    ----------
    df_ab: pd.DataFrame
        The data frame with binary columns "has_name" and "adopted_30d".

    Returns
    -------
    z_stat : float
        Z-statistics.
    p-value: float
        p-value.
    rates: tuple[float, float]
        30-day adoption rates for named and unnamed cats, respectively.
    """

    success = [
        df_ab.loc[df_ab["has_name"], "adopted_30d"].sum(),
        df_ab.loc[~df_ab["has_name"], "adopted_30d"].sum()
    ]

    nobs = [
        len(df_ab[df_ab["has_name"]]),
        len(df_ab[~df_ab["has_name"]])
    ]
    
    rates = [
        df_ab.loc[df_ab["has_name"], "adopted_30d"].mean(),
        df_ab.loc[~df_ab["has_name"], "adopted_30d"].mean()
    ]

    z_stat, p_value = proportions_ztest(success, nobs)
    
    return z_stat, p_value, rates


def mwu_test(df_adopted_30d: pd.DataFrame) -> tuple[float, float]:
    """
    Test for distributions of LOS among cats that are adopted within 30 days 
    of intake: named cats (first group) and unnamed cats (second group).
    

    Parameters
    ----------
    df_ab: pd.DataFrame
        The data frame with binary column "has_name" and numeric column 
        "length_of_stay_days".

    Returns
    -------
    U_statistic : float
        U-statistics.
    p-value: float
        p-value.
    """
    
    los_named = df_adopted_30d.loc[
        df_adopted_30d["has_name"], 
        "length_of_stay_days"
    ]
    
    los_unnamed = df_adopted_30d.loc[
        ~df_adopted_30d["has_name"], 
        "length_of_stay_days"
    ]
    
    U_statistic, p_value = mannwhitneyu(
        los_named, 
        los_unnamed, 
        alternative="two-sided"
    )
    
    return U_statistic, p_value


def simulations(
        df_sim: pd.DataFrame, 
        roll_out_rates: list[float], 
        n_sim: int, 
        uplift: float,
        seed: int=2026
    ) -> pd.DataFrame:
    """
    Run simulations to compute 30-day adoption rates after naming 
    under different rollout scenarios.

    Parameters
    ----------
    df_sim: pd.DataFrame
        The data frame with numeric column "baseline_adoption_prob".
    roll_out_rates: list[float]
        Different rollout rates.
    n_sim: int
        Number of simulations.
    uplift: float
        Assumed absolute uplift from having a name.
    seed: int=206
        Seed for randomization.
        
    Returns
    -------
    sim_results : pd.DataFrame
        Results of the simulations: 30-day adoption rates after naming 
        under the given rollout scenarios.
    """
    
    results = []

    rng = np.random.default_rng(seed=seed)

    for rollout in roll_out_rates:
        sim_adoptions = []

        for _ in range(n_sim):
            treated = rng.random(len(df_sim)) < rollout
        
            probs = df_sim["baseline_adoption_prob"].values.copy()
            probs[treated] = np.clip(probs[treated] + uplift, 0, 1)

            outcomes = rng.binomial(1, probs)
            sim_adoptions.append(outcomes.mean())

        results.append({
            "rollout_rate": rollout,
            "expected_30d_adoption_rate": np.mean(sim_adoptions),
            "std_dev": np.std(sim_adoptions)
        })

    sim_results = pd.DataFrame(results)
    
    return sim_results


def compute_30d_adoptions(
        df: pd.DataFrame, 
        sim_results: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Using simulation results from above, calculate annual 30-day adoptions 
    under given rollout scenarios and increased annual 30-day adoptions 
    compared to baseline.

    Parameters
    ----------
    df: pd.DataFrame
        The data frame with binary columns "has_name" and "is_adopted", 
        datetime column "datetime_intake" and numeric column 
        "length_of_stay_days".
    sim_results: pd.DataFrame
        Simulation results from function `simulation`.
        
    Returns
    -------
    sim_results: pd.DataFrame
        Simulation results with added columns "annual_30d_adoptions" and
        "increased_30d_adoptions".
    """
    
    annual_intakes_no_name = (
        df[~df["has_name"]]
        .assign(year=lambda x: x["datetime_intake"].dt.year)
        .groupby("year")
        .size()
        .mean()
    )

    print(f"Annual intakes of unnamed cats: {round(annual_intakes_no_name)}.")
    
    baseline_adoption_30d_rate = (
        df[~df["has_name"]]
        .assign(adoption_30d=lambda x: (x["is_adopted"]) & (x["length_of_stay_days"]<=30))
        ["adoption_30d"].mean()
    )
    
    print(f"Baseline 30-day adoption rate among unnamed cats: {baseline_adoption_30d_rate:.2%}.\n")

    baseline_adoptions_30d = annual_intakes_no_name * baseline_adoption_30d_rate

    sim_results["annual_30d_adoptions"] = (
        sim_results["expected_30d_adoption_rate"]*annual_intakes_no_name
    )
    
    sim_results["increased_30d_adoptions"] = (
        sim_results["annual_30d_adoptions"] - baseline_adoptions_30d
    )
    
    return sim_results