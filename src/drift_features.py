"""
Generate Evidently data drift report for Cat Adoption Analysis.
"""

import pandas as pd
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset


def generate_evidently_report(
        numeric_features: list[str], 
        categorical_features: list[str], 
        reference: pd.DataFrame, 
        new: pd.DataFrame,
        output_path: str = "artifacts/drift_report.html"):
    
    # specify columns
    feature_cols = numeric_features + categorical_features
    
    data_definition = DataDefinition(
        numerical_columns=numeric_features,
        categorical_columns=categorical_features
    )
    
    # run data drift report 
    report = Report(metrics=[DataDriftPreset()])

    report_results = report.run(
        reference_data=Dataset.from_pandas(
            data=reference[feature_cols], 
            data_definition=data_definition
        ),
        current_data=Dataset.from_pandas(
            data=new[feature_cols], 
            data_definition=data_definition
        )
    )
    
    # save report to html
    report_results.save_html(output_path)
    
    return report_results


def get_features_drift(
        numeric_features: list[str], 
        categorical_features: list[str], 
        reference: pd.DataFrame, 
        new: pd.DataFrame
    ) -> tuple[float, list[str]]:
    
    # extract metrics from results
    report_results = generate_evidently_report(
        numeric_features,
        categorical_features,
        reference,
        new
    )
    results_dict = report_results.dict()
    metrics = results_dict["metrics"]
    
    # extract share of drifted columns
    drift_share = metrics[0]["value"]["share"]

    # extract drifted columns
    drifted_columns = [
        metrics[i]["config"]["column"] for i in range(1, len(metrics))
        if metrics[i]["value"] > metrics[i]["config"]["threshold"]
    ]
    
    return drift_share, drifted_columns