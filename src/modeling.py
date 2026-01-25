"""
Modeling helper functions for Cat Adoption Analysis.
"""

import pandas as pd
import numpy as np
from typing import Union, Any
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from xgboost import XGBRegressor

from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, r2_score
import shap

def preprocess_features(
        categorical_cols: list[str],
        numeric_cols: list[str],
        binary_cols: list[str]
    ) -> list[tuple[str, Any, Union[str, list[str], int]]]:
    """
    Preprocess features for adoption modeling.

    Parameters
    ----------
    categorical_cols: list[str]
        Column names of categorical variables.
    numeric_cols: list[str]
        Column names of numeric variables.
    binary_cols: list[str]
        Column names of binary variables.

    Returns
    -------
    preprocessor : list[tuple[str, Any, Union[str, list[str], int]]]
        Preprocessor that transforms feature formats for modeling.
    """
    
    # log1p is used to reduce skew in numeric features (assumes non-negative inputs)
    log_transformer = FunctionTransformer(
        func=np.log1p, 
        validate=True, 
        feature_names_out='one-to-one'
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("log_trans", log_transformer, numeric_cols),
            ("binary", "passthrough", binary_cols)
        ]
    )

    return preprocessor 


def train_logistic_regression(
        preprocessor: list[tuple[str, Any, Union[str, list[str], int]]], 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        max_iter: int=1000,
        solver: str="liblinear",
        l1_ratio: int=0
    ) -> Pipeline:
    """
    Train logistic regression model.

    Parameters
    ----------
    preprocessor : list[tuple[str, Any, Union[str, list[str], int]]]
        Preprocessor that transforms feature formats for modeling.
    X_train: pd.DataFrame
        Features in the training set.
    y_train: pd.Series
        Response variable in the training set.
    max_iter: int, default=1000
        Maximum number of iterations taken for the solvers to converge.
    solver: str, default="liblinear"
        Algorithm to use in the optimization problem.
    l1_ratio: int, default=0
        Setting l1_ratio=1 gives a L1-penalty and setting l1_ratio=0 
        gives a L2-penalty.

    Returns
    -------
    log_reg : Pipeline
        Trained logistic regression model.
    """
    
    log_reg = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", LogisticRegression(
            max_iter=max_iter,
            solver=solver,
            l1_ratio=l1_ratio
        ))
    ])

    log_reg.fit(X_train, y_train)
    
    return log_reg


def train_xgb_classifier(
        preprocessor: list[tuple[str, Any, Union[str, list[str], int]]], 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        n_estimators: int=200,
        max_depth: int=4,
        learning_rate: int=0.05,
        eval_metric: str="logloss",
        random_state: int=2026
    ) -> Pipeline:
    """
    Train XGBoost Classifier model.

    Parameters
    ----------
    preprocessor : list[tuple[str, Any, Union[str, list[str], int]]]
        Preprocessor that transforms feature formats for modeling.
    X_train: pd.DataFrame
        Features in the training set.
    y_train: pd.Series
        Response variable in the training set.
    n_estimators: int, default=200
        Number of boosting rounds.
    max_depth: int, default=4
        Maximum tree depth for base learners.
    learning_rate: int, default=0.05,
        Boosting learning rate.
    eval_metric: str, default="logloss"
        Metric used for monitoring the training result and early stopping. 
    random_state: int=2026
        Random number seed.
    
    Returns
    -------
    xgb : Pipeline
        Trained XGBoost Classifier model.
    """
    
    xgb = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                eval_metric=eval_metric,
                random_state=random_state
            ))
        ]
    )

    xgb.fit(X_train, y_train)
    
    return xgb

def train_xgb_regressor(
        preprocessor: list[tuple[str, Any, Union[str, list[str], int]]], 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        n_estimators: int=200,
        max_depth: int=4,
        learning_rate: int=0.05,
        eval_metric: str="rmse",
        random_state: int=2026
    ) -> Pipeline:
    """
    Train XGBoost Regressor model.

    Parameters
    ----------
    preprocessor : list[tuple[str, Any, Union[str, list[str], int]]]
        Preprocessor that transforms feature formats for modeling.
    X_train: pd.DataFrame
        Features in the training set.
    y_train: pd.Series
        Response variable in the training set.
    n_estimators: int, default=200
        Number of boosting rounds.
    max_depth: int, default=4
        Maximum tree depth for base learners.
    learning_rate: int, default=0.05,
        Boosting learning rate.
    eval_metric: str, default="rmse"
        Metric used for monitoring the training result and early stopping. 
    random_state: int=2026
        Random number seed.

    Returns
    -------
    xgb : Pipeline
        Trained XGBoost Regressor model.
    """
    
    xgb = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                eval_metric=eval_metric,
                random_state=random_state
            ))
        ]
    )

    xgb.fit(X_train, y_train)
    
    return xgb
   
def evaluate_classifier(
        model: Pipeline, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ):
    """
    Evaluate classifier model.

    Parameters
    ----------
    model: Pipeline
        Trained classifier model.
    X_test: pd.DataFrame
       Features in the test set.
    y_test: pd.Series
        Response variable in the test set.
    """
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred))
    
    # compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # visualize
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=["Not Adopted", "Adopted"]
    )
    disp.plot(cmap=plt.cm.Blues) 
    plt.title('Confusion Matrix')
    plt.show()
    

def evaluate_regressor(
        model: Pipeline, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ):
    """
    Evaluate regressor model.

    Parameters
    ----------
    model: Pipeline
        Trained regressor model.
    X_test: pd.DataFrame
       Features in the test set.
    y_test: pd.Series
        Response variable in the test set.
    """

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    print(f"R-squared: {r2:.4f}")
    print(f"Mean squared error: {mse:.4f}")
    print(f"Root mean squared error: {rmse:.4f}")
    
    
def feature_importance_logistic(
        reg: Pipeline, 
        feature_list: list[str]
    ) -> pd.DataFrame:
    """
    Compute grouped feature importance of features in a logistic regression model.
    Grouped feature importance is defined as the sum of absolute coefficients
    of all features a given group.

    Parameters
    ----------
    reg: Pipeline
        Trained logistic regression model.
    feature_list: list[str]
        List of grouped features.
        
    Returns
    -------
    group_importance: pd.DataFrame
        Grouped features and their importance.
    """
    
    feature_names = reg.named_steps["prep"].get_feature_names_out()
    coefs = reg.named_steps["model"].coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "abs_coef": np.abs(coefs)
    })

    # group by feature family
    coef_df["group"] = coef_df["feature"].str.extract(
        "(" + "|".join(feature_list) + ")"
    )

    group_importance = (
        coef_df
        .dropna(subset=["group"])
        .groupby("group")["abs_coef"]
        .sum()
        .sort_values(ascending=False)
    )

    return group_importance


def feature_importance_shap(
        xgb: Pipeline, 
        X_test: pd.DataFrame, 
        feature_list: list[str]
    ) -> pd.DataFrame:
    """
    Compute grouped feature importance of features in an XGBoost model.
    Grouped feature importance is defined as the sum of absolute SHAP values
    of all features a given group.

    Parameters
    ----------
    xgb: Pipeline
        Trained XBoost model.
    X_test: pd.DataFrame
        Features in the test set.
    feature_list: list[str]
        List of grouped features.
        
    Returns
    -------
    group_importance: pd.DataFrame
        Grouped features and their importance.
    """

    feature_names = xgb.named_steps["prep"].get_feature_names_out()
    xgb_model = xgb.named_steps["model"]
    X_test_transformed = xgb.named_steps["prep"].transform(X_test)
    
    X_test_dense = X_test_transformed.toarray()
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(X_test_dense)

    # mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    shap_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    })

    # group by feature family
    shap_importance["group"] = shap_importance["feature"].str.extract(
        "(" + "|".join(feature_list) + ")"
    )

    group_shap_importance = (
        shap_importance
        .groupby("group")["mean_abs_shap"]
        .sum()
        .sort_values(ascending=False)
    )
    
    return group_shap_importance
    