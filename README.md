# 🐱 Cat Adoption Analysis

## Project Overview

This project analyzes cat intake and adoption data from the Austin Animal Center to understand **which intake-related factors drive adoption likelihood and speed**, and to evaluate a simple **intervention: assigning names to cats upon intake**. This analysis includes ETL, exploratory analysis, predictive modeling, SQL analysis, A/B testing, and an interactive dashboard.

## Key Questions

* Which intake-related factors most strongly predict adoption?
* How do age, name presence and appearance affect adoption rate and time to adoption?
* If naming cats increases adoption, is it worth doing so at scale?
* What are the trade-offs and risks of naming or not naming cats?

## Data

* Source: Austin Animal Center [intake](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Intakes-10-01-2013-to-05-05-2/wter-evkm/about_data) and [outcome](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes-10-01-2013-to-05-05-/9t4d-g238) data sets from October 1, 2013 to May 5, 2025

  * Intakes: intake date, age, intake type, condition, name presence, breed, color
  * Outcomes: adoption outcome (binary) and length of stay (days)

* Scope: Cats only

These datasets are cleaned, standardized, and joined into a single canonical intake-level table during ETL. All predictive models are trained using only intake-related features to avoid outcome leakage.

## Methodology

### 1\. ETL \& Feature Engineering

* Clean and join intake and outcome records
* Handle multiple intakes per cat using time-aware joins
* Engineer intake-related features (age, breed, color, name presence, intake condition, etc.)
* Group rare categorical levels to reduce variance

### 2\. Exploratory Data Analysis (EDA)

Key findings:

* Kittens are adopted faster and at higher rates
* Cats with names at intake have higher adoption rates
* Intake volume and adoption patterns are strongly seasonal
* ~1% of intakes lack outcomes (mostly recent), excluded from adoption-rate denominators

### 3\. Predictive Modeling

Two complementary models to predict 30-day adoption likelihood:

* Logistic regression (interpretable baseline)
* XGBoost (flexible, higher-performing model)

Best-performing model (XGBoost):

* ROC-AUC ≈ 0.84
* Balanced precision/recall
* Color, intake condition, breed, age at intake, and name presence are strong predictors

Length-of-stay (LOS) modeling among adopted cats shows limited predictive power, which suggests that many drivers of time to adoption occur post-intake.

### 4\. SQL Analysis

SQL queries replicate key metrics:

* Adoption rate and median LOS by age group at intake
* Seasonal trends in intake volume and adoption rate
* Adoption outcomes by name presence

This section translates analysis into stakeholder-facing queries.

### 5\. A/B Testing \& Simulation

Because naming is not randomized, the analysis is observational, not a randomized experiment.

Approach:

* Compare 30-day adoption outcomes for named vs unnamed cats
* Use statistical tests for adoption rate and LOS differences as **directional evidence**, not causal proof
* Simulate rollout scenarios using baseline adoption probabilities

Outcome:

* Name presence shows a positive lift
* Simulations suggest meaningful upside at scale with limited downside risk

Recommendation:

* Proceed with a **limited rollout of naming at intake**
* Monitor 30-day adoption rate as the primary metric and use LOS as a guardrail.
* Evaluate results before expanding the intervention at scale

## Dashboard

\[Link to Streamlit app](https://cat-adoptions.streamlit.app/)



An interactive Streamlit dashboard allows non-technical users to:

* Filter by age group and name presence
* Explore adoption rates and time to adoption
* View seasonal intake trends
* See the final recommendation in context

## Project Scope \& Framing

This project is structured to mirror a typical DS workflow: defining a business problem, working with intake-related data constraints, building interpretable and flexible models, and using statistical analysis to inform an intervention decision.

Predictive modeling serves as one input into the decision, along with exploratory analysis, SQL-based metrics, and simulated impact at scale.

## Tech Stack

* Python (pandas, numpy, scikit-learn, xgboost, shap)
* SQL (MySQL, window functions, CTEs)
* statsmodels, SciPy
* Streamlit
* GitHub
