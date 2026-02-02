"""
Interactive dashboard for Cat Adoption Analysis.
"""

import streamlit as st
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.resolve().parent
DATA_PATH = BASE_DIR / "data" /  "processed" / "aac_processed.csv"


# -----Config-----
st.set_page_config(
    page_title="Cat Adoption Dashboard 🐈",
    layout="wide"
)


# -----Load data-----
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime_intake", 
                                             "month_year_intake"])
    return df

df = load_data()
df["has_name"] = df["name_intake"].notna()


# -----Header-----
st.title("Cat Adoption Analysis 🐱")
st.subheader("**What makes a cat more adoptable and faster?**")
st.write(
    """
    A look into adoption outcomes using ~70,000 cat intakes at the 
    Austin Animal Center from October 2013 to May 2025.
    """
)


# -----Sidebar filters-----
st.sidebar.header("Filters")


age_order = ["kitten", "young adult", "mature adult", "senior"]
age_group = st.sidebar.multiselect(
    "Age Group at Intake",
    options=age_order,
    default=age_order
)

has_name = st.sidebar.selectbox(
    "Has Name at Intake",
    options=["All", "Yes", "No"]
)


# -----Apply filters-----
if age_group:
    filtered = df[df["age_group_intake"].isin(age_group)]
else: # handle the case where no age group is selected
    filtered = df.copy()
    st.warning(
        """
        Please select at least one age group. When no age group is selected,
        no age group filters are applied.
        """)

if has_name != "All":
    filtered = filtered[filtered["has_name"] == (has_name == "Yes")]

adopted_df = filtered[filtered["is_adopted"]]

# -----Overall metrics-----
st.subheader("Overall Adoption Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Intakes", f"{len(filtered):,}")

with col2:
    rate = filtered.loc[filtered['has_outcome'],'is_adopted'].mean()
    st.metric(
        "Adoption Rate",
        f"{rate:.2%}" if pd.notna(rate) else "N/A"
    )

with col3:
    st.metric(
        "Median Length of Stay (days)",
        int(filtered.loc[filtered["is_adopted"], "length_of_stay_days"].median())
    )


# -----Adoption by age group-----
st.subheader("Adoption by Age Group at Intake")

age_los = (
    adopted_df
    .groupby("age_group_intake")["length_of_stay_days"]
    .mean()
)

age_stats = (
    filtered[filtered["has_outcome"]]
    .groupby("age_group_intake")["is_adopted"]
    .agg(["count", "mean"])
    .join(age_los)
    .reset_index()
)

age_stats.columns = ["Age Group at Intake", 
                     "Intake Count", 
                     "Adoption Rate",
                     "Time to Adoptions (Days)"]

age_stats["Age Group at Intake"] = pd.Categorical(
    age_stats["Age Group at Intake"],
    categories=age_order,
    ordered=True
)

age_stats = age_stats.sort_values("Age Group at Intake")

st.write(
    """
    Kittens are adopted faster and at higher rates.
    
    **Average Time to Adoptions (Days)**
    """
)

col1, col2 = st.columns([8, 5], gap="medium")

with col1:
    st.bar_chart(
        age_stats.set_index("Age Group at Intake")["Time to Adoptions (Days)"]
    )

with col2:
    st.dataframe(
        age_stats[["Age Group at Intake", "Intake Count", "Adoption Rate"]],
        hide_index=True
    )


# -----Adoption by name presence-----
st.subheader("Adoption by Name Presence at Intake")

name_stats = (
    filtered[filtered["has_outcome"]]
    .groupby("has_name")["is_adopted"]
    .agg(["count", "mean"])
    .reset_index()
    .sort_values(by="count", ascending=False)
)

name_stats["has_name"] = name_stats["has_name"].map(
    {True: "Has Name", False: "No Name"}
)

name_stats.columns = ["Has Name", 
                     "Intake Count", 
                     "Adoption Rate"]

st.write(
    """
    Cats with names at intake show higher adoption rates.
    
    **Adoption Rate**
    """
)

col1, col2 = st.columns([5, 2], gap="medium")

with col1:
    st.bar_chart(
        name_stats.set_index("Has Name")["Adoption Rate"]
    )

with col2:
    st.dataframe(
        name_stats[["Has Name", "Intake Count"]],
        hide_index=True
    )


# -----Length of stay (LOS)-----
st.subheader("Time to Adoption")

st.write("**Distribution of Time to Adoption (Days) within 90 Days**")

st.line_chart(
    adopted_df[adopted_df["length_of_stay_days"]<=90]
    ["length_of_stay_days"]
    .value_counts()
    .sort_index()
)

# -----Seasonality-----
st.subheader("Seasonality of Intakes")

st.write(
    """
    Intake volume surges in spring and summer (kitten season), 
    and drops in fall and winter.
    
    **Monthly Intake Volume**
    """
)
monthly_intakes = (
    filtered
    .groupby(["month_year_intake", "age_group_intake"])
    ["animal_id"]
    .count()
    .reset_index()
)

monthly_intakes_age = (
    monthly_intakes
    .pivot(
        index="month_year_intake", 
        columns="age_group_intake", 
        values="animal_id")
)

st.line_chart(
    monthly_intakes_age
)


# -----Recommendation-----
st.subheader("Summary & Recommendation")

st.write(
    """
    Kittens are adopted faster and at higher rates, with intake volume 
    peaking in spring and summer.
    
    Cats with names at intake show higher adoption rates.
    
    I suggest **starting with a small trial of naming at intake**, measuring 
    success by *30-day adoption rate*, and monitoring *length of stay* before 
    expanding it more widely.
    """
)
