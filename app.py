"""
Streamlit UI for M&A Acquisition Prediction.

This lightweight web app allows a user to:
- Enter key company characteristics (funding, sector, geography, age, etc.).
- Obtain a model-backed estimate of the probability of being acquired.
- Inspect the prediction along with a brief interpretation.

To run locally:
    streamlit run app.py
"""

import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")


@st.cache_resource
def load_artifacts() -> Tuple[object, object, pd.DataFrame]:
    """Load pre-fitted preprocessor, trained model and training dataframe."""
    preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
    model = joblib.load(os.path.join(MODEL_DIR, "acquisition_model.joblib"))
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
    return preprocessor, model, train_df


def main() -> None:
    st.set_page_config(page_title="M&A Acquisition Predictor", layout="centered")
    st.title("Mergers & Acquisitions – Acquisition Probability Estimator")
    st.write(
        """Interactive demo built on a Random Forest model trained over a large
        Crunchbase company dataset (this project uses a sampled subset)."""
    )

    preprocessor, model, train_df = load_artifacts()

    st.sidebar.header("Configuration")
    st.sidebar.write(
        "Provide company details in the main panel and click **Predict** "
        "to estimate acquisition likelihood."
    )

    st.header("Company profile")

    # Helper lists for categorical inputs from training data
    categories = sorted(train_df["category_code"].dropna().unique().tolist())
    countries = sorted(train_df["country_code"].dropna().unique().tolist())
    states = sorted(train_df["state_code"].dropna().unique().tolist())

    col1, col2 = st.columns(2)

    with col1:
        category_code = st.selectbox("Category / Sector", options=categories)
        country_code = st.selectbox("Country code", options=countries)
        state_code = st.selectbox("State / Region", options=states)
        ipo = st.checkbox("Company is already IPO'd", value=False)

    with col2:
        average_funded = st.number_input(
            "Average funding per round (USD)",
            min_value=0.0,
            value=float(train_df["average_funded"].median()),
            step=100000.0,
        )
        total_rounds = st.number_input(
            "Total funding rounds",
            min_value=0,
            value=int(train_df["total_rounds"].median()),
            step=1,
        )
        average_participants = st.number_input(
            "Average investors per round",
            min_value=0.0,
            value=float(train_df["average_participants"].median()),
            step=1.0,
        )
        age = st.number_input(
            "Company age (years)",
            min_value=0.0,
            value=float(train_df["age"].median()),
            step=1.0,
        )

    st.subheader("Operational footprint and founding team")

    col3, col4 = st.columns(2)
    with col3:
        products_number = st.number_input(
            "Number of products", min_value=0.0, value=1.0, step=1.0
        )
        offices = st.number_input(
            "Number of offices", min_value=0.0, value=1.0, step=1.0
        )
        acquired_companies = st.number_input(
            "Number of companies already acquired",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )
    with col4:
        mba_degree = st.number_input(
            "Founders / execs with MBA", min_value=0.0, value=1.0, step=1.0
        )
        phd_degree = st.number_input(
            "Founders / execs with PhD", min_value=0.0, value=0.0, step=1.0
        )
        ms_degree = st.number_input(
            "Founders / execs with MS", min_value=0.0, value=1.0, step=1.0
        )
        other_degree = st.number_input(
            "Founders / execs with other degrees",
            min_value=0.0,
            value=2.0,
            step=1.0,
        )

    st.subheader("Prediction")

    if st.button("Predict acquisition likelihood"):
        # Build single-row DataFrame consistent with training schema
        row = {
            "company_id": "interactive_company",
            "category_code": category_code,
            "country_code": country_code,
            "state_code": state_code,
            "average_funded": average_funded,
            "total_rounds": total_rounds,
            "average_participants": average_participants,
            "products_number": products_number,
            "offices": offices,
            "acquired_companies": acquired_companies,
            "mba_degree": mba_degree,
            "phd_degree": phd_degree,
            "ms_degree": ms_degree,
            "other_degree": other_degree,
            "ipo": bool(ipo),
            "is_closed": False,  # assume operating company
            "age": age,
        }

        X_new = pd.DataFrame([row])
        Xt_new = preprocessor.transform(X_new)
        proba = model.predict_proba(Xt_new)[0, 1]
        pred_label = "High" if proba >= 0.5 else "Low"

        st.metric(
            label="Estimated acquisition probability",
            value=f"{proba * 100:.1f}%",
            delta=None,
        )

        st.write(
            f"According to the model, this company has a **{pred_label}** "
            f"likelihood of being acquired in the future (threshold 50%)."
        )
        st.info(
            "This estimate is based on historical Crunchbase data and should be "
            "used as a directional signal alongside domain expertise, not as "
            "financial advice."
        )

    st.markdown("---")
    st.caption("M&A Acquisition Predictor – educational / academic project")


if __name__ == "__main__":  # pragma: no cover
    main()
