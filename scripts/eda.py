"""
Exploratory Data Analysis (EDA) for the M&A acquisitions dataset.

This script:
- Loads the full Crunchbase-derived dataset from data/data.csv
- Computes descriptive statistics for key business variables
- Creates a set of Seaborn visualizations to understand:
    * Acquisition status distribution
    * Relationship between funding and acquisition probability
    * Impact of company age on acquisition
    * Acquisition rate across major sectors

Outputs:
- A summary CSV file with basic statistics.
- PNG plots saved under reports/eda/.
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(ROOT_DIR, "data", "data.csv")
REPORT_DIR = os.path.join(ROOT_DIR, "reports", "eda")

os.makedirs(REPORT_DIR, exist_ok=True)


def run_eda() -> None:
    df = pd.read_csv(DATA_PATH)

    # Basic summary statistics
    summary = df.describe(include="all")
    summary.to_csv(os.path.join(REPORT_DIR, "summary_statistics.csv"))

    # 1. Acquisition vs non-acquisition counts
    plt.figure()
    sns.countplot(data=df, x="is_acquired")
    plt.title("Acquisition Outcome Distribution")
    plt.xlabel("Is acquired")
    plt.ylabel("Number of companies")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "acquisition_distribution.png"))
    plt.close()

    # 2. Funding vs acquisition rate
    plt.figure()
    sns.boxplot(data=df, x="is_acquired", y="average_funded")
    plt.title("Average Funding vs Acquisition Outcome")
    plt.xlabel("Is acquired")
    plt.ylabel("Average funding (USD)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "funding_vs_acquisition.png"))
    plt.close()

    # 3. Company age vs acquisition
    plt.figure()
    sns.boxplot(data=df, x="is_acquired", y="age")
    plt.title("Company Age vs Acquisition Outcome")
    plt.xlabel("Is acquired")
    plt.ylabel("Company age (years)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "age_vs_acquisition.png"))
    plt.close()

    # 4. Sector-level acquisition rate (top categories)
    if "category_code" in df.columns:
        top_categories = (
            df["category_code"]
            .value_counts()
            .head(10)
            .index
            .tolist()
        )
        subset = df[df["category_code"].isin(top_categories)]

        plt.figure()
        sns.barplot(
            data=subset,
            x="category_code",
            y="is_acquired",
            estimator=lambda x: sum(x) / len(x),
        )
        plt.xticks(rotation=45, ha="right")
        plt.title("Acquisition Rate by Sector (Top 10 Categories)")
        plt.xlabel("Category")
        plt.ylabel("Acquisition rate")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, "acquisition_rate_by_sector.png"))
        plt.close()

    print(f"EDA complete. Reports written to {REPORT_DIR}")


if __name__ == "__main__":  # pragma: no cover
    run_eda()
