"""
01_eda.py

Tasks covered:
1. Load and profile Titanic dataset
2. Missing-value handling
3. Univariate analysis and outliers
4. Bivariate analysis and correlation
5. Multivariate data story
6. Standardization sanity check

IMPORTANT:
sns.load_dataset("titanic") is called exactly ONCE.
The raw dataset is immediately saved as titanic.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

RANDOM_STATE = 42


# ============================================================
# Task 1 — Load and profile dataset
# ============================================================

print("=" * 80)
print("TASK 1 — LOAD AND PROFILE DATASET")
print("=" * 80)

# ONE AND ONLY ONE network/cache load
df = sns.load_dataset("titanic")

print("\nDataFrame information:")
df.info()

print("\nDescriptive statistics:")
print(df.describe(include="all"))

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# Immediately create offline fallback
# ------------------------------------------------------------

csv_path = BASE_DIR / "titanic.csv"

df.to_csv(csv_path, index=False)

print(f"\nRaw dataset saved to: {csv_path}")


# ------------------------------------------------------------
# Missing-value percentages
# ------------------------------------------------------------

missing_percentage = (
    df.isnull()
      .mean()
      .mul(100)
      .sort_values(ascending=False)
)

missing_percentage = missing_percentage[missing_percentage > 0]

print("\nMissing-value percentage for affected columns:")
print(missing_percentage.to_string())


# Save missing-value report
missing_report = missing_percentage.reset_index()
missing_report.columns = ["column", "missing_percentage"]

missing_report.to_csv(
    OUTPUT_DIR / "missing_value_report.csv",
    index=False
)


# ============================================================
# Task 2 — Missing-value handling
# ============================================================

print("\n" + "=" * 80)
print("TASK 2 — MISSING VALUE HANDLING")
print("=" * 80)

cleaned_df = df.copy()

print("\nMissing-value strategy:")

for column, percentage in missing_percentage.items():

    if percentage < 5:
        print(
            f"{column}: {percentage:.2f}% missing -> "
            "DROP rows containing missing values."
        )

    elif 5 <= percentage <= 30:
        print(
            f"{column}: {percentage:.2f}% missing -> "
            "IMPUTE using median/mode."
        )

    else:
        print(
            f"{column}: {percentage:.2f}% missing -> "
            "DROP COLUMN because missingness is too high "
            "for reliable imputation."
        )


# ------------------------------------------------------------
# High-missing column
# ------------------------------------------------------------

# deck has very high missingness (~77%), so it is removed.
if "deck" in cleaned_df.columns:
    cleaned_df = cleaned_df.drop(columns=["deck"])

print(
    "\n'deck' removed because its missing percentage is "
    f"{missing_percentage.get('deck', 0):.2f}%."
)


# ------------------------------------------------------------
# Under 5% missing: drop rows
# ------------------------------------------------------------

low_missing_columns = [
    column
    for column, percentage in missing_percentage.items()
    if percentage < 5
]

if low_missing_columns:
    before = len(cleaned_df)

    cleaned_df = cleaned_df.dropna(
        subset=[
            column
            for column in low_missing_columns
            if column in cleaned_df.columns
        ]
    )

    after = len(cleaned_df)

    print(
        f"\nDropped {before - after} rows because of "
        f"missing values in: {low_missing_columns}"
    )


# ------------------------------------------------------------
# 5%–30% missing: median imputation for numeric columns
# ------------------------------------------------------------

medium_missing_columns = [
    column
    for column, percentage in missing_percentage.items()
    if 5 <= percentage <= 30
]

for column in medium_missing_columns:

    if column not in cleaned_df.columns:
        continue

    if pd.api.types.is_numeric_dtype(cleaned_df[column]):

        median_value = cleaned_df[column].median()

        cleaned_df[column] = cleaned_df[column].fillna(
            median_value
        )

        print(
            f"Imputed {column} with median = "
            f"{median_value:.2f}"
        )

    else:

        mode_value = cleaned_df[column].mode()[0]

        cleaned_df[column] = cleaned_df[column].fillna(
            mode_value
        )

        print(
            f"Imputed {column} with mode = "
            f"{mode_value}"
        )


print("\nRemaining missing values:")
print(
    cleaned_df.isnull()
              .sum()
              .sort_values(ascending=False)
              .head(20)
)


print("\nCleaned dataset shape:")
print(cleaned_df.shape)


# ============================================================
# Task 3 — Univariate analysis
# ============================================================

print("\n" + "=" * 80)
print("TASK 3 — UNIVARIATE ANALYSIS")
print("=" * 80)


# ------------------------------------------------------------
# Age histogram
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    cleaned_df["age"],
    bins=30,
    kde=True
)

plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "age_histogram.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Age boxplot
# ------------------------------------------------------------

plt.figure(figsize=(8, 4))

sns.boxplot(
    x=cleaned_df["age"]
)

plt.title("Box Plot of Age")
plt.xlabel("Age")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "age_boxplot.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Fare histogram
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    cleaned_df["fare"],
    bins=40,
    kde=True
)

plt.title("Distribution of Fare")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "fare_histogram.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Fare boxplot
# ------------------------------------------------------------

plt.figure(figsize=(8, 4))

sns.boxplot(
    x=cleaned_df["fare"]
)

plt.title("Box Plot of Fare")
plt.xlabel("Fare")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "fare_boxplot.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# IQR outlier function
# ------------------------------------------------------------

def iqr_outlier_count(series):
    """
    Count observations outside:
    [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    """

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = (
        (series < lower_bound) |
        (series > upper_bound)
    )

    return (
        int(outliers.sum()),
        q1,
        q3,
        iqr,
        lower_bound,
        upper_bound
    )


age_result = iqr_outlier_count(cleaned_df["age"])
fare_result = iqr_outlier_count(cleaned_df["fare"])


print("\nAge IQR analysis:")
print(f"Q1: {age_result[1]:.2f}")
print(f"Q3: {age_result[2]:.2f}")
print(f"IQR: {age_result[3]:.2f}")
print(f"Lower bound: {age_result[4]:.2f}")
print(f"Upper bound: {age_result[5]:.2f}")
print(f"Number of outliers: {age_result[0]}")


print("\nFare IQR analysis:")
print(f"Q1: {fare_result[1]:.2f}")
print(f"Q3: {fare_result[2]:.2f}")
print(f"IQR: {fare_result[3]:.2f}")
print(f"Lower bound: {fare_result[4]:.2f}")
print(f"Upper bound: {fare_result[5]:.2f}")
print(f"Number of outliers: {fare_result[0]}")


# ------------------------------------------------------------
# Fare mean, median, mode
# ------------------------------------------------------------

fare_mean = cleaned_df["fare"].mean()
fare_median = cleaned_df["fare"].median()
fare_mode = cleaned_df["fare"].mode().iloc[0]

print("\nFare statistics:")
print(f"Mean   : {fare_mean:.4f}")
print(f"Median : {fare_median:.4f}")
print(f"Mode   : {fare_mode:.4f}")


if fare_mean > fare_median > fare_mode:
    fare_skew = "right-skewed"

elif fare_mean < fare_median < fare_mode:
    fare_skew = "left-skewed"

else:
    fare_skew = "approximately symmetric"


print(
    f"\nFare distribution is {fare_skew} "
    "because of the ordering of mean, median and mode."
)


# ============================================================
# Task 4 — Bivariate analysis
# ============================================================

print("\n" + "=" * 80)
print("TASK 4 — BIVARIATE ANALYSIS")
print("=" * 80)


# ------------------------------------------------------------
# Survival rate by sex
# ------------------------------------------------------------

survival_by_sex = (
    cleaned_df
    .groupby("sex")["survived"]
    .mean()
    .mul(100)
    .reset_index(name="survival_rate_percent")
)

print("\nSurvival rate by sex:")
print(survival_by_sex)


# ------------------------------------------------------------
# Survival rate by pclass
# ------------------------------------------------------------

survival_by_pclass = (
    cleaned_df
    .groupby("pclass")["survived"]
    .mean()
    .mul(100)
    .reset_index(name="survival_rate_percent")
)

print("\nSurvival rate by pclass:")
print(survival_by_pclass)


# ------------------------------------------------------------
# Survival rate by sex AND pclass
# ------------------------------------------------------------

survival_by_sex_class = (
    cleaned_df
    .groupby(["sex", "pclass"])["survived"]
    .mean()
    .mul(100)
    .reset_index(name="survival_rate_percent")
)

print("\nSurvival rate by sex and pclass:")
print(survival_by_sex_class)


# ------------------------------------------------------------
# Boolean masking examples
# ------------------------------------------------------------

female_first_class = cleaned_df[
    (cleaned_df["sex"] == "female") &
    (cleaned_df["pclass"] == 1)
]

male_third_class = cleaned_df[
    (cleaned_df["sex"] == "male") &
    (cleaned_df["pclass"] == 3)
]

print(
    "\nFemale + first-class survival rate:",
    female_first_class["survived"].mean() * 100
)

print(
    "Male + third-class survival rate:",
    male_third_class["survived"].mean() * 100
)


# ------------------------------------------------------------
# Correlation matrix
# EXACTLY six requested columns
# ------------------------------------------------------------

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = cleaned_df[
    correlation_columns
].corr()

print("\nCorrelation matrix:")
print(correlation_matrix)


# ------------------------------------------------------------
# Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True
)

plt.title("Titanic Numeric Feature Correlation Matrix")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Find two strongest correlations
# ------------------------------------------------------------

corr_abs = correlation_matrix.abs()

# Remove diagonal
np.fill_diagonal(
    corr_abs.values,
    np.nan
)

pairs = []

for i in range(len(correlation_columns)):
    for j in range(i + 1, len(correlation_columns)):

        feature_1 = correlation_columns[i]
        feature_2 = correlation_columns[j]

        coefficient = correlation_matrix.loc[
            feature_1,
            feature_2
        ]

        pairs.append(
            {
                "feature_1": feature_1,
                "feature_2": feature_2,
                "correlation": coefficient,
                "absolute_correlation": abs(coefficient)
            }
        )

strongest_pairs = (
    pd.DataFrame(pairs)
    .sort_values(
        "absolute_correlation",
        ascending=False
    )
    .head(2)
)

print("\nTwo strongest correlations:")
print(strongest_pairs)


strongest_pairs.to_csv(
    OUTPUT_DIR / "strongest_correlations.csv",
    index=False
)


# ============================================================
# Task 5 — Multivariate data story
# ============================================================

print("\n" + "=" * 80)
print("TASK 5 — MULTIVARIATE DATA STORY")
print("=" * 80)


# ------------------------------------------------------------
# Chart 1 — Survival by sex
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=cleaned_df,
    x="sex",
    y="survived",
    errorbar=None
)

plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "survival_by_sex.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Chart 2 — Survival by passenger class
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=cleaned_df,
    x="pclass",
    y="survived",
    errorbar=None
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "survival_by_class.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Chart 3 — Sex + class survival heatmap
# ------------------------------------------------------------

sex_class_pivot = cleaned_df.pivot_table(
    index="sex",
    columns="pclass",
    values="survived",
    aggfunc="mean"
)

plt.figure(figsize=(8, 5))

sns.heatmap(
    sex_class_pivot,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    vmin=0,
    vmax=1
)

plt.title("Survival Rate by Sex and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Sex")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "survival_sex_class_heatmap.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Chart 4 — Age vs fare by survival
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=cleaned_df,
    x="age",
    y="fare",
    hue="survived",
    alpha=0.7
)

plt.title("Age vs Fare by Survival")
plt.xlabel("Age")
plt.ylabel("Fare")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "age_fare_survival.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Additional chart — Fare by survival
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=cleaned_df,
    x="survived",
    y="fare"
)

plt.title("Fare Distribution by Survival")
plt.xlabel("Survived")
plt.ylabel("Fare")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "fare_by_survival.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Additional chart — Age by survival
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=cleaned_df,
    x="survived",
    y="age"
)

plt.title("Age Distribution by Survival")
plt.xlabel("Survived")
plt.ylabel("Age")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "age_by_survival.png",
    dpi=300
)

plt.close()


# ============================================================
# Task 6 — Standardization sanity check
# ============================================================

print("\n" + "=" * 80)
print("TASK 6 — STANDARDIZATION CHECK")
print("=" * 80)


standardization_df = cleaned_df.copy()

for column in ["age", "fare"]:

    mean_value = standardization_df[column].mean()
    std_value = standardization_df[column].std()

    standardization_df[f"{column}_z"] = (
        standardization_df[column] - mean_value
    ) / std_value


before_after = pd.DataFrame(
    {
        "age_before_mean": [
            standardization_df["age"].mean()
        ],
        "age_before_std": [
            standardization_df["age"].std()
        ],
        "age_after_mean": [
            standardization_df["age_z"].mean()
        ],
        "age_after_std": [
            standardization_df["age_z"].std()
        ],
        "fare_before_mean": [
            standardization_df["fare"].mean()
        ],
        "fare_before_std": [
            standardization_df["fare"].std()
        ],
        "fare_after_mean": [
            standardization_df["fare_z"].mean()
        ],
        "fare_after_std": [
            standardization_df["fare_z"].std()
        ]
    }
)

print("\nBefore/after standardization:")
print(before_after.T)

before_after.to_csv(
    OUTPUT_DIR / "standardization_summary.csv",
    index=False
)


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

sns.histplot(
    standardization_df["age_z"],
    kde=True,
    ax=axes[0]
)

axes[0].set_title("Standardized Age")
axes[0].set_xlabel("Age z-score")

sns.histplot(
    standardization_df["fare_z"],
    kde=True,
    ax=axes[1]
)

axes[1].set_title("Standardized Fare")
axes[1].set_xlabel("Fare z-score")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "standardization_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# Save cleaned data
# ============================================================

# This is the same dataset that the modeling stage will consume.
cleaned_df.to_csv(
    BASE_DIR / "titanic_cleaned.csv",
    index=False
)

print(
    "\nCleaned dataset saved to "
    f"{BASE_DIR / 'titanic_cleaned.csv'}"
)


print("\nEDA completed successfully.")