"""
02_modeling.py

Tasks covered:
7. Stratified train/test split
8. Leakage-safe preprocessing
9. Three classifiers
10. Full classification evaluation
11. Imbalance comparison
12. Random Forest GridSearchCV + OOB
13. Fare regression
14. Final model comparison + recommendation
15. Save/reload complete pipeline
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# Load cleaned dataset
# ============================================================

print("=" * 80)
print("LOADING CLEANED DATASET")
print("=" * 80)

data_path = BASE_DIR / "titanic_cleaned.csv"

df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# Task 7 — Stratified train/test split
# ============================================================

print("\n" + "=" * 80)
print("TASK 7 — STRATIFIED TRAIN/TEST SPLIT")
print("=" * 80)


TARGET = "survived"

FEATURES = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]


X = df[FEATURES].copy()
y = df[TARGET].copy()


print("\nOverall class balance:")
print(
    y.value_counts(normalize=True)
     .sort_index()
     .mul(100)
     .rename("percentage")
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)

print("\nTraining class balance:")
print(
    y_train.value_counts(normalize=True)
          .sort_index()
          .mul(100)
)

print("\nTesting class balance:")
print(
    y_test.value_counts(normalize=True)
          .sort_index()
          .mul(100)
)


# ============================================================
# Task 8 — Preprocessing
# ============================================================

print("\n" + "=" * 80)
print("TASK 8 — TRAINING-ONLY PREPROCESSING")
print("=" * 80)


NUMERIC_FEATURES = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

CATEGORICAL_FEATURES = [
    "sex",
    "embarked"
]


# ------------------------------------------------------------
# Numeric preprocessing
# ------------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ------------------------------------------------------------
# Categorical preprocessing
# ------------------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ------------------------------------------------------------
# ColumnTransformer
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ]
)


print(
    "\nPreprocessing consists of:\n"
    "- Median imputation + StandardScaler for numeric features\n"
    "- Most-frequent imputation + OneHotEncoder for categorical features\n"
    "- Fitted only on training data through Pipeline"
)


# ============================================================
# Task 9 — Train three classifiers
# ============================================================

print("\n" + "=" * 80)
print("TASK 9 — THREE CLASSIFIERS")
print("=" * 80)


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=5
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE
    )
}


trained_models = {}


for model_name, estimator in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                estimator
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    trained_models[model_name] = pipeline

    print(f"{model_name} trained successfully.")


# ============================================================
# Task 9 — Decision Tree visualization
# ============================================================

print("\nCreating Decision Tree visualization...")


decision_tree_pipeline = trained_models["Decision Tree"]

fitted_preprocessor = (
    decision_tree_pipeline
    .named_steps["preprocessor"]
)

decision_tree_model = (
    decision_tree_pipeline
    .named_steps["model"]
)


feature_names = fitted_preprocessor.get_feature_names_out()

plt.figure(figsize=(24, 14))

plot_tree(
    decision_tree_model,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree for Titanic Survival")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "decision_tree.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Task 10 — Evaluation helper
# ============================================================

def evaluate_classifier(
    name,
    pipeline,
    X_test,
    y_test
):

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(
        y_test,
        predictions
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "probabilities": probabilities,
        "predictions": predictions
    }


# ============================================================
# Task 10 — Evaluate all models
# ============================================================

print("\n" + "=" * 80)
print("TASK 10 — MODEL EVALUATION")
print("=" * 80)


evaluation_results = {}

for name, pipeline in trained_models.items():

    result = evaluate_classifier(
        name,
        pipeline,
        X_test,
        y_test
    )

    evaluation_results[name] = result

    print(f"\n{name}")

    print(
        "Confusion Matrix:\n",
        result["confusion_matrix"]
    )

    print(
        f"Accuracy : {result['accuracy']:.4f}"
    )

    print(
        f"Precision: {result['precision']:.4f}"
    )

    print(
        f"Recall   : {result['recall']:.4f}"
    )

    print(
        f"F1 Score : {result['f1']:.4f}"
    )

    print(
        f"AUC      : {result['auc']:.4f}"
    )


# ------------------------------------------------------------
# Classification comparison table
# ------------------------------------------------------------

classification_metrics = pd.DataFrame(
    [
        {
            "Model": name,
            "Accuracy": result["accuracy"],
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1": result["f1"],
            "AUC": result["auc"]
        }

        for name, result in evaluation_results.items()
    ]
)

print("\nClassification comparison:")
print(classification_metrics)

classification_metrics.to_csv(
    OUTPUT_DIR / "classification_metrics.csv",
    index=False
)


# ============================================================
# Task 10 — Confusion matrices
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4)
)


for ax, (name, result) in zip(
    axes,
    evaluation_results.items()
):

    sns.heatmap(
        result["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrices.png",
    dpi=300
)

plt.close()


# ============================================================
# Task 10 — ROC curves
# ============================================================

plt.figure(figsize=(9, 7))


for name, result in evaluation_results.items():

    fpr, tpr, _ = roc_curve(
        y_test,
        result["probabilities"]
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={result['auc']:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curves")

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roc_curves.png",
    dpi=300
)

plt.close()


# ============================================================
# Task 11 — Class imbalance
# ============================================================

print("\n" + "=" * 80)
print("TASK 11 — IMBALANCE COMPARISON")
print("=" * 80)


class_balance = (
    y.value_counts()
     .rename(index={
         0: "Not Survived",
         1: "Survived"
     })
)

print("\nClass counts:")
print(class_balance)

print("\nClass percentages:")
print(
    y.value_counts(normalize=True)
     .mul(100)
     .rename(index={
         0: "Not Survived",
         1: "Survived"
     })
)


# ------------------------------------------------------------
# Use Logistic Regression for imbalance comparison
# ------------------------------------------------------------

# A fresh preprocessor is created so each pipeline has
# completely independent fitted preprocessing.

def create_preprocessor():

    numeric = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical,
                CATEGORICAL_FEATURES
            )
        ]
    )


# ------------------------------------------------------------
# A. Baseline
# ------------------------------------------------------------

baseline_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ]
)

baseline_pipeline.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# B. class_weight='balanced'
# ------------------------------------------------------------

balanced_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]
)

balanced_pipeline.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# C. SMOTE
# ------------------------------------------------------------

smote_pipeline = ImbPipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "smote",
            SMOTE(
                random_state=RANDOM_STATE
            )
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ]
)

smote_pipeline.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# Evaluate imbalance variants
# ------------------------------------------------------------

imbalance_models = {
    "Baseline": baseline_pipeline,
    "Class Weight Balanced": balanced_pipeline,
    "SMOTE": smote_pipeline
}

imbalance_results = []


for name, pipeline in imbalance_models.items():

    predictions = pipeline.predict(X_test)

    imbalance_results.append(
        {
            "Strategy": name,

            "Precision": precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "Recall": recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "F1": f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        }
    )


imbalance_df = pd.DataFrame(
    imbalance_results
)

print("\nImbalance comparison:")
print(imbalance_df)

imbalance_df.to_csv(
    OUTPUT_DIR / "imbalance_comparison.csv",
    index=False
)


# Automatically identify best F1 strategy
best_imbalance = (
    imbalance_df
    .sort_values("F1", ascending=False)
    .iloc[0]
)

print(
    "\nBest imbalance strategy based on F1:",
    best_imbalance["Strategy"]
)


# ============================================================
# Task 12 — Random Forest GridSearchCV
# ============================================================

print("\n" + "=" * 80)
print("TASK 12 — RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 80)


rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "model",
            RandomForestClassifier(
                oob_score=True,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


param_grid = {
    "model__n_estimators": [
        100,
        200
    ],

    "model__max_depth": [
        None,
        5,
        10
    ],

    "model__max_features": [
        "sqrt",
        "log2"
    ]
}


grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    return_train_score=True
)


grid_search.fit(
    X_train,
    y_train
)


print("\nBest parameters:")
print(grid_search.best_params_)

print(
    "\nBest cross-validation F1:",
    grid_search.best_score_
)


# ------------------------------------------------------------
# Refit best configuration on complete training set
# ------------------------------------------------------------

best_params = grid_search.best_params_


tuned_rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=best_params[
                    "model__n_estimators"
                ],

                max_depth=best_params[
                    "model__max_depth"
                ],

                max_features=best_params[
                    "model__max_features"
                ],

                oob_score=True,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


tuned_rf_pipeline.fit(
    X_train,
    y_train
)


oob_score = (
    tuned_rf_pipeline
    .named_steps["model"]
    .oob_score_
)

print(
    "\nRandom Forest OOB score:",
    oob_score
)


with open(
    OUTPUT_DIR / "random_forest_tuning.txt",
    "w"
) as file:

    file.write(
        f"Best parameters: {best_params}\n"
    )

    file.write(
        f"Best CV F1: {grid_search.best_score_:.6f}\n"
    )

    file.write(
        f"OOB score: {oob_score:.6f}\n"
    )


# ============================================================
# Task 13 — Multivariate linear regression
# ============================================================

print("\n" + "=" * 80)
print("TASK 13 — FARE REGRESSION")
print("=" * 80)


REGRESSION_TARGET = "fare"


# Use the same cleaned dataset.
# fare is predicted from other available features.

regression_features = [
    "survived",
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked"
]


X_reg = df[regression_features].copy()
y_reg = df[REGRESSION_TARGET].copy()


X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=RANDOM_STATE
    )
)


REG_NUMERIC_FEATURES = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch"
]


REG_CATEGORICAL_FEATURES = [
    "sex",
    "embarked"
]


reg_numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


reg_categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            reg_numeric_pipeline,
            REG_NUMERIC_FEATURES
        ),
        (
            "categorical",
            reg_categorical_pipeline,
            REG_CATEGORICAL_FEATURES
        )
    ]
)


regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            reg_preprocessor
        ),
        (
            "model",
            LinearRegression()
        )
    ]
)


regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)


fare_predictions = regression_pipeline.predict(
    X_reg_test
)


# ------------------------------------------------------------
# Regression metrics
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_reg_test,
    fare_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        fare_predictions
    )
)

r2 = r2_score(
    y_reg_test,
    fare_predictions
)


# Number of observations
n = len(y_reg_test)

# Number of predictors after preprocessing
X_reg_test_transformed = (
    regression_pipeline
    .named_steps["preprocessor"]
    .transform(X_reg_test)
)

p = X_reg_test_transformed.shape[1]


adjusted_r2 = (
    1
    - (
        (1 - r2) * (n - 1)
        / (n - p - 1)
    )
)


print("\nRegression metrics:")
print(f"MAE        : {mae:.4f}")
print(f"RMSE       : {rmse:.4f}")
print(f"R²         : {r2:.4f}")
print(f"Adjusted R²: {adjusted_r2:.4f}")


regression_metrics = pd.DataFrame(
    [
        {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Adjusted_R2": adjusted_r2
        }
    ]
)


regression_metrics.to_csv(
    OUTPUT_DIR / "regression_metrics.csv",
    index=False
)


# ------------------------------------------------------------
# Residual plot
# ------------------------------------------------------------

residuals = (
    y_reg_test.values
    - fare_predictions
)


plt.figure(figsize=(9, 6))

sns.scatterplot(
    x=fare_predictions,
    y=residuals,
    alpha=0.7
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Fare Regression Residual Plot")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "fare_residuals.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Heteroscedasticity check
# ------------------------------------------------------------

abs_residual_correlation = np.corrcoef(
    fare_predictions,
    np.abs(residuals)
)[0, 1]


print(
    "\nCorrelation between predicted fare and "
    "absolute residuals:",
    abs_residual_correlation
)


if abs_residual_correlation > 0.30:

    heteroscedasticity_conclusion = (
        "The residual plot suggests possible "
        "heteroscedasticity because the residual spread "
        "increases with predicted fare."
    )

else:

    heteroscedasticity_conclusion = (
        "The residual plot does not show strong evidence "
        "of heteroscedasticity; the residual spread is "
        "reasonably random."
    )


print("\nHeteroscedasticity conclusion:")
print(heteroscedasticity_conclusion)


# ============================================================
# Task 14 — Final model comparison
# ============================================================

print("\n" + "=" * 80)
print("TASK 14 — FINAL MODEL COMPARISON")
print("=" * 80)


# Classification metrics

classification_for_comparison = (
    classification_metrics.copy()
)

classification_for_comparison.columns = [
    "Classifier",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUC"
]


# Regression metrics are kept as a separate metric group.

final_comparison = classification_for_comparison.copy()


# Add regression metrics as separate columns.
final_comparison["Regression_MAE"] = mae
final_comparison["Regression_RMSE"] = rmse
final_comparison["Regression_R2"] = r2
final_comparison["Regression_Adjusted_R2"] = adjusted_r2


print("\nFinal model comparison:")
print(final_comparison)


final_comparison.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)


# ------------------------------------------------------------
# Choose best classifier
# ------------------------------------------------------------

best_classifier_row = (
    classification_for_comparison
    .sort_values(
        by=["F1", "AUC"],
        ascending=False
    )
    .iloc[0]
)


best_classifier_name = (
    best_classifier_row["Classifier"]
)


best_f1 = best_classifier_row["F1"]
best_auc = best_classifier_row["AUC"]
best_accuracy = best_classifier_row["Accuracy"]
best_precision = best_classifier_row["Precision"]
best_recall = best_classifier_row["Recall"]


print(
    f"\nRecommended classifier: "
    f"{best_classifier_name}"
)

print(
    f"Accuracy : {best_accuracy:.4f}"
)

print(
    f"Precision: {best_precision:.4f}"
)

print(
    f"Recall   : {best_recall:.4f}"
)

print(
    f"F1       : {best_f1:.4f}"
)

print(
    f"AUC      : {best_auc:.4f}"
)


# ============================================================
# Task 15 — Save complete pipeline
# ============================================================

print("\n" + "=" * 80)
print("TASK 15 — SAVE COMPLETE PIPELINE")
print("=" * 80)


# ------------------------------------------------------------
# Select final pipeline
# ------------------------------------------------------------

# We use the best classifier based on F1/AUC.
#
# Importantly, the saved object includes:
#   preprocessing
#   imputation
#   encoding
#   scaling
#   final estimator

if best_classifier_name == "Logistic Regression":

    final_pipeline = trained_models[
        "Logistic Regression"
    ]

elif best_classifier_name == "Decision Tree":

    final_pipeline = trained_models[
        "Decision Tree"
    ]

else:

    final_pipeline = trained_models[
        "Random Forest"
    ]


model_path = (
    MODEL_DIR /
    "titanic_best_pipeline.joblib"
)


joblib.dump(
    final_pipeline,
    model_path
)


print(
    f"\nComplete pipeline saved to:\n"
    f"{model_path}"
)


# ============================================================
# Reload saved pipeline
# ============================================================

loaded_pipeline = joblib.load(
    model_path
)


print("\nSaved pipeline successfully reloaded.")


# ------------------------------------------------------------
# Confirm prediction using RAW input
# ------------------------------------------------------------

raw_sample = pd.DataFrame(
    [
        {
            "pclass": 1,
            "sex": "female",
            "age": 30,
            "sibsp": 0,
            "parch": 0,
            "fare": 100,
            "embarked": "S"
        }
    ]
)


sample_prediction = loaded_pipeline.predict(
    raw_sample
)

sample_probability = (
    loaded_pipeline
    .predict_proba(raw_sample)[:, 1]
)


print("\nRaw input:")
print(raw_sample)

print(
    "\nPredicted survival:",
    sample_prediction[0]
)

print(
    "Predicted survival probability:",
    sample_probability[0]
)


# ============================================================
# Create final recommendation text
# ============================================================

recommendation = f"""
Final Recommendation

The recommended classifier is {best_classifier_name}. It achieved
an accuracy of {best_accuracy:.3f}, precision of {best_precision:.3f},
recall of {best_recall:.3f}, F1 score of {best_f1:.3f}, and ROC-AUC
of {best_auc:.3f} on the held-out test set.

The recommendation prioritizes F1 and AUC because the objective is
to balance correctly identifying survivors with avoiding incorrect
predictions. The final pipeline also preserves the complete
preprocessing workflow, including imputation, categorical encoding,
and numerical scaling, so it can accept raw passenger data directly.

The fare regression model achieved MAE of {mae:.3f}, RMSE of {rmse:.3f},
R² of {r2:.3f}, and adjusted R² of {adjusted_r2:.3f}. These regression
metrics are reported separately from the classification metrics
because they measure a different predictive task.
"""


with open(
    OUTPUT_DIR / "final_recommendation.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(recommendation)

    file.write(
        "\n\nHeteroscedasticity conclusion:\n"
    )

    file.write(
        heteroscedasticity_conclusion
    )


print("\n" + recommendation)

print("\nModeling pipeline completed successfully.")