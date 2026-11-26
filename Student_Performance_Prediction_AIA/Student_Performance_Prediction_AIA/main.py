"""
AIA Project: Student Performance Prediction using Machine Learning
Dataset: Kaggle - Student Performance Factors
Target: Exam_Score
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "Exam_Score"
RANDOM_STATE = 42


def locate_dataset():
    """Find the Kaggle CSV in data/ or project root."""
    candidates = list(DATA_DIR.glob("*.csv")) + list(ROOT.glob("*.csv"))
    for file in candidates:
        if file.name.lower() not in {"metrics.csv", "feature_importance.csv"}:
            return file
    raise FileNotFoundError(
        "Dataset CSV not found. Download StudentPerformanceFactors.csv from Kaggle "
        "and place it in the project's data/ folder."
    )


def load_and_clean(path):
    df = pd.read_csv(path)
    print("\\n--- ORIGINAL DATASET ---")
    print("Shape:", df.shape)
    print(df.head())

    # Standardize column names while preserving expected Kaggle names.
    df.columns = [str(c).strip() for c in df.columns]
    if TARGET not in df.columns:
        # Case-insensitive target column detection
        target_match = next((c for c in df.columns if c.lower() == TARGET.lower()), None)
        if target_match:
            df.rename(columns={target_match: TARGET}, inplace=True)
        else:
            raise ValueError(f"Target column '{TARGET}' was not found. Available columns: {list(df.columns)}")

    missing_before = int(df.isna().sum().sum())
    duplicates_before = int(df.duplicated().sum())

    # Remove exact duplicate records.
    df = df.drop_duplicates().copy()

    # Convert target to numeric and remove rows with missing target.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()

    # Remove impossible exam scores if this dataset uses a 0-100 scale.
    df = df[df[TARGET].between(0, 100)].copy()

    print("\\n--- CLEANING SUMMARY ---")
    print(f"Missing cells before cleaning: {missing_before}")
    print(f"Duplicate rows removed: {duplicates_before}")
    print("Rows after removing duplicate/missing-target/impossible-score rows:", len(df))
    print("Remaining missing values are imputed inside the training pipeline.")

    df.to_csv(OUTPUT_DIR / "cleaned_dataset.csv", index=False)
    return df


def make_pipeline(X):
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ])

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
        n_jobs=-1
    )
    return Pipeline(steps=[("preprocessing", preprocessor), ("model", model)])


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2 Score": r2}, predictions


def main():
    csv_path = locate_dataset()
    print("Dataset:", csv_path)
    df = load_and_clean(csv_path)

    # Correlation analysis for numeric features.
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] >= 2:
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", square=False)
        plt.title("Correlation Heatmap - Numerical Features")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=300)
        plt.close()
        numeric_df.corr().to_csv(OUTPUT_DIR / "numerical_correlation.csv")

    # Separate features and target.
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"\\nTraining records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    # Random Forest model with preprocessing in pipeline.
    rf_pipeline = make_pipeline(X_train)
    rf_pipeline.fit(X_train, y_train)
    rf_result, rf_predictions = evaluate_model("Random Forest Regressor", rf_pipeline, X_test, y_test)

    # Baseline model for comparison.
    baseline_preprocessor = rf_pipeline.named_steps["preprocessing"]
    baseline = Pipeline(steps=[
        ("preprocessing", baseline_preprocessor),
        ("model", LinearRegression())
    ])
    baseline.fit(X_train, y_train)
    lr_result, lr_predictions = evaluate_model("Linear Regression", baseline, X_test, y_test)

    results = pd.DataFrame([lr_result, rf_result])
    results.to_csv(OUTPUT_DIR / "model_evaluation.csv", index=False)
    print("\\n--- MODEL EVALUATION ---")
    print(results.to_string(index=False))

    # Actual vs predicted chart for Random Forest.
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, rf_predictions, alpha=0.65)
    low = min(float(y_test.min()), float(np.min(rf_predictions)))
    high = max(float(y_test.max()), float(np.max(rf_predictions)))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual Exam Score")
    plt.ylabel("Predicted Exam Score")
    plt.title("Actual vs Predicted Scores (Random Forest)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=300)
    plt.close()

    # Feature importance (encoded feature names).
    try:
        preprocessor = rf_pipeline.named_steps["preprocessing"]
        model = rf_pipeline.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

        top = importance.head(15).sort_values("Importance")
        plt.figure(figsize=(10, 7))
        plt.barh(top["Feature"], top["Importance"])
        plt.xlabel("Importance")
        plt.title("Top 15 Feature Importances - Random Forest")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=300)
        plt.close()
    except Exception as e:
        print("Could not generate feature importance:", e)

    # Example prediction using one test record.
    example = X_test.iloc[[0]]
    example_prediction = float(rf_pipeline.predict(example)[0])
    print(f"\\nExample predicted exam score: {example_prediction:.2f}")
    print("\\nFiles saved in outputs/ folder.")
    print("Note: Dataset patterns show association, not proof of causation.")


if __name__ == "__main__":
    main()
