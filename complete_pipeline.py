import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, shapiro, ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURE_COLS = [
    "Gonial_Angle",
    "Condylar_Height",
    "Coronoid_Ramus_Height",
    "Intercondylar_Distance",
    "Sigmoid_Notch_Angle",
    "Mental_Foramen_to_Inferior_Border_Ramus",
]
TARGET_COL = "Gender"


def preprocess_data(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(file_path)
    df[TARGET_COL] = df[TARGET_COL].replace(
        {"Male": 1, "Female": 0, "male": 1, "female": 0, "M": 1, "F": 0}
    )
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET_COL]).copy()
    imputer = SimpleImputer(strategy="median")
    df[FEATURE_COLS] = imputer.fit_transform(df[FEATURE_COLS])

    corr_matrix = df[FEATURE_COLS + [TARGET_COL]].corr()
    return df, corr_matrix


def run_statistical_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    normality_results: list[dict[str, Any]] = []
    for col in FEATURE_COLS:
        stat, p_val = shapiro(df[col])
        normality_results.append(
            {
                "Feature": col,
                "Shapiro_Statistic": stat,
                "p_value": p_val,
                "Normality": "Normal" if p_val > 0.05 else "Not Normal",
            }
        )

    male_df = df[df[TARGET_COL] == 1]
    female_df = df[df[TARGET_COL] == 0]
    stat_results: list[dict[str, Any]] = []
    for col in FEATURE_COLS:
        p_male = shapiro(male_df[col])[1]
        p_female = shapiro(female_df[col])[1]
        if p_male > 0.05 and p_female > 0.05:
            test_name = "Independent t-test"
            stat, p_val = ttest_ind(male_df[col], female_df[col], equal_var=False)
        else:
            test_name = "Mann-Whitney U test"
            stat, p_val = mannwhitneyu(male_df[col], female_df[col], alternative="two-sided")
        stat_results.append(
            {
                "Feature": col,
                "Male_Mean": male_df[col].mean(),
                "Female_Mean": female_df[col].mean(),
                "Male_Median": male_df[col].median(),
                "Female_Median": female_df[col].median(),
                "Test_Used": test_name,
                "Test_Statistic": stat,
                "p_value": p_val,
                "Significant_at_0.05": "Yes" if p_val < 0.05 else "No",
            }
        )

    return pd.DataFrame(normality_results), pd.DataFrame(stat_results)


def select_features(corr_matrix: pd.DataFrame) -> list[str]:
    target_correlations = corr_matrix[TARGET_COL].drop(TARGET_COL).sort_values(key=abs, ascending=False)
    selected_features = target_correlations[abs(target_correlations) >= 0.05].index.tolist()
    return selected_features if len(selected_features) >= 2 else FEATURE_COLS.copy()


def train_and_evaluate(df: pd.DataFrame, selected_features: list[str]) -> dict[str, Any]:
    X = df[selected_features]
    y = df[TARGET_COL].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    rf_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )
    rf_pipeline.fit(X_train, y_train)
    y_pred = rf_pipeline.predict(X_test)
    y_proba = rf_pipeline.predict_proba(X_test)[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_pipeline, X, y, cv=cv, scoring="accuracy")
    auc_score = roc_auc_score(y_test, y_proba)
    comparison_models = {
        "Random Forest": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=200, random_state=42))]
        ),
        "Logistic Regression": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42, max_iter=1000))]
        ),
        "SVM": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", SVC(probability=True, random_state=42))]
        ),
    }
    comparison_rows = []
    for name, model in comparison_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        comparison_rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred),
                "Recall": recall_score(y_test, pred),
                "F1_score": f1_score(y_test, pred),
            }
        )

    return {
        "rf_pipeline": rf_pipeline,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "cv_scores": cv_scores,
        "auc_score": auc_score,
        "classification_report": classification_report(y_test, y_pred, target_names=["Female", "Male"]),
        "comparison_df": pd.DataFrame(comparison_rows).sort_values(by="Accuracy", ascending=False),
    }


def save_model(model: Pipeline, model_path: str = "gender_rf_model.pkl") -> None:
    with Path(model_path).open("wb") as model_file:
        pickle.dump(model, model_file)


def plot_eda(df: pd.DataFrame, corr_matrix: pd.DataFrame) -> None:
    sns.set(style="whitegrid", context="talk")
    plt.figure(figsize=(18, 12))
    for i, col in enumerate(FEATURE_COLS, 1):
        plt.subplot(2, 3, i)
        sns.histplot(df[col], kde=True, bins=20, color="steelblue")
        plt.title(f"Histogram of {col}")
    plt.tight_layout()
    plt.show()

    df_plot = df.copy()
    df_plot["Gender_Label"] = df_plot[TARGET_COL].map({0: "Female", 1: "Male"})
    plt.figure(figsize=(18, 12))
    for i, col in enumerate(FEATURE_COLS, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(data=df_plot, x="Gender_Label", y=col, palette="Set2")
        plt.title(f"{col} by Gender")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()


def plot_model_results(selected_features: list[str], training_results: dict[str, Any]) -> None:
    y_test = training_results["y_test"]
    y_pred = training_results["y_pred"]
    y_proba = training_results["y_proba"]
    rf_model = training_results["rf_pipeline"].named_steps["model"]
    importances = pd.DataFrame(
        {"Feature": selected_features, "Importance": rf_model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Female", "Male"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Random Forest")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importances, x="Importance", y="Feature", palette="viridis")
    plt.title("Feature Importance - Random Forest")
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Random Forest (AUC = {training_results['auc_score']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Random Forest")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    dataset_path = "mandibular_gender_dataset.csv"
    df_data, corr = preprocess_data(dataset_path)
    normality_df, stats_df = run_statistical_analysis(df_data)
    selected = select_features(corr)
    results = train_and_evaluate(df_data, selected)
    save_model(results["rf_pipeline"])

    print("Selected features:", selected)
    print("\nNormality results:")
    print(normality_df)
    print("\nStatistical comparison:")
    print(stats_df)
    print("\nClassification report:")
    print(results["classification_report"])
    print("\nCV mean accuracy:", float(np.mean(results["cv_scores"])))
    print("AUC score:", float(results["auc_score"]))
    print("\nModel comparison:")
    print(results["comparison_df"])
