from typing import Any

import pandas as pd
from scipy.stats import mannwhitneyu, shapiro, ttest_ind


def run_statistical_analysis(
    df: pd.DataFrame, feature_cols: list[str], target_col: str = "Gender"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normality_results: list[dict[str, Any]] = []
    for col in feature_cols:
        stat, p_val = shapiro(df[col])
        normality_results.append(
            {
                "Feature": col,
                "Shapiro_Statistic": stat,
                "p_value": p_val,
                "Normality": "Normal" if p_val > 0.05 else "Not Normal",
            }
        )
    normality_df = pd.DataFrame(normality_results)

    male_df = df[df[target_col] == 1]
    female_df = df[df[target_col] == 0]
    stat_results: list[dict[str, Any]] = []

    for col in feature_cols:
        p_male = shapiro(male_df[col])[1]
        p_female = shapiro(female_df[col])[1]

        if p_male > 0.05 and p_female > 0.05:
            test_name = "Independent t-test"
            stat, p_val = ttest_ind(male_df[col], female_df[col], equal_var=False)
        else:
            test_name = "Mann-Whitney U test"
            stat, p_val = mannwhitneyu(
                male_df[col], female_df[col], alternative="two-sided"
            )

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

    stat_results_df = pd.DataFrame(stat_results)
    return normality_df, stat_results_df