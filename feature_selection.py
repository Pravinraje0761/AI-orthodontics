import pandas as pd


def select_features(
    corr_matrix: pd.DataFrame, feature_cols: list[str], threshold: float = 0.05
) -> list[str]:
    target_correlations = corr_matrix["Gender"].drop("Gender").sort_values(
        key=abs, ascending=False
    )

    selected_features = target_correlations[
        abs(target_correlations) >= threshold
    ].index.tolist()

    if len(selected_features) < 2:
        selected_features = feature_cols.copy()

    return selected_features
