# src/eda/categorical.py

import pandas as pd


def analyze_categorical(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    dict
        Categorical analysis results.
    """

    categorical_df = df.select_dtypes(include=["object", "category"])

    if categorical_df.empty:
        return {
            "categorical_columns": [],
            "summary": pd.DataFrame()
        }

    summary = []

    for column in categorical_df.columns:

        value_counts = categorical_df[column].value_counts(dropna=False)

        summary.append({
            "Column": column,
            "Unique Values": categorical_df[column].nunique(dropna=True),
            "Most Frequent": value_counts.index[0],
            "Frequency": value_counts.iloc[0]
        })

    summary = pd.DataFrame(summary)

    return {
        "categorical_columns": categorical_df.columns.tolist(),
        "summary": summary
    }