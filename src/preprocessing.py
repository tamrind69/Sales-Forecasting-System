from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

import pandas as pd


# Feature / Target Split

def split_features_target(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
):
    """
    Split dataframe into features and target.
    """

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    return X, y



# Train-Test Split

def split_dataset(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
):
    """
    Split dataset into train and test sets.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )



# Scaler

def get_scaler(name: str):
    """
    Return scaler object.
    """

    scalers = {
        "None": "passthrough",
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
    }

    return scalers[name]



# Build Preprocessor

def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    scaler_name: str,
):
    """
    Create preprocessing pipeline.
    """

    numeric_transformer = Pipeline(
        steps=[
            (
                "scaler",
                get_scaler(scaler_name),
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_columns,
            ),
            (
                "cat",
                categorical_transformer,
                categorical_columns,
            ),
        ]
    )

    return preprocessor



# Fit & Transform

def fit_preprocessor(
    preprocessor,
    X_train,
):
    """
    Fit preprocessing pipeline.
    """

    preprocessor.fit(X_train)

    return preprocessor


def transform_dataset(
    preprocessor,
    X,
):
    """
    Transform dataset.
    """

    return preprocessor.transform(X)





# Complete Pipeline

def preprocess_dataset(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    scaler_name: str = "StandardScaler",
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
):
    """
    Complete preprocessing workflow.
    """

    # Split features and target
    X, y = split_features_target(
        df,
        target_column,
        feature_columns,
    )

    # Train-test split
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_dataset(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    # Build preprocessor
    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        scaler_name=scaler_name,
    )

    # Fit on training data
    preprocessor = fit_preprocessor(
        preprocessor,
        X_train,
    )

    # Transform datasets
    X_train = transform_dataset(
        preprocessor,
        X_train,
    )

    X_test = transform_dataset(
        preprocessor,
        X_test,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
    }