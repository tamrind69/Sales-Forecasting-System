import streamlit as st


DEFAULT_SESSION_STATE = {

    # ------------------------------
    # Dataset
    # ------------------------------

    "raw_df": None,
    "processed_df": None,
    "dataset_name": None,

    # ------------------------------
    # Column Configuration
    # ------------------------------

    "numeric_columns": [],
    "categorical_columns": [],
    "date_columns": [],

    # ------------------------------
    # Forecast Preprocessing
    # ------------------------------

    "target_column": None,
    "date_column": None,

    "preprocessing": {
        "monthly_sales": None,
        "forecast_data": None,

        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,

        "train_dates": None,
        "test_dates": None,

        "feature_columns": None,
    },

    # ------------------------------
    # Model Training
    # ------------------------------

    "model": None,
    "model_name": None,
    "metrics": None,
    "predictions": None,

    # ------------------------------
    # Forecasting
    # ------------------------------

    "forecasting": {
        "forecast_df": None,
    },
}


def initialize_session():
    """Initialize application session state."""

    for key, value in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_pipeline():
    """Reset everything derived from the uploaded dataset."""

    # Cleaned Dataset

    st.session_state["processed_df"] = None

    # Column Configuration

    st.session_state["numeric_columns"] = []
    st.session_state["categorical_columns"] = []
    st.session_state["date_columns"] = []

    # Forecast Preprocessing

    st.session_state["target_column"] = None
    st.session_state["date_column"] = None

    st.session_state["preprocessing"] = {
        "monthly_sales": None,
        "forecast_data": None,

        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,

        "train_dates": None,
        "test_dates": None,

        "feature_columns": None,
    }

    # Model Training

    st.session_state["model"] = None
    st.session_state["model_name"] = None
    st.session_state["metrics"] = None
    st.session_state["predictions"] = None

    # Forecasting

    st.session_state["forecasting"] = {
        "forecast_df": None,
    }