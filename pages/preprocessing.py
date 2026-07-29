import streamlit as st
import plotly.express as px

from src.preprocessing import (
    prepare_monthly_sales,
    create_forecast_features,
    split_time_series,
)


def show():
    st.set_page_config(page_title="Preprocessing")

    st.title("⚙️ Forecast Preprocessing")


    # Check previous stage

    if st.session_state.processed_df is None:
        st.warning("Please complete Data Cleaning first.")
        st.stop()

    df = st.session_state.processed_df


    # Target Column

    st.subheader("1. Target Variable")

    numeric_columns = [
        col
        for col in st.session_state.numeric_columns
        if col in df.columns
    ]

    if len(numeric_columns) == 0:
        st.warning("No numerical columns are available.")
        st.stop()

    if (
        st.session_state.target_column is not None
        and st.session_state.target_column in numeric_columns
    ):
        target_index = numeric_columns.index(
            st.session_state.target_column
        )
    else:
        target_index = 0

    target_column = st.selectbox(
        "Select sales target column",
        options=numeric_columns,
        index=target_index,
    )


    # Date Column

    st.subheader("2. Date Column")

    date_columns = [
        col
        for col in st.session_state.date_columns
        if col in df.columns
    ]

    if len(date_columns) == 0:
        st.warning(
            "No date columns are available. "
            "Please configure a date column first."
        )
        st.stop()

    if (
        st.session_state.date_column is not None
        and st.session_state.date_column in date_columns
    ):
        date_index = date_columns.index(
            st.session_state.date_column
        )
    else:
        date_index = 0

    date_column = st.selectbox(
        "Select date column",
        options=date_columns,
        index=date_index,
    )


    # Forecast Features

    st.subheader("3. Forecast Features")

    st.write(
        "The following historical and seasonal features "
        "will be generated:"
    )

    st.write(
        """
        - Lag 1 month
        - Lag 2 months
        - Lag 3 months
        - Lag 12 months
        - Month
        - Quarter
        """
    )

    st.caption(
        "Recent lag features capture short-term sales patterns, "
        "while Lag 12 and calendar features help represent "
        "yearly seasonality."
    )


    # Train / Test Split

    st.subheader("4. Train / Test Split")

    test_size = st.slider(
        "Test Size",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
    )

    st.caption(
        "The split is chronological. Earlier months are used "
        "for training and later months are used for testing."
    )


    # Apply Preprocessing

    st.divider()

    if st.button(
        "Apply Preprocessing",
        use_container_width=True,
    ):

        monthly_sales = prepare_monthly_sales(
            df=df,
            date_column=date_column,
            target_column=target_column,
        )

        forecast_data = create_forecast_features(
            monthly_sales=monthly_sales,
            date_column=date_column,
            target_column=target_column,
        )

        split_data = split_time_series(
            forecast_data=forecast_data,
            date_column=date_column,
            target_column=target_column,
            test_size=test_size,
        )

        st.session_state.target_column = target_column
        st.session_state.date_column = date_column

        st.session_state.preprocessing = {
            "monthly_sales": monthly_sales,
            "forecast_data": forecast_data,

            "X_train": split_data["X_train"],
            "X_test": split_data["X_test"],
            "y_train": split_data["y_train"],
            "y_test": split_data["y_test"],

            "train_dates": split_data["train_dates"],
            "test_dates": split_data["test_dates"],
            "feature_columns": split_data["feature_columns"],
        }

        # Previous model is invalid after preprocessing changes

        st.session_state.model = None
        st.session_state.model_name = None
        st.session_state.metrics = None
        st.session_state.predictions = None

        # Previous future forecast is also invalid

        st.session_state.forecasting = {
            "forecast_df": None,
        }

        st.success(
            "Forecast preprocessing completed successfully!"
        )


    # Preprocessing Summary

    if st.session_state.preprocessing["X_train"] is not None:

        preprocessing = st.session_state.preprocessing

        monthly_sales = preprocessing["monthly_sales"]
        forecast_data = preprocessing["forecast_data"]
        feature_columns = preprocessing["feature_columns"]

        st.divider()

        st.subheader("Preprocessing Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Historical Months",
                len(monthly_sales),
            )

        with col2:
            st.metric(
                "Training Months",
                len(preprocessing["X_train"]),
            )

        with col3:
            st.metric(
                "Testing Months",
                len(preprocessing["X_test"]),
            )

        st.write(
            f"**Target Variable:** "
            f"{st.session_state.target_column}"
        )

        st.write(
            f"**Date Column:** "
            f"{st.session_state.date_column}"
        )

        st.write(
            f"**Generated Features:** "
            f"{len(feature_columns)}"
        )


        # Generated Features

        st.subheader("Generated Forecast Features")

        for feature in feature_columns:
            st.write(f"- {feature}")


        # Historical Monthly Sales

        st.subheader("Monthly Sales")

        st.dataframe(
            monthly_sales,
            hide_index=True,
            use_container_width=True,
        )


        # Historical Sales Trend

        st.subheader("Historical Sales Trend")

        fig = px.line(
            monthly_sales,
            x=st.session_state.date_column,
            y=st.session_state.target_column,
            markers=True,
            title="Monthly Sales Trend",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


        # Forecast Dataset

        st.subheader("Forecast Dataset")

        st.dataframe(
            forecast_data,
            hide_index=True,
            use_container_width=True,
        )


        # Chronological Split

        st.subheader("Chronological Split")

        train_dates = preprocessing["train_dates"]
        test_dates = preprocessing["test_dates"]

        st.write(
            f"**Training Period:** "
            f"{train_dates.iloc[0].strftime('%b %Y')} "
            f"to "
            f"{train_dates.iloc[-1].strftime('%b %Y')}"
        )

        st.write(
            f"**Testing Period:** "
            f"{test_dates.iloc[0].strftime('%b %Y')} "
            f"to "
            f"{test_dates.iloc[-1].strftime('%b %Y')}"
        )