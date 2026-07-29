import streamlit as st
import plotly.express as px

from src.forecasting import (
    prepare_monthly_sales,
    create_lag_features,
    split_time_series,
)


def show():
    st.set_page_config(page_title="Sales Forecasting")

    st.title("📈 Sales Forecasting")


    # Check previous stage

    if st.session_state.processed_df is None:
        st.warning("Please complete Data Cleaning first.")
        st.stop()

    if st.session_state.target_column is None:
        st.warning("Please complete Preprocessing first.")
        st.stop()


    # Retrieve dataset

    df = st.session_state.processed_df

    target_column = st.session_state.target_column


    # Forecast Configuration

    st.subheader("Forecast Configuration")

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

    date_column = st.selectbox(
        "Date Column",
        options=date_columns,
    )

    st.write(
        f"**Target Variable:** {target_column}"
    )


    # Prepare Monthly Sales

    if st.button(
        "Prepare Time Series",
        use_container_width=True,
    ):

        monthly_sales = prepare_monthly_sales(
            df=df,
            date_column=date_column,
            target_column=target_column,
        )

        st.session_state["forecasting"]["monthly_sales"] = (
            monthly_sales
        )

        # Previous forecast is invalid

        st.session_state["forecasting"]["forecast_df"] = None
        st.session_state["forecasting"]["model"] = None
        st.session_state["forecasting"]["metrics"] = None

        st.success(
            "Monthly sales time series prepared successfully!"
        )


    # Monthly Sales

    monthly_sales = (
        st.session_state["forecasting"]["monthly_sales"]
    )

    if monthly_sales is not None:

        st.divider()

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
            x=date_column,
            y=target_column,
            markers=True,
            title="Monthly Sales Trend",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # Lag Feature Configuration

        st.divider()

        st.subheader("Lag Features")

        n_lags = st.slider(
            "Number of Previous Months",
            min_value=1,
            max_value=12,
            value=3,
            step=1,
        )

        lagged_data = create_lag_features(
            monthly_sales=monthly_sales,
            target_column=target_column,
            n_lags=n_lags,
        )

        st.write(
            f"Using the previous **{n_lags} months** "
            "to predict the current month's sales."
        )

        st.dataframe(
            lagged_data,
            hide_index=True,
            use_container_width=True,
        )

        # Forecast Train / Test Split

        st.divider()

        st.subheader("Forecast Train / Test Split")

        forecast_test_size = st.slider(
            "Forecast Test Size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
        )

        split_data = split_time_series(
            lagged_data=lagged_data,
            date_column=date_column,
            target_column=target_column,
            test_size=forecast_test_size,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Training Months",
                len(split_data["X_train"]),
            )

        with col2:
            st.metric(
                "Testing Months",
                len(split_data["X_test"]),
            )

        st.write(
            f"**Training Period:** "
            f"{split_data['train_dates'].iloc[0].strftime('%b %Y')} "
            f"to "
            f"{split_data['train_dates'].iloc[-1].strftime('%b %Y')}"
        )

        st.write(
            f"**Testing Period:** "
            f"{split_data['test_dates'].iloc[0].strftime('%b %Y')} "
            f"to "
            f"{split_data['test_dates'].iloc[-1].strftime('%b %Y')}"
        )        