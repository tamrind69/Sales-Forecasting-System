import pandas as pd
import streamlit as st
import plotly.express as px

from src.forecasting import generate_future_forecast


def show():
    st.set_page_config(page_title="Sales Forecasting")

    st.title("📈 Future Sales Forecast")


    # Check previous stage

    if st.session_state.model is None:
        st.warning(
            "Please train a forecasting model first."
        )
        st.stop()

    if st.session_state.preprocessing["monthly_sales"] is None:
        st.warning(
            "Please complete Forecast Preprocessing first."
        )
        st.stop()


    # Retrieve pipeline results

    model = st.session_state.model
    model_name = st.session_state.model_name

    monthly_sales = (
        st.session_state.preprocessing["monthly_sales"]
    )

    date_column = st.session_state.date_column
    target_column = st.session_state.target_column


    # Forecast Configuration

    st.subheader("Forecast Configuration")

    st.write(
        f"**Model:** {model_name}"
    )

    st.write(
        f"**Historical Data Ends:** "
        f"{monthly_sales[date_column].iloc[-1].strftime('%b %Y')}"
    )

    forecast_months = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=12,
        value=6,
        step=1,
    )

    st.caption(
        "Future months are predicted recursively using "
        "previous historical sales and earlier forecasts."
    )


    # Generate Forecast

    if st.button(
        "Generate Forecast",
        use_container_width=True,
    ):

        forecast_df = generate_future_forecast(
            model=model,
            monthly_sales=monthly_sales,
            date_column=date_column,
            target_column=target_column,
            forecast_months=forecast_months,
        )

        st.session_state.forecasting = {
            "forecast_df": forecast_df,
        }

        st.success(
            f"{forecast_months}-month forecast generated successfully!"
        )


    # Forecast Results

    if st.session_state.forecasting["forecast_df"] is not None:

        forecast_df = (
            st.session_state.forecasting["forecast_df"]
        )

        st.divider()

        st.subheader("Forecast Results")

        st.dataframe(
            forecast_df,
            hide_index=True,
            use_container_width=True,
        )


        # Forecast Summary

        st.subheader("Forecast Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Forecast Months",
                len(forecast_df),
            )

        with col2:
            st.metric(
                "Average Forecast",
                f"{forecast_df['Predicted Sales'].mean():,.2f}",
            )

        with col3:
            st.metric(
                "Total Forecast",
                f"{forecast_df['Predicted Sales'].sum():,.2f}",
            )


        # Historical + Forecast Data

        historical_df = monthly_sales[
            [
                date_column,
                target_column,
            ]
        ].copy()

        historical_df = historical_df.rename(
            columns={
                date_column: "Date",
                target_column: "Sales",
            }
        )

        historical_df["Series"] = "Historical"


        future_df = forecast_df.rename(
            columns={
                date_column: "Date",
                "Predicted Sales": "Sales",
            }
        )

        future_df["Series"] = "Forecast"


        # Add final historical point to forecast series
        # so the forecast line connects to history

        forecast_start = historical_df.tail(1).copy()
        forecast_start["Series"] = "Forecast"

        future_plot_df = pd.concat(
            [
                forecast_start,
                future_df,
            ],
            ignore_index=True,
        )

        plot_df = pd.concat(
            [
                historical_df,
                future_plot_df,
            ],
            ignore_index=True,
        )


        # Historical + Forecast Plot

        st.subheader("Historical and Forecast Sales")

        fig = px.line(
            plot_df,
            x="Date",
            y="Sales",
            color="Series",
            markers=True,
            title="Monthly Sales Forecast",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )