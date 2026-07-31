import pandas as pd
import streamlit as st
import plotly.express as px
from src.reporting import generate_forecast_pdf


def show():
    st.set_page_config(page_title="Reports")

    st.title("📄 Sales Forecast Report")


    # Check required pipeline stages

    if st.session_state.model is None:
        st.warning(
            "Please train a forecasting model first."
        )
        st.stop()

    if st.session_state.forecasting["forecast_df"] is None:
        st.warning(
            "Please generate a future sales forecast first."
        )
        st.stop()


    # Retrieve results

    preprocessing = st.session_state.preprocessing

    monthly_sales = preprocessing["monthly_sales"]

    forecast_df = (
        st.session_state.forecasting["forecast_df"]
    )

    model_name = st.session_state.model_name
    metrics = st.session_state.metrics

    date_column = st.session_state.date_column
    target_column = st.session_state.target_column


    # Report Overview

    st.subheader("Report Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Historical Months",
            len(monthly_sales),
        )

    with col2:
        st.metric(
            "Forecast Months",
            len(forecast_df),
        )

    with col3:
        st.metric(
            "Forecasting Model",
            model_name,
        )


    # Model Performance

    st.divider()

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "MAE",
            f"{metrics['mae']:,.2f}",
        )

    with col2:
        st.metric(
            "RMSE",
            f"{metrics['rmse']:,.2f}",
        )

    with col3:
        st.metric(
            "R² Score",
            f"{metrics['r2']:.3f}",
        )


    # Forecast Summary

    st.divider()

    st.subheader("Forecast Summary")

    forecast_total = (
        forecast_df["Predicted Sales"].sum()
    )

    forecast_average = (
        forecast_df["Predicted Sales"].mean()
    )

    forecast_highest = (
        forecast_df["Predicted Sales"].max()
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Forecast Sales",
            f"{forecast_total:,.2f}",
        )

    with col2:
        st.metric(
            "Average Monthly Forecast",
            f"{forecast_average:,.2f}",
        )

    with col3:
        st.metric(
            "Highest Monthly Forecast",
            f"{forecast_highest:,.2f}",
        )


    # Forecast Table

    st.divider()

    st.subheader("Future Sales Forecast")

    st.dataframe(
        forecast_df,
        hide_index=True,
        use_container_width=True,
    )

    # CSV Export

    csv_data = forecast_df.to_csv(
        index=False,
    ).encode("utf-8")

    st.download_button(
        label="Download Forecast CSV",
        data=csv_data,
        file_name="sales_forecast.csv",
        mime="text/csv",
        use_container_width=True,
    )


    # Historical + Forecast Chart

    st.subheader("Historical and Forecast Sales")

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


    # Connect forecast line to historical data

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


    fig = px.line(
        plot_df,
        x="Date",
        y="Sales",
        color="Series",
        markers=True,
        title="Historical and Forecast Monthly Sales",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    
    # PDF Export

    pdf_data = generate_forecast_pdf(
        model_name=model_name,
        metrics=metrics,
        monthly_sales=monthly_sales,
        forecast_df=forecast_df,
        date_column=date_column,
        target_column=target_column,
    )

    st.download_button(
        label="Download Forecast PDF",
        data=pdf_data,
        file_name="sales_forecast_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )