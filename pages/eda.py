# pages/eda.py

import streamlit as st
import plotly.express as px

from src.eda.overview import generate_overview
from src.eda.numerical import analyze_numerical
from src.eda.categorical import analyze_categorical
from src.eda.correlation import compute_correlation
from src.eda.outliers import detect_outliers


def show():

    st.title("📊 Exploratory Data Analysis")

    if st.session_state.processed_df is None:
        st.warning("Please clean your dataset first.")
        st.stop()

    df = st.session_state.processed_df

    numeric_columns = st.session_state.numeric_columns
    categorical_columns = st.session_state.categorical_columns
    date_columns = st.session_state.date_columns

    # Dataset Overview

    st.header("Dataset Overview")

    overview = generate_overview(
        df,
        numeric_columns,
        categorical_columns,
        date_columns,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", overview["rows"])
        st.metric(
            "Numeric Columns",
            overview["numeric_columns"]
        )

    with col2:
        st.metric("Columns", overview["columns"])
        st.metric(
            "Categorical Columns",
            overview["categorical_columns"]
        )

    with col3:
        st.metric(
            "Memory Usage",
            f'{overview["memory_usage_mb"]} MB'
        )
        st.metric(
            "Datetime Columns",
            overview["datetime_columns"]
        )

    st.dataframe(
        overview["column_summary"],
        use_container_width=True
    )

    st.divider()

    # Numerical Analysis

    st.header("Numerical Analysis")

    numerical = analyze_numerical(
        df,
        numeric_columns,
    )

    if numerical["summary"].empty:

        st.info("No numerical columns found.")

    else:

        st.dataframe(
            numerical["summary"],
            use_container_width=True
        )

        selected_numeric = st.selectbox(
            "Select Numerical Column",
            numerical["numeric_columns"]
        )

        fig = px.histogram(
            df,
            x=selected_numeric,
            title=f"{selected_numeric} Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fig = px.box(
            df,
            y=selected_numeric,
            title=f"{selected_numeric} Box Plot"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # Categorical Analysis

    st.header("Categorical Analysis")

    categorical = analyze_categorical(
        df,
        categorical_columns,
    )

    if categorical["summary"].empty:

        st.info("No categorical columns found.")

    else:

        st.dataframe(
            categorical["summary"],
            use_container_width=True
        )

        selected_category = st.selectbox(
            "Select Categorical Column",
            categorical["categorical_columns"]
        )

        counts = (
            df[selected_category]
            .value_counts()
            .head(20)
            .reset_index()
        )

        counts.columns = [
            selected_category,
            "Count"
        ]

        fig = px.bar(
            counts,
            x=selected_category,
            y="Count",
            title=f"{selected_category} Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # Correlation Analysis

    st.header("Correlation Analysis")

    corr = compute_correlation(
        df,
        numeric_columns,
    )

    if corr.empty:

        st.info("At least two numerical columns are required.")

    else:

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # Outlier Analysis

    st.header("Outlier Analysis")

    outliers = detect_outliers(
        df,
        numeric_columns,
    )

    if outliers.empty:

        st.info("No numerical columns found.")

    else:

        st.dataframe(
            outliers,
            use_container_width=True
        )