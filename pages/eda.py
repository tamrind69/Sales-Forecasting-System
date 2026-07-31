# pages/eda.py

import pandas as pd
import streamlit as st
import plotly.express as px

from src.eda.overview import generate_overview
from src.eda.correlation import compute_correlation


def show():

    st.title("📊 Sales Analysis")

    # Check previous stage

    if st.session_state.processed_df is None:
        st.warning("Please clean your dataset first.")
        st.stop()

    df = st.session_state.processed_df

    numeric_columns = [
        col
        for col in st.session_state.numeric_columns
        if col in df.columns
    ]

    categorical_columns = [
        col
        for col in st.session_state.categorical_columns
        if col in df.columns
    ]

    date_columns = [
        col
        for col in st.session_state.date_columns
        if col in df.columns
    ]


    # --------------------------------------------------
    # Dataset Overview
    # --------------------------------------------------

    st.header("Dataset Overview")

    overview = generate_overview(
        df,
        numeric_columns,
        categorical_columns,
        date_columns,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            overview["rows"],
        )

        st.metric(
            "Numeric Columns",
            overview["numeric_columns"],
        )

    with col2:
        st.metric(
            "Columns",
            overview["columns"],
        )

        st.metric(
            "Categorical Columns",
            overview["categorical_columns"],
        )

    with col3:
        st.metric(
            "Memory Usage",
            f'{overview["memory_usage_mb"]} MB',
        )

        st.metric(
            "Datetime Columns",
            overview["datetime_columns"],
        )

    st.dataframe(
        overview["column_summary"],
        use_container_width=True,
    )

    st.divider()


    # --------------------------------------------------
    # Sales Configuration
    # --------------------------------------------------

    st.header("Sales Configuration")

    if len(numeric_columns) == 0:
        st.warning(
            "No numerical columns are available for sales analysis."
        )
        st.stop()

    sales_column = st.selectbox(
        "Select Sales Column",
        options=numeric_columns,
    )

    st.divider()


    # --------------------------------------------------
    # Sales Overview
    # --------------------------------------------------

    st.header("Sales Overview")

    sales_data = pd.to_numeric(
        df[sales_column],
        errors="coerce",
    )

    total_sales = sales_data.sum()
    average_sales = sales_data.mean()
    median_sales = sales_data.median()
    transactions = sales_data.notna().sum()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Sales",
            f"{total_sales:,.2f}",
        )

    with col2:
        st.metric(
            "Average Sale",
            f"{average_sales:,.2f}",
        )

    with col3:
        st.metric(
            "Median Sale",
            f"{median_sales:,.2f}",
        )

    with col4:
        st.metric(
            "Transactions",
            f"{transactions:,}",
        )

    st.divider()


    # --------------------------------------------------
    # Monthly Sales Trend
    # --------------------------------------------------

    st.header("Monthly Sales Trend")

    if len(date_columns) == 0:

        st.info(
            "No date columns are available for monthly sales analysis."
        )

    else:

        date_column = st.selectbox(
            "Select Date Column",
            options=date_columns,
        )

        monthly_data = df[
            [
                date_column,
                sales_column,
            ]
        ].copy()

        monthly_data[date_column] = pd.to_datetime(
            monthly_data[date_column],
            errors="coerce",
        )

        monthly_data[sales_column] = pd.to_numeric(
            monthly_data[sales_column],
            errors="coerce",
        )

        monthly_data = monthly_data.dropna(
            subset=[
                date_column,
                sales_column,
            ]
        )

        monthly_sales = (
            monthly_data
            .set_index(date_column)
            .resample("MS")[sales_column]
            .sum()
            .reset_index()
        )

        fig = px.line(
            monthly_sales,
            x=date_column,
            y=sales_column,
            markers=True,
            title="Monthly Sales Trend",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.divider()


    # --------------------------------------------------
    # Regional Sales Analysis
    # --------------------------------------------------

    st.header("Regional Sales Analysis")

    if len(categorical_columns) == 0:

        st.info(
            "No categorical columns are available "
            "for regional analysis."
        )

    else:

        region_column = st.selectbox(
            "Select Regional Column",
            options=categorical_columns,
        )

        regional_data = df[
            [
                region_column,
                sales_column,
            ]
        ].copy()

        regional_data[sales_column] = pd.to_numeric(
            regional_data[sales_column],
            errors="coerce",
        )

        regional_data = regional_data.dropna(
            subset=[
                region_column,
                sales_column,
            ]
        )

        regional_sales = (
            regional_data
            .groupby(
                region_column,
                as_index=False,
            )[sales_column]
            .sum()
            .sort_values(
                by=sales_column,
                ascending=False,
            )
        )

        st.dataframe(
            regional_sales,
            hide_index=True,
            use_container_width=True,
        )

        fig = px.bar(
            regional_sales,
            x=region_column,
            y=sales_column,
            title=f"Sales by {region_column}",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.divider()


    # --------------------------------------------------
    # Product Performance
    # --------------------------------------------------

    st.header("Product Performance")

    if len(categorical_columns) == 0:

        st.info(
            "No categorical columns are available "
            "for product analysis."
        )

    else:

        product_column = st.selectbox(
            "Select Product / Category Column",
            options=categorical_columns,
        )

        top_n = st.slider(
            "Number of Top Items",
            min_value=5,
            max_value=30,
            value=10,
            step=5,
        )

        product_data = df[
            [
                product_column,
                sales_column,
            ]
        ].copy()

        product_data[sales_column] = pd.to_numeric(
            product_data[sales_column],
            errors="coerce",
        )

        product_data = product_data.dropna(
            subset=[
                product_column,
                sales_column,
            ]
        )

        product_sales = (
            product_data
            .groupby(
                product_column,
                as_index=False,
            )[sales_column]
            .sum()
            .sort_values(
                by=sales_column,
                ascending=False,
            )
            .head(top_n)
        )

        st.dataframe(
            product_sales,
            hide_index=True,
            use_container_width=True,
        )

        fig = px.bar(
            product_sales,
            x=sales_column,
            y=product_column,
            orientation="h",
            title=f"Top {top_n} {product_column} by Sales",
        )

        fig.update_layout(
            yaxis={
                "categoryorder": "total ascending"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.divider()


    # --------------------------------------------------
    # Sales Distribution
    # --------------------------------------------------

    st.header("Sales Distribution")

    distribution_data = df[
        [sales_column]
    ].copy()

    distribution_data[sales_column] = pd.to_numeric(
        distribution_data[sales_column],
        errors="coerce",
    )

    distribution_data = distribution_data.dropna()

    fig = px.histogram(
        distribution_data,
        x=sales_column,
        title=f"{sales_column} Distribution",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    fig = px.box(
        distribution_data,
        y=sales_column,
        title=f"{sales_column} Box Plot",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.divider()


    # --------------------------------------------------
    # Additional Analysis
    # --------------------------------------------------

    st.header("Additional Analysis")

    st.subheader("Correlation Analysis")

    corr = compute_correlation(
        df,
        numeric_columns,
    )

    if corr.empty:

        st.info(
            "At least two numerical columns are required."
        )

    else:

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )