import streamlit as st


def show():
    st.title("🏠 Home")

    st.markdown("""
    Welcome to the **Sales Forecasting System**.

    This application allows you to:

    - 📂 Upload sales datasets
    - 🧹 Preprocess data
    - 📊 Perform Exploratory Data Analysis (EDA)
    - 🤖 Train Machine Learning models
    - 📈 Forecast future sales
    - 📄 Generate downloadable reports

    Use the navigation menu to get started.
    """)