import pandas as pd
import streamlit as st


def show():
    st.set_page_config(page_title="Model Training")

    st.title("🤖 Model Training")

    # Check previous stage

    if st.session_state.preprocessing["X_train"] is None:
        st.warning("Please complete the Preprocessing step first.")
        st.stop()

    # Retrieve preprocessing results

    preprocessing = st.session_state.preprocessing

    X_train = preprocessing["X_train"]
    X_test = preprocessing["X_test"]
    y_train = preprocessing["y_train"]
    y_test = preprocessing["y_test"]
    preprocessor = preprocessing["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    # Convert sparse matrices for display

    if hasattr(X_train, "toarray"):
        X_train_display = X_train.toarray()
    else:
        X_train_display = X_train

    if hasattr(X_test, "toarray"):
        X_test_display = X_test.toarray()
    else:
        X_test_display = X_test

    X_train_df = pd.DataFrame(
        X_train_display,
        columns=feature_names,
    )

    X_test_df = pd.DataFrame(
        X_test_display,
        columns=feature_names,
    )

    # Dataset Summary

    st.subheader("Dataset Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Training Samples",
            X_train.shape[0],
        )

        st.metric(
            "Training Target",
            len(y_train),
        )

    with col2:
        st.metric(
            "Testing Samples",
            X_test.shape[0],
        )

        st.metric(
            "Testing Target",
            len(y_test),
        )

    with col3:
        st.metric(
            "Generated Features",
            len(feature_names),
        )

    st.divider()

    # Generated Features

    st.subheader("Generated Features")

    feature_df = pd.DataFrame(
        {
            "Feature Name": feature_names,
        }
    )

    st.dataframe(
        feature_df,
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    # Processed Training Dataset

    st.subheader("Processed Training Dataset")

    st.dataframe(
        X_train_df.head(),
        use_container_width=True,
    )

    st.divider()

    # Processed Testing Dataset

    st.subheader("Processed Testing Dataset")

    st.dataframe(
        X_test_df.head(),
        use_container_width=True,
    )

    st.divider()

    # Model Configuration

    st.subheader("Model Configuration")

    model_name = st.selectbox(
        "Regression Model",
        [
            "Linear Regression",
            "Decision Tree Regressor",
            "Random Forest Regressor",
            "Gradient Boosting Regressor",
        ],
    )

    if st.button(
        "Train Model",
        use_container_width=True,
    ):
        st.info(
            f"Training **{model_name}**... (implementation coming next)"
        )