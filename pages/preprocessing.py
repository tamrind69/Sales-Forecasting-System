import streamlit as st

from src.preprocessing import preprocess_dataset


def show():
    st.set_page_config(page_title="Preprocessing")

    st.title("⚙️ Preprocessing")

    # Check previous stage

    if st.session_state.processed_df is None:
        st.warning("Please complete Data Cleaning first.")
        st.stop()

    df = st.session_state.processed_df


    # Target Column

    st.subheader("1. Target Variable")

    if (
        st.session_state.target_column is not None
        and st.session_state.target_column in df.columns
    ):
        target_index = list(df.columns).index(
            st.session_state.target_column
        )
    else:
        target_index = 0

    target_column = st.selectbox(
        "Select target column",
        df.columns,
        index=target_index,
    )


    # Feature Selection

    st.subheader("2. Feature Selection")

    available_features = [
        col
        for col in (
            st.session_state.numeric_columns
            + st.session_state.categorical_columns
        )
        if col != target_column and col in df.columns
    ]

    # Use previous feature selection if preprocessing
    # has already been configured

    if st.session_state.feature_columns:
        default_features = [
            col
            for col in st.session_state.feature_columns
            if col in available_features
        ]
    else:
        default_features = available_features

    feature_columns = st.multiselect(
        "Select feature columns",
        options=available_features,
        default=default_features,
    )


    # Scaling

    st.subheader("3. Numerical Scaling")

    scaler_name = st.selectbox(
        "Scaler",
        [
            "None",
            "StandardScaler",
            "MinMaxScaler",
            "RobustScaler",
        ],
    )


    # Train Test Split

    st.subheader("4. Train / Test Split")

    test_size = st.slider(
        "Test Size",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
    )

    random_state = st.number_input(
        "Random State",
        value=42,
    )

    shuffle = st.checkbox(
        "Shuffle Dataset",
        value=True,
    )


    # Apply

    st.divider()

    if st.button(
        "Apply Preprocessing",
        use_container_width=True,
    ):

        if len(feature_columns) == 0:
            st.error("Select at least one feature.")
            st.stop()

        numeric_columns = [
            col
            for col in st.session_state.numeric_columns
            if col in feature_columns
        ]

        categorical_columns = [
            col
            for col in st.session_state.categorical_columns
            if col in feature_columns
        ]

        results = preprocess_dataset(
            df=df,
            target_column=target_column,
            feature_columns=feature_columns,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            scaler_name=scaler_name,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )

        st.session_state.target_column = target_column
        st.session_state.feature_columns = feature_columns

        st.session_state.preprocessing = results

        # Previous model is no longer valid after preprocessing changes

        st.session_state.model = None
        st.session_state.metrics = None
        st.session_state.predictions = None

        st.success("Preprocessing completed successfully!")


    # Preprocessing Summary

    if st.session_state.preprocessing["X_train"] is not None:

        st.divider()

        st.subheader("Preprocessing Summary")

        numeric_columns = [
            col
            for col in st.session_state.numeric_columns
            if col in st.session_state.feature_columns
        ]

        categorical_columns = [
            col
            for col in st.session_state.categorical_columns
            if col in st.session_state.feature_columns
        ]

        feature_names = (
            st.session_state.preprocessing["preprocessor"]
            .get_feature_names_out()
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Training Samples",
                st.session_state.preprocessing["X_train"].shape[0],
            )

            st.metric(
                "Selected Features",
                len(st.session_state.feature_columns),
            )

            st.metric(
                "Numeric Features",
                len(numeric_columns),
            )

        with col2:
            st.metric(
                "Testing Samples",
                st.session_state.preprocessing["X_test"].shape[0],
            )

            st.metric(
                "Target Variable",
                st.session_state.target_column,
            )

            st.metric(
                "Categorical Features",
                len(categorical_columns),
            )

        st.info(
            f"""
            Preprocessing completed successfully.

            - Final feature count after preprocessing: **{len(feature_names)}**
            - Training feature matrix shape: **{st.session_state.preprocessing["X_train"].shape}**
            - Testing feature matrix shape: **{st.session_state.preprocessing["X_test"].shape}**

            Categorical features have been encoded and numerical features have been scaled.
            """
        )