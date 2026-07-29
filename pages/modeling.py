import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

        if model_name == "Linear Regression":
            model = LinearRegression()

        elif model_name == "Decision Tree Regressor":
            model = DecisionTreeRegressor(
                random_state=42
            )

        elif model_name == "Random Forest Regressor":
            model = RandomForestRegressor(
                random_state=42
            )

        elif model_name == "Gradient Boosting Regressor":
            model = GradientBoostingRegressor(
                random_state=42
            )

        # Train model

        model.fit(
            X_train,
            y_train,
        )

        # Generate predictions

        y_pred = model.predict(X_test)


        # Evaluation metrics

        mae = mean_absolute_error(
            y_test,
            y_pred,
        )

        rmse = mean_squared_error(
            y_test,
            y_pred,
        ) ** 0.5

        r2 = r2_score(
            y_test,
            y_pred,
        )


        # Store model results

        st.session_state["model"] = model

        st.session_state["predictions"] = y_pred

        st.session_state["metrics"] = {
            "model_name": model_name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

        st.success(
            f"{model_name} trained successfully!"
        )


    # Model Performance

    if st.session_state["metrics"] is not None:

        metrics = st.session_state["metrics"]

        st.divider()

        st.subheader("Model Performance")

        st.write(
            f"**Model:** {metrics['model_name']}"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "MAE",
                f"{metrics['mae']:.2f}",
            )

        with col2:
            st.metric(
                "RMSE",
                f"{metrics['rmse']:.2f}",
            )

        with col3:
            st.metric(
                "R² Score",
                f"{metrics['r2']:.3f}",
            )


    # Prediction Results

    if st.session_state["predictions"] is not None:

        st.divider()

        st.subheader("Prediction Results")

        prediction_df = pd.DataFrame(
            {
                "Actual Sales": y_test.to_numpy(),
                "Predicted Sales": st.session_state["predictions"],
            }
        )

        prediction_df["Error"] = (
            prediction_df["Actual Sales"]
            - prediction_df["Predicted Sales"]
        )

        st.dataframe(
            prediction_df,
            use_container_width=True,
            hide_index=True,
        )


        # Actual vs Predicted Plot

        st.subheader("Actual vs Predicted Sales")

        fig = px.scatter(
            prediction_df,
            x="Actual Sales",
            y="Predicted Sales",
            title="Actual vs Predicted Sales",
        )

        min_value = min(
            prediction_df["Actual Sales"].min(),
            prediction_df["Predicted Sales"].min(),
        )

        max_value = max(
            prediction_df["Actual Sales"].max(),
            prediction_df["Predicted Sales"].max(),
        )

        fig.add_shape(
            type="line",
            x0=min_value,
            y0=min_value,
            x1=max_value,
            y1=max_value,
            line=dict(
                dash="dash",
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # Model Comparison

        st.divider()

        st.subheader("Model Comparison")

        st.write(
            "Compare all available regression models using the current "
            "training and testing dataset."
        )

        if st.button(
            "Compare Models",
            use_container_width=True,
        ):

            models = {
                "Linear Regression": LinearRegression(),

                "Decision Tree Regressor": DecisionTreeRegressor(
                    random_state=42
                ),

                "Random Forest Regressor": RandomForestRegressor(
                    random_state=42
                ),

                "Gradient Boosting Regressor": GradientBoostingRegressor(
                    random_state=42
                ),
            }

            comparison_results = []

            for name, comparison_model in models.items():

                comparison_model.fit(
                    X_train,
                    y_train,
                )

                comparison_pred = comparison_model.predict(
                    X_test
                )

                comparison_mae = mean_absolute_error(
                    y_test,
                    comparison_pred,
                )

                comparison_rmse = mean_squared_error(
                    y_test,
                    comparison_pred,
                ) ** 0.5

                comparison_r2 = r2_score(
                    y_test,
                    comparison_pred,
                )

                comparison_results.append(
                    {
                        "Model": name,
                        "MAE": comparison_mae,
                        "RMSE": comparison_rmse,
                        "R² Score": comparison_r2,
                    }
                )

            comparison_df = pd.DataFrame(
                comparison_results
            )

            comparison_df = comparison_df.sort_values(
                by="R² Score",
                ascending=False,
            )

            st.dataframe(
                comparison_df,
                hide_index=True,
                use_container_width=True,
            )

            st.subheader("R² Score Comparison")

            comparison_fig = px.bar(
                comparison_df,
                x="Model",
                y="R² Score",
                title="Model R² Score Comparison",
            )

            st.plotly_chart(
                comparison_fig,
                use_container_width=True,
            )