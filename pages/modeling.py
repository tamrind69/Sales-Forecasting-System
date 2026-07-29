import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def show():
    st.set_page_config(page_title="Model Training")

    st.title("🤖 Forecast Model Training")


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

    train_dates = preprocessing["train_dates"]
    test_dates = preprocessing["test_dates"]

    feature_columns = preprocessing["feature_columns"]


    # Dataset Summary

    st.subheader("Dataset Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Training Months",
            len(X_train),
        )

    with col2:
        st.metric(
            "Testing Months",
            len(X_test),
        )

    with col3:
        st.metric(
            "Forecast Features",
            len(feature_columns),
        )

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

    st.divider()


    # Forecast Features

    st.subheader("Forecast Features")

    feature_df = pd.DataFrame(
        {
            "Feature Name": feature_columns,
        }
    )

    st.dataframe(
        feature_df,
        hide_index=True,
        use_container_width=True,
    )


    # Training Dataset

    st.subheader("Training Dataset")

    training_df = X_train.copy()

    training_df["Target Sales"] = y_train.to_numpy()

    st.dataframe(
        training_df.head(),
        hide_index=True,
        use_container_width=True,
    )


    # Testing Dataset

    st.subheader("Testing Dataset")

    testing_df = X_test.copy()

    testing_df["Target Sales"] = y_test.to_numpy()

    st.dataframe(
        testing_df.head(),
        hide_index=True,
        use_container_width=True,
    )

    st.divider()


    # Model Configuration

    st.subheader("Model Configuration")

    model_name = st.selectbox(
        "Forecasting Model",
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
                random_state=42,
            )

        elif model_name == "Random Forest Regressor":
            model = RandomForestRegressor(
                random_state=42,
            )

        elif model_name == "Gradient Boosting Regressor":
            model = GradientBoostingRegressor(
                random_state=42,
            )


        # Train forecasting model

        model.fit(
            X_train,
            y_train,
        )


        # Predict later test months

        y_pred = model.predict(
            X_test
        )


        # Evaluation Metrics

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
        st.session_state["model_name"] = model_name
        st.session_state["predictions"] = y_pred

        st.session_state["metrics"] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }


        # Previous future forecast is invalid

        st.session_state["forecasting"] = {
            "forecast_df": None,
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
            f"**Model:** {st.session_state.model_name}"
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

        st.subheader("Test Period Predictions")

        prediction_df = pd.DataFrame(
            {
                "Date": test_dates.to_numpy(),
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
            hide_index=True,
            use_container_width=True,
        )


        # Actual vs Predicted Over Time

        st.subheader("Actual vs Predicted Sales")

        plot_df = prediction_df.melt(
            id_vars="Date",
            value_vars=[
                "Actual Sales",
                "Predicted Sales",
            ],
            var_name="Series",
            value_name="Sales",
        )

        fig = px.line(
            plot_df,
            x="Date",
            y="Sales",
            color="Series",
            markers=True,
            title="Actual vs Predicted Monthly Sales",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


        # Model Comparison

        st.divider()

        st.subheader("Model Comparison")

        st.write(
            "Compare forecasting models and a seasonal baseline "
            "using the same chronological testing period."
        )

        if st.button(
            "Compare Models",
            use_container_width=True,
        ):

            models = {
                "Linear Regression": LinearRegression(),

                "Decision Tree Regressor": DecisionTreeRegressor(
                    random_state=42,
                ),

                "Random Forest Regressor": RandomForestRegressor(
                    random_state=42,
                ),

                "Gradient Boosting Regressor": GradientBoostingRegressor(
                    random_state=42,
                ),
            }

            comparison_results = []

            # Machine Learning Models

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


            # Seasonal Naive Baseline

            seasonal_pred = X_test[
                "Lag_12"
            ].to_numpy()

            seasonal_mae = mean_absolute_error(
                y_test,
                seasonal_pred,
            )

            seasonal_rmse = mean_squared_error(
                y_test,
                seasonal_pred,
            ) ** 0.5

            seasonal_r2 = r2_score(
                y_test,
                seasonal_pred,
            )

            comparison_results.append(
                {
                    "Model": "Seasonal Naive Baseline",
                    "MAE": seasonal_mae,
                    "RMSE": seasonal_rmse,
                    "R² Score": seasonal_r2,
                }
            )


            # Comparison Table

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


            # R² Comparison Plot

            st.subheader("R² Score Comparison")

            comparison_fig = px.bar(
                comparison_df,
                x="Model",
                y="R² Score",
                title="Forecast Model R² Comparison",
            )

            st.plotly_chart(
                comparison_fig,
                use_container_width=True,
            )