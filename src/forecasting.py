import pandas as pd


def generate_future_forecast(
    model,
    monthly_sales: pd.DataFrame,
    date_column: str,
    target_column: str,
    forecast_months: int,
):
    """
    Generate recursive monthly sales forecasts.

    Future predictions are added to the sales history so
    they can be used as lag features for later forecasts.
    """

    history = monthly_sales.copy()

    history[date_column] = pd.to_datetime(
        history[date_column]
    )

    history = (
        history
        .sort_values(date_column)
        .reset_index(drop=True)
    )

    forecast_results = []


    for _ in range(forecast_months):

        # Next month to forecast

        next_date = (
            history[date_column].iloc[-1]
            + pd.offsets.MonthBegin(1)
        )


        # Build features for the next month

        features = pd.DataFrame(
            {
                "Lag_1": [
                    history[target_column].iloc[-1]
                ],

                "Lag_2": [
                    history[target_column].iloc[-2]
                ],

                "Lag_3": [
                    history[target_column].iloc[-3]
                ],

                "Lag_12": [
                    history[target_column].iloc[-12]
                ],

                "Month": [
                    next_date.month
                ],

                "Quarter": [
                    next_date.quarter
                ],
            }
        )


        # Predict future sales

        predicted_sales = model.predict(
            features
        )[0]


        # Store forecast result

        forecast_results.append(
            {
                date_column: next_date,
                "Predicted Sales": predicted_sales,
            }
        )


        # Add prediction to history

        new_row = pd.DataFrame(
            {
                date_column: [next_date],
                target_column: [predicted_sales],
            }
        )

        history = pd.concat(
            [
                history,
                new_row,
            ],
            ignore_index=True,
        )


    forecast_df = pd.DataFrame(
        forecast_results
    )

    return forecast_df