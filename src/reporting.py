from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


def generate_forecast_pdf(
    model_name,
    metrics,
    monthly_sales,
    forecast_df,
):
    """
    Generate a PDF summary of the sales forecast.
    """

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
    )

    styles = getSampleStyleSheet()

    elements = []


    # Title

    elements.append(
        Paragraph(
            "Sales Forecast Report",
            styles["Title"],
        )
    )

    elements.append(
        Spacer(1, 0.25 * inch)
    )


    # Report Overview

    elements.append(
        Paragraph(
            "Report Overview",
            styles["Heading2"],
        )
    )

    overview_data = [
        ["Metric", "Value"],
        ["Historical Months", str(len(monthly_sales))],
        ["Forecast Months", str(len(forecast_df))],
        ["Forecasting Model", model_name],
    ]

    overview_table = Table(
        overview_data,
        colWidths=[
            2.5 * inch,
            2.5 * inch,
        ],
    )

    overview_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    elements.append(overview_table)

    elements.append(
        Spacer(1, 0.25 * inch)
    )


    # Model Performance

    elements.append(
        Paragraph(
            "Model Performance",
            styles["Heading2"],
        )
    )

    performance_data = [
        ["Metric", "Value"],
        ["MAE", f"{metrics['mae']:,.2f}"],
        ["RMSE", f"{metrics['rmse']:,.2f}"],
        ["R² Score", f"{metrics['r2']:.3f}"],
    ]

    performance_table = Table(
        performance_data,
        colWidths=[
            2.5 * inch,
            2.5 * inch,
        ],
    )

    performance_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    elements.append(performance_table)

    elements.append(
        Spacer(1, 0.25 * inch)
    )


    # Forecast Summary

    elements.append(
        Paragraph(
            "Forecast Summary",
            styles["Heading2"],
        )
    )

    forecast_total = forecast_df[
        "Predicted Sales"
    ].sum()

    forecast_average = forecast_df[
        "Predicted Sales"
    ].mean()

    forecast_highest = forecast_df[
        "Predicted Sales"
    ].max()

    summary_data = [
        ["Metric", "Value"],
        ["Total Forecast Sales", f"{forecast_total:,.2f}"],
        ["Average Monthly Forecast", f"{forecast_average:,.2f}"],
        ["Highest Monthly Forecast", f"{forecast_highest:,.2f}"],
    ]

    summary_table = Table(
        summary_data,
        colWidths=[
            2.5 * inch,
            2.5 * inch,
        ],
    )

    summary_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    elements.append(summary_table)

    elements.append(
        Spacer(1, 0.25 * inch)
    )


    # Future Forecast

    elements.append(
        Paragraph(
            "Future Sales Forecast",
            styles["Heading2"],
        )
    )

    forecast_table_data = [
        ["Date", "Predicted Sales"]
    ]

    for _, row in forecast_df.iterrows():

        forecast_table_data.append(
            [
                row.iloc[0].strftime("%b %Y"),
                f"{row['Predicted Sales']:,.2f}",
            ]
        )

    forecast_table = Table(
        forecast_table_data,
        colWidths=[
            2.5 * inch,
            2.5 * inch,
        ],
    )

    forecast_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    elements.append(forecast_table)


    # Build PDF

    document.build(elements)

    buffer.seek(0)

    return buffer.getvalue()