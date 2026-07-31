from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.widgets.markers import makeMarker

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


# Forecast Chart

def create_forecast_chart(
    monthly_sales,
    forecast_df,
    date_column,
    target_column,
):
    """
    Create historical and forecast sales chart
    for the PDF report.
    """

    historical_dates = (
        monthly_sales[date_column]
        .tolist()
    )

    historical_sales = (
        monthly_sales[target_column]
        .astype(float)
        .tolist()
    )

    forecast_dates = (
        forecast_df[date_column]
        .tolist()
    )

    forecast_sales = (
        forecast_df["Predicted Sales"]
        .astype(float)
        .tolist()
    )


    # Historical points

    historical_points = [
        (index, value)
        for index, value in enumerate(
            historical_sales
        )
    ]


    # Forecast starts at final historical point

    forecast_start_index = (
        len(historical_sales) - 1
    )

    forecast_points = [
        (
            forecast_start_index,
            historical_sales[-1],
        )
    ]

    forecast_points.extend(
        [
            (
                forecast_start_index + index,
                value,
            )
            for index, value in enumerate(
                forecast_sales,
                start=1,
            )
        ]
    )


    # Complete timeline

    all_dates = (
        historical_dates
        + forecast_dates
    )


    # Create drawing

    drawing = Drawing(
        500,
        270,
    )

    chart = LinePlot()

    chart.x = 60
    chart.y = 50
    chart.width = 390
    chart.height = 170

    chart.data = [
        historical_points,
        forecast_points,
    ]


    # Historical line

    chart.lines[0].strokeColor = colors.blue
    chart.lines[0].strokeWidth = 1.5

    chart.lines[0].symbol = makeMarker(
        "FilledCircle"
    )

    chart.lines[0].symbol.size = 4


    # Forecast line

    chart.lines[1].strokeColor = colors.red
    chart.lines[1].strokeWidth = 1.5

    chart.lines[1].symbol = makeMarker(
        "FilledCircle"
    )

    chart.lines[1].symbol.size = 4


    # Y Axis

    all_sales = (
        historical_sales
        + forecast_sales
    )

    minimum_sales = min(all_sales)
    maximum_sales = max(all_sales)

    sales_range = (
        maximum_sales - minimum_sales
    )

    if sales_range == 0:
        sales_range = 1

    padding = sales_range * 0.1

    chart.yValueAxis.valueMin = max(
        0,
        minimum_sales - padding,
    )

    chart.yValueAxis.valueMax = (
        maximum_sales + padding
    )

    chart.yValueAxis.visibleGrid = True

    chart.yValueAxis.gridStrokeColor = (
        colors.lightgrey
    )

    chart.yValueAxis.labels.fontSize = 7


    # X Axis

    chart.xValueAxis.valueMin = 0

    chart.xValueAxis.valueMax = (
        len(all_dates) - 1
    )


    # Show approximately one date label per year

    tick_step = 12

    tick_positions = list(
        range(
            0,
            len(all_dates),
            tick_step,
        )
    )

    # Include final forecast date

    final_position = len(all_dates) - 1

    if final_position not in tick_positions:
        tick_positions.append(
            final_position
        )

    chart.xValueAxis.valueSteps = (
        tick_positions
    )


    # Convert numeric positions into date labels

    def format_date(value):

        index = int(round(value))

        if 0 <= index < len(all_dates):

            return all_dates[
                index
            ].strftime("%b %Y")

        return ""


    chart.xValueAxis.labelTextFormat = (
        format_date
    )

    chart.xValueAxis.labels.fontSize = 7


    # Legend

    legend = Legend()

    legend.x = 180
    legend.y = 245

    legend.alignment = "right"

    legend.colorNamePairs = [
        (
            colors.blue,
            "Historical",
        ),
        (
            colors.red,
            "Forecast",
        ),
    ]


    drawing.add(chart)
    drawing.add(legend)

    return drawing


# Forecast PDF

def generate_forecast_pdf(
    model_name,
    metrics,
    monthly_sales,
    forecast_df,
    date_column,
    target_column,
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
        [
            "Metric",
            "Value",
        ],
        [
            "Historical Months",
            str(len(monthly_sales)),
        ],
        [
            "Forecast Months",
            str(len(forecast_df)),
        ],
        [
            "Forecasting Model",
            model_name,
        ],
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
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    elements.append(
        overview_table
    )

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
        [
            "Metric",
            "Value",
        ],
        [
            "MAE",
            f"{metrics['mae']:,.2f}",
        ],
        [
            "RMSE",
            f"{metrics['rmse']:,.2f}",
        ],
        [
            "R² Score",
            f"{metrics['r2']:.3f}",
        ],
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
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    elements.append(
        performance_table
    )

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

    forecast_total = (
        forecast_df[
            "Predicted Sales"
        ].sum()
    )

    forecast_average = (
        forecast_df[
            "Predicted Sales"
        ].mean()
    )

    forecast_highest = (
        forecast_df[
            "Predicted Sales"
        ].max()
    )

    summary_data = [
        [
            "Metric",
            "Value",
        ],
        [
            "Total Forecast Sales",
            f"{forecast_total:,.2f}",
        ],
        [
            "Average Monthly Forecast",
            f"{forecast_average:,.2f}",
        ],
        [
            "Highest Monthly Forecast",
            f"{forecast_highest:,.2f}",
        ],
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
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    elements.append(
        summary_table
    )

    elements.append(
        Spacer(1, 0.25 * inch)
    )


    # Historical + Forecast Chart

    elements.append(
        Paragraph(
            "Historical and Forecast Sales",
            styles["Heading2"],
        )
    )

    forecast_chart = create_forecast_chart(
        monthly_sales=monthly_sales,
        forecast_df=forecast_df,
        date_column=date_column,
        target_column=target_column,
    )

    elements.append(
        forecast_chart
    )

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
        [
            "Date",
            "Predicted Sales",
        ]
    ]

    for _, row in forecast_df.iterrows():

        forecast_table_data.append(
            [
                row[
                    date_column
                ].strftime("%b %Y"),

                f"{row['Predicted Sales']:,.2f}",
            ]
        )

    forecast_table = Table(
        forecast_table_data,
        colWidths=[
            2.5 * inch,
            2.5 * inch,
        ],
        repeatRows=1,
    )

    forecast_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    elements.append(
        forecast_table
    )


    # Build PDF

    document.build(
        elements
    )

    buffer.seek(0)

    return buffer.getvalue()