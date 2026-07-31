# Sales Forecasting System

A machine learning-based sales forecasting application built using Python and Streamlit.

The application processes historical sales data, analyzes sales patterns, trains regression models, and generates future monthly sales forecasts through an interactive dashboard.

## Features

- CSV dataset upload
- Data cleaning and validation
- Sales-focused exploratory data analysis
- Monthly sales aggregation
- Lag and calendar feature generation
- Chronological train-test splitting
- Training and comparison of forecasting models
- Future monthly sales forecasting
- Interactive forecast visualizations
- CSV and PDF report export

## Models

The application supports:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Models are evaluated using:

- MAE
- RMSE
- R² Score

## Workflow

```text
Dataset Upload
      ↓
Data Cleaning
      ↓
Sales Analysis
      ↓
Forecast Preprocessing
      ↓
Model Training
      ↓
Future Forecasting
      ↓
Report Generation
```

Historical transaction data is aggregated into monthly sales. Lag features and calendar features are then generated for forecasting.

Training and testing data are split chronologically to preserve the time-series structure.

## Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly
- ReportLab

## Project Structure

```text
Sales-Forecasting-System/
├── app.py
├── pages/
│   ├── home.py
│   ├── upload.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── forecasting.py
│   └── report.py
├── src/
│   ├── eda/
│   ├── cleaning.py
│   ├── data_loader.py
│   ├── forecasting.py
│   ├── preprocessing.py
│   ├── reporting.py
│   ├── session.py
│   └── validator.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/tamrind69/Sales-Forecasting-System
cd Sales-Forecasting-System
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Dataset

The application requires a historical sales dataset containing at least:

- A date column
- A numerical sales column

Additional fields such as region, category, sub-category, and product name can be used for sales analysis.

## Output

The system generates future monthly sales forecasts and provides:

- Forecast tables
- Historical vs forecast sales visualization
- CSV export
- PDF forecast report

## Limitations

Forecast accuracy depends on the available historical data. The current system uses historical sales and calendar-based features and does not account for external factors such as promotions, holidays, pricing changes, or economic conditions.