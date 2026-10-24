# Apple Stock Price Prediction Using LSTM

A time-series forecasting project that uses a Long Short-Term Memory (LSTM) neural network to predict Apple Inc. (AAPL) closing stock prices from historical market data. The project includes a Jupyter notebook, a Python script, and a CSV dataset.

## Project Overview

The model learns patterns from the previous 60 trading observations and predicts the next closing price. Predictions are compared with actual prices using regression metrics and visualizations.

**Model:** Stacked LSTM neural network  
**Target variable:** `Close`  
**Sequence length:** 60 observations  
**Train/test split:** Chronological 80/20 split of generated sequences  
**Training configuration:** 65 epochs, batch size 32, Adam optimizer  
**Loss function:** Mean squared error

## Dataset

The included `apple_stock.csv` contains **2,475 rows** and seven columns, covering **January 2, 2014, through October 31, 2023**. The supplied file has no missing values.

| Column | Description |
|---|---|
| `Date` | Trading date |
| `Open` | Opening price |
| `High` | Highest price during the trading session |
| `Low` | Lowest price during the trading session |
| `Close` | Closing price; the prediction target |
| `Adj Close` | Adjusted closing price |
| `Volume` | Trading volume |

Although the dataset includes several market features, the current model uses only the `Close` column.

## Workflow

1. Load the historical stock-price dataset and parse the date column.
2. Inspect the data and check for missing values.
3. Plot distributions for the available numerical columns.
4. Normalize closing prices to the range 0–1 with `MinMaxScaler`.
5. Create input sequences of 60 observations, with the following observation as each target.
6. Split the sequences chronologically into training and testing sets (80% and 20%).
7. Train a stacked LSTM model with two LSTM layers containing 50 units each, followed by a one-unit dense output layer.
8. Generate predictions for the test set and convert predictions and targets back to the original price scale.
9. Evaluate predictions using RMSE, MSE, MAE, R², and MAPE.
10. Display actual-versus-predicted line and scatter plots.

## Model Architecture

```text
60 previous closing-price observations
              |
       LSTM (50 units,
   return_sequences=True)
              |
        LSTM (50 units)
              |
       Dense (1 output)
              |
   Predicted closing price
```

## Repository Contents

```text
APPLE-STOCK-PRICE-PREDICTION-USING-LSTM/
├── apple_stock.csv
├── TASK_2_STOCK_PRICE_PREDICTION_USING_LSTM.ipynb
├── task_2_stock_price_prediction_using_lstm.py
└── APPLE COMPANY STOCK PRICE PREDICTION.zip
```

The ZIP file contains another copy of `apple_stock.csv`. The notebook is the convenient option for exploring the workflow and viewing plots step by step.

## Requirements

The code imports the following Python libraries:

- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- TensorFlow (Keras)

Install the dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

## How to Run

### Option 1: Run in Google Colab

1. Upload `apple_stock.csv` to the Colab environment.
2. Open `TASK_2_STOCK_PRICE_PREDICTION_USING_LSTM.ipynb` in Google Colab.
3. Update the dataset path in the loading cell if necessary. The supplied code expects `/content/sample_data/apple_stock.csv`.
4. Run the notebook cells in order.

### Option 2: Run locally

1. Clone or download this repository.
2. Install the dependencies listed above.
3. Open the notebook in Jupyter and run the cells in order, or update the CSV path in `task_2_stock_price_prediction_using_lstm.py` before running the script.

The Python script currently uses a Google Colab-specific absolute path (`/content/sample_data/apple_stock.csv`), so it must be changed to the location of the CSV on your computer when running locally.

## Evaluation

The project calculates:

- **RMSE:** Root Mean Squared Error
- **MSE:** Mean Squared Error
- **MAE:** Mean Absolute Error
- **R²:** Coefficient of determination
- **MAPE:** Mean Absolute Percentage Error

It also creates a line plot and a scatter plot comparing predicted and actual closing prices. The code computes these metrics when run; no verified metric values are documented here because they are not provided as fixed results in the repository.

## Notes and Limitations

- **Potential data leakage:** The scaler is fitted on the full closing-price series before the train/test split. For a more reliable evaluation, fit the scaler on training data only, then transform the test data using that fitted scaler.
- **Single-feature model:** The current model uses only historical closing prices, not `Open`, `High`, `Low`, `Adj Close`, or `Volume`.
- **Not a trading system:** The model is an educational forecasting project. It does not account for transaction costs, market events, or investment risk, and its outputs should not be treated as financial advice.
- **No saved model:** The provided code trains the model when executed; it does not save a trained model for later inference.
- **Reproducibility:** Results may vary between runs because random seeds are not explicitly fixed.

## Future Improvements

- Prevent data leakage by fitting preprocessing steps on training data only.
- Compare LSTM results with baseline forecasts and other time-series models.
- Add error plots and report the metric values from a reproducible run.
- Save the trained model and scaler for reuse.
- Evaluate performance across different time periods and market conditions.

## Disclaimer

This project is for learning and experimentation with time-series forecasting. It is not financial advice and should not be used as the sole basis for investment decisions.
