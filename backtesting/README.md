# Stock Prediction Backtesting System

This directory contains scripts to backtest stock predictions against historical data using ClickHouse.

## Setup

1.  **Credentials:** Ensure you have a `.env` file in this `backtesting` directory with your ClickHouse credentials:
    ```dotenv
    # Polygon.io API credentials (Not directly used by backtester, but often present)
    # POLYGON_API_KEY="YOUR_KEY"
    # POLYGON_API_URL="http://..."

    # ClickHouse configuration
    CLICKHOUSE_HOST=your_clickhouse_host.clickhouse.cloud
    CLICKHOUSE_HTTP_PORT=8443
    CLICKHOUSE_USER=your_user
    CLICKHOUSE_PASSWORD=your_password
    CLICKHOUSE_DATABASE=your_database_name
    CLICKHOUSE_SECURE=true
    ```

2.  **Model Files:** Place your trained prediction model (e.g., `random_forest_stock_prediction_model.pkl`) and the corresponding list of feature names used by the model (e.g., `feature_columns.pkl`) into a subdirectory named `saved_models` within this `backtesting` folder.
    ```
    backtesting/
    ├── saved_models/
    │   ├── random_forest_stock_prediction_model.pkl
    │   └── feature_columns.pkl
    ├── .env
    ├── db_utils.py
    ├── predict_historical.py
    ├── calculate_pnl.py
    ├── export_pnl_csv.py
    ├── requirements.txt
    └── README.md
    ```
    *   **Note:** Update the `MODEL_FILENAME` and `FEATURES_FILENAME` constants in `predict_historical.py` if your filenames differ.

3.  **Dependencies:** Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Source Data:** Ensure your ClickHouse database (`CLICKHOUSE_DATABASE`) contains the necessary source tables:
    *   `stock_normalized`: This table must contain the features listed in your `feature_columns.pkl` file, along with `timestamp` (DateTime64) and `ticker` (String) columns. This is the input for the prediction model.
    *   `stock_quotes`: This table must contain historical quote data with `ticker` (String), `sip_timestamp` (DateTime64), `bid_price` (Float64), and `ask_price` (Float64). **Crucially, this table should be created with `ORDER BY (ticker, sip_timestamp)`** for the P&L calculation (`ASOF JOIN`) to work correctly and efficiently.

## Workflow

Run the scripts in the following order:

1.  **Generate Historical Predictions:**
    ```bash
    python predict_historical.py
    ```
    *   This script reads data from `stock_normalized`.
    *   Loads the model from `./saved_models/`.
    *   Generates predictions (raw and categorized 0-5).
    *   Creates/replaces the `stock_historical_predictions` table in ClickHouse with the results.

2.  **Calculate Profit and Loss (P&L):**
    ```bash
    python calculate_pnl.py
    ```
    *   This script reads from `stock_historical_predictions` and `stock_quotes`.
    *   Performs `ASOF JOIN`s based on timestamps (+1m10s for entry, +16m10s for exit relative to prediction time).
    *   Calculates trade details (entry/exit prices, P&L per share, total P&L) based on prediction category (Long: 3,4,5; Short: 0,1,2).
    *   Creates/replaces the `stock_pnl` table in ClickHouse with the calculated results.

3.  **Export P&L Results to CSV:**
    ```bash
    python export_pnl_csv.py
    ```
    *   This script reads from the `stock_pnl` table.
    *   Formats the data into the specified two-row-per-trade CSV format.
    *   Saves the results to `backtesting/backtest_results.csv`.

## Output CSV Format

The `backtest_results.csv` file will have the following columns:

`Date,Time,Symbol,Quantity,Price,Side`

*   **Date:** MM/DD/YYYY (UTC)
*   **Time:** HH:MM:SS (UTC)
*   **Symbol:** Stock ticker (e.g., SPY)
*   **Quantity:** Number of shares (defined in `calculate_pnl.py`, default 100)
*   **Price:** Execution price for the trade leg (Ask for Buy/Cover, Bid for Sell/Short)
*   **Side:** "Buy" or "Sell" 