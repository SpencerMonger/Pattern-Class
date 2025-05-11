import os
from datetime import timedelta, datetime
import pandas as pd
import argparse

# Assuming db_utils is in the same directory
from db_utils import ClickHouseClient

# --- Configuration ---
# Database table names
PREDICTIONS_TABLE = "stock_historical_predictions"
QUOTES_TABLE = "stock_quotes"
# Changed PNL table name for mid-price calculation
PNL_MID_TABLE = "stock_pnl_mid"

# Share size for P&L calculation
SHARE_SIZE = 100

# Time offsets for entry and exit relative to prediction timestamp
# Entry: 1 minute + 10 seconds after prediction timestamp
# Exit: 15 minutes after entry time
ENTRY_OFFSET_SECONDS = 70  # 1 * 60 + 10
EXIT_OFFSET_SECONDS = ENTRY_OFFSET_SECONDS + (15 * 60)
# ---------------------

# Define the schema for the PnL table (remains the same structure)
# Using DateTime64(9, 'UTC') for timestamps to match predictions and quotes
PNL_SCHEMA = {
    'prediction_timestamp': "DateTime64(9, 'UTC')", # Original prediction timestamp
    'ticker': 'String',
    'prediction_raw': 'Float64',
    'prediction_cat': 'UInt8',
    'pos_long': 'UInt8',       # 1 if long, 0 otherwise
    'pos_short': 'UInt8',      # 1 if short, 0 otherwise
    'entry_timestamp': "DateTime64(9, 'UTC')",  # Timestamp for entry quote
    'exit_timestamp': "DateTime64(9, 'UTC')",   # Timestamp for exit quote
    'entry_bid_price': 'Float64', # Original entry bid
    'entry_ask_price': 'Float64', # Original entry ask
    'exit_bid_price': 'Float64',  # Original exit bid
    'exit_ask_price': 'Float64',  # Original exit ask
    'entry_mid_price': 'Float64', # Calculated entry mid-price
    'exit_mid_price': 'Float64',  # Calculated exit mid-price
    'price_diff_per_share': 'Float64', # P&L per share based on mid-price logic
    'share_size': 'UInt32',
    'pnl_total': 'Float64'      # Total P&L for the trade (price_diff_per_share * share_size)
}

# Define table engine and sorting key for PnL table
PNL_ENGINE = "ENGINE = MergeTree()" # Standard MergeTree is fine here
PNL_ORDER_BY = "ORDER BY (ticker, prediction_timestamp)"

# Batching configuration for inner loop (predictions per ticker)
PREDICTION_CHUNK_SIZE = 10000

# Buffer added to the min/max quote time range for the quote filter
# Increased buffer slightly in case the very first quote needed is just outside the prev buffer
QUOTE_TIME_BUFFER = timedelta(hours=1, minutes=10)

# Renamed function
def run_mid_pnl_calculation(start_date: str | None = None, end_date: str | None = None):
    """Connects to DB, calculates MID-PRICE P&L, and stores it in the PNL_MID_TABLE in batches using ASOF JOIN >= logic."""
    db_client = None
    try:
        # 1. Initialize Database Connection
        db_client = ClickHouseClient()
        db_name = db_client.database

        # Check if source tables exist
        if not db_client.table_exists(PREDICTIONS_TABLE):
            print(f"Error: Predictions table '{PREDICTIONS_TABLE}' not found. Please run predict_historical.py first.")
            return
        if not db_client.table_exists(QUOTES_TABLE):
            print(f"Error: Quotes table '{QUOTES_TABLE}' not found. Ensure quote data is available.")
            return

        # 2. Get list of distinct tickers to process
        print(f"Fetching distinct tickers from {PREDICTIONS_TABLE}...")
        ticker_query = f"SELECT DISTINCT ticker FROM `{db_name}`.`{PREDICTIONS_TABLE}` ORDER BY ticker"
        ticker_df = db_client.query_dataframe(ticker_query)
        if ticker_df is None or ticker_df.empty:
            print(f"No tickers found in {PREDICTIONS_TABLE}. Nothing to process.")
            return
        tickers_to_process = ticker_df['ticker'].tolist()
        print(f"Found {len(tickers_to_process)} tickers to process.")

        # 3. Drop and Recreate the PnL Table (using the new name)
        print(f"Dropping existing Mid-Price PnL table '{PNL_MID_TABLE}' (if it exists)...")
        db_client.drop_table_if_exists(PNL_MID_TABLE)
        print(f"Creating empty Mid-Price PnL table '{PNL_MID_TABLE}'...")
        # Updated PNL_SCHEMA to include mid prices
        pnl_schema_with_mid = PNL_SCHEMA.copy()
        create_table_query = f"""
        CREATE TABLE `{db_name}`.`{PNL_MID_TABLE}` (
            {', '.join([f'`{col}` {dtype}' for col, dtype in pnl_schema_with_mid.items()])}
        )
        {PNL_ENGINE}
        {PNL_ORDER_BY}
        """
        create_result = db_client.execute(create_table_query)
        if not create_result and not db_client.table_exists(PNL_MID_TABLE):
            print(f"Failed to create table '{PNL_MID_TABLE}'. Aborting.")
            return
        print(f"Successfully created or confirmed table '{PNL_MID_TABLE}'.")

        # 4. Process P&L Calculation and Insertion in Batches
        total_rows_inserted_overall = 0
        start_time_all = datetime.now()

        # Build WHERE clause for date filtering on prediction_timestamp
        pred_where_clauses = []
        if start_date:
            pred_where_clauses.append(f"timestamp >= toDateTime('{start_date} 00:00:00', 'UTC')")
        if end_date:
            pred_where_clauses.append(f"timestamp < toDateTime('{end_date}', 'UTC') + INTERVAL 1 DAY")

        prediction_date_filter = ""
        if pred_where_clauses:
            prediction_date_filter = " AND ".join(pred_where_clauses)
            print(f"Applying prediction date filter: {prediction_date_filter}")

        for i, ticker in enumerate(tickers_to_process):
            print(f"\n--- Processing ticker {i+1}/{len(tickers_to_process)}: {ticker} ---")
            start_time_ticker = datetime.now()

            # Get total predictions count for this ticker for chunking (WITH DATE FILTER)
            count_filter = f"ticker = {{ticker:String}}"
            if prediction_date_filter:
                count_filter += f" AND {prediction_date_filter}"

            count_query = f"SELECT count() FROM `{db_name}`.`{PREDICTIONS_TABLE}` WHERE {count_filter}"
            count_result = db_client.execute(count_query, params={'ticker': ticker})
            if not count_result or not count_result.result_rows:
                print(f"Could not get prediction count for ticker {ticker}. Skipping...")
                continue
            total_predictions_for_ticker = count_result.result_rows[0][0]
            if total_predictions_for_ticker == 0:
                print(f"No predictions found for {ticker} within the date range. Skipping...")
                continue
            print(f"Found {total_predictions_for_ticker} predictions for {ticker}. Processing in chunks of {PREDICTION_CHUNK_SIZE}...")

            # Inner loop for prediction chunks within the current ticker
            offset = 0
            while offset < total_predictions_for_ticker:
                print(f"  Processing chunk: offset {offset}, limit {PREDICTION_CHUNK_SIZE}")
                start_time_chunk = datetime.now()

                # --- Step 1: Fetch target times for the current chunk ---
                target_times_subquery_filter = f"ticker = {{ticker:String}}"
                if prediction_date_filter:
                    target_times_subquery_filter += f" AND {prediction_date_filter}"

                target_times_query = f"""
                SELECT
                    min(timestamp + INTERVAL {ENTRY_OFFSET_SECONDS} SECOND) AS min_entry_time,
                    max(timestamp + INTERVAL {EXIT_OFFSET_SECONDS} SECOND) AS max_exit_time
                FROM (
                    SELECT timestamp
                    FROM `{db_name}`.`{PREDICTIONS_TABLE}`
                    WHERE {target_times_subquery_filter}
                    ORDER BY timestamp
                    LIMIT {PREDICTION_CHUNK_SIZE} OFFSET {{offset:UInt64}}
                )
                """
                params_chunk_range = {'ticker': ticker, 'offset': offset}
                target_times_df = db_client.query_dataframe(target_times_query, params=params_chunk_range)

                if target_times_df is None or target_times_df.empty or target_times_df.iloc[0]['min_entry_time'] is None:
                    print(f"  Warning: Could not fetch valid time range for chunk (offset {offset}). Skipping chunk.")
                    offset += PREDICTION_CHUNK_SIZE
                    continue

                min_target_time = pd.to_datetime(target_times_df.iloc[0]['min_entry_time'], utc=True)
                max_target_time = pd.to_datetime(target_times_df.iloc[0]['max_exit_time'], utc=True)

                # --- Step 2: Determine time range for filtering quotes ---
                quote_range_start = min_target_time - QUOTE_TIME_BUFFER
                quote_range_end = max_target_time + QUOTE_TIME_BUFFER
                print(f"    Quote Time Range Filter: {quote_range_start} to {quote_range_end}")

                # --- Step 3: Construct and Execute INSERT query using ASOF JOIN (>= condition) ---
                quote_start_str = quote_range_start.strftime('%Y-%m-%d %H:%M:%S.%f')
                quote_end_str = quote_range_end.strftime('%Y-%m-%d %H:%M:%S.%f')

                prediction_chunk_filter = f"ticker = {{ticker:String}}"
                if prediction_date_filter:
                    prediction_chunk_filter += f" AND {prediction_date_filter}"

                # Modified INSERT query with an extra CTE layer for calculations
                insert_pnl_query = f"""
                INSERT INTO `{db_name}`.`{PNL_MID_TABLE}`
                WITH prediction_chunk AS (
                    -- Select predictions for the current chunk
                    SELECT
                        timestamp AS prediction_timestamp,
                        ticker,
                        prediction_raw,
                        prediction_cat,
                        prediction_timestamp + INTERVAL {ENTRY_OFFSET_SECONDS} SECOND AS target_entry_time,
                        prediction_timestamp + INTERVAL {EXIT_OFFSET_SECONDS} SECOND AS target_exit_time
                    FROM `{db_name}`.`{PREDICTIONS_TABLE}`
                    WHERE {prediction_chunk_filter}
                    ORDER BY prediction_timestamp
                    LIMIT {PREDICTION_CHUNK_SIZE} OFFSET {{offset:UInt64}}
                ),
                quotes_filtered AS (
                    -- Pre-filter quotes AND ensure correct ORDER BY for ASOF JOIN
                    SELECT sip_timestamp, ticker, bid_price, ask_price
                    FROM `{db_name}`.`{QUOTES_TABLE}`
                    WHERE ticker = {{ticker:String}}
                      AND sip_timestamp >= '{quote_start_str}'
                      AND sip_timestamp <= '{quote_end_str}'
                    ORDER BY ticker, sip_timestamp -- Crucial for ASOF JOIN
                ),
                joined_data AS (
                    -- Perform ASOF JOINs first
                    SELECT
                        pc.prediction_timestamp,
                        pc.ticker,
                        pc.prediction_raw,
                        pc.prediction_cat,
                        q_entry.sip_timestamp AS entry_timestamp,
                        q_exit.sip_timestamp AS exit_timestamp,
                        q_entry.bid_price AS entry_bid_price,
                        q_entry.ask_price AS entry_ask_price,
                        q_exit.bid_price AS exit_bid_price,
                        q_exit.ask_price AS exit_ask_price
                    FROM prediction_chunk pc
                    ASOF LEFT JOIN quotes_filtered q_entry ON pc.ticker = q_entry.ticker AND pc.target_entry_time <= q_entry.sip_timestamp
                    ASOF LEFT JOIN quotes_filtered q_exit ON pc.ticker = q_exit.ticker AND pc.target_exit_time <= q_exit.sip_timestamp
                    WHERE -- Filter out rows where ASOF JOIN didn't find a match or prices are invalid
                        entry_timestamp IS NOT NULL AND exit_timestamp IS NOT NULL AND
                        entry_bid_price > 0 AND entry_ask_price > 0 AND
                        exit_bid_price > 0 AND exit_ask_price > 0
                )
                -- Final selection directly from joined_data, calculating everything here
                -- Explicitly qualify columns with joined_data.
                SELECT
                    joined_data.prediction_timestamp,
                    joined_data.ticker,
                    joined_data.prediction_raw,
                    joined_data.prediction_cat,
                    -- Calculate pos_long/pos_short directly, qualifying source column
                    if(joined_data.prediction_cat IN (3, 4, 5), 1, 0) AS pos_long,
                    if(joined_data.prediction_cat IN (0, 1, 2), 1, 0) AS pos_short,
                    joined_data.entry_timestamp,
                    joined_data.exit_timestamp,
                    joined_data.entry_bid_price,
                    joined_data.entry_ask_price,
                    joined_data.exit_bid_price,
                    joined_data.exit_ask_price,
                    -- Calculate mid prices directly, qualifying source columns
                    (joined_data.entry_bid_price + joined_data.entry_ask_price) / 2 AS entry_mid_price,
                    (joined_data.exit_bid_price + joined_data.exit_ask_price) / 2 AS exit_mid_price,
                    -- Calculate price_diff_per_share using direct calculations, qualifying source columns
                    multiIf(
                        if(joined_data.prediction_cat IN (3, 4, 5), 1, 0) = 1, ((joined_data.exit_bid_price + joined_data.exit_ask_price) / 2) - ((joined_data.entry_bid_price + joined_data.entry_ask_price) / 2),
                        if(joined_data.prediction_cat IN (0, 1, 2), 1, 0) = 1, ((joined_data.entry_bid_price + joined_data.entry_ask_price) / 2) - ((joined_data.exit_bid_price + joined_data.exit_ask_price) / 2),
                        0
                    ) AS price_diff_per_share,
                    {SHARE_SIZE} AS share_size,
                    -- Calculate pnl_total using the price_diff_per_share alias
                    price_diff_per_share * {SHARE_SIZE} AS pnl_total
                FROM joined_data -- Removed jd alias, FROM references the CTE name
                """

                # Execute the query
                params = {'ticker': ticker, 'offset': offset}
                print(f"  Executing MID-PRICE calculation and insertion for {ticker} chunk (offset {offset})...")
                insert_result = db_client.execute(insert_pnl_query, params=params)
                end_time_chunk = datetime.now()

                if insert_result:
                    print(f"  Successfully processed mid-price chunk for {ticker} (offset {offset}). Time: {end_time_chunk - start_time_chunk}")
                else:
                    print(f"  Failed to process mid-price chunk for {ticker} (offset {offset}). Check query and logs. Skipping chunk...")

                offset += PREDICTION_CHUNK_SIZE

            end_time_ticker = datetime.now()
            print(f"--- Finished processing ticker {ticker}. Total Time: {end_time_ticker - start_time_ticker} --- ")

        end_time_all = datetime.now()
        print(f"\n--- Finished processing all tickers --- ")
        print(f"Total batch processing time: {end_time_all - start_time_all}")

        # Final count (using the new table name)
        count_result = db_client.execute(f"SELECT count() FROM `{db_name}`.`{PNL_MID_TABLE}`")
        if count_result and count_result.result_rows:
            total_rows_inserted_overall = count_result.result_rows[0][0]
            print(f"Total rows inserted into '{PNL_MID_TABLE}': {total_rows_inserted_overall}")
        else:
            print(f"Could not retrieve final row count for '{PNL_MID_TABLE}'.")

    except FileNotFoundError as e:
        print(f"Error: {e}.")
    except ValueError as e:
        print(f"Configuration or data error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during MID-PRICE P&L calculation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if db_client:
            db_client.close()

if __name__ == "__main__":
    # Updated description
    parser = argparse.ArgumentParser(description="Calculate MID-PRICE P&L based on historical predictions and quotes, using the average of bid/ask for entry/exit.")
    parser.add_argument("--start-date", type=str, help="Start date for P&L calculation range (YYYY-MM-DD). Filters predictions used.")
    parser.add_argument("--end-date", type=str, help="End date for P&L calculation range (YYYY-MM-DD). Filters predictions used.")
    args = parser.parse_args()

    # Basic validation (remains the same)
    if args.start_date and not args.end_date:
        parser.error("--start-date requires --end-date.")
    if args.end_date and not args.start_date:
        parser.error("--end-date requires --start-date.")
    if args.start_date and args.end_date:
        try:
            datetime.strptime(args.start_date, '%Y-%m-%d')
            datetime.strptime(args.end_date, '%Y-%m-%d')
            if args.start_date > args.end_date:
                 parser.error("Start date cannot be after end date.")
        except ValueError:
            parser.error("Invalid date format. Please use YYYY-MM-DD.")

    print("=== Starting MID-PRICE P&L Calculation ===")
    # Call the renamed function
    run_mid_pnl_calculation(start_date=args.start_date, end_date=args.end_date)
    print("========================================") 