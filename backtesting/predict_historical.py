import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import argparse
import re

# Assuming db_utils is in the same directory
from db_utils import ClickHouseClient

# --- Configuration ---
# Database table names
SOURCE_TABLE = "stock_normalized" # Table with features for prediction
TARGET_TABLE = "stock_historical_predictions"

# Model and feature paths
# Calculate path relative to the script's parent directory for portability
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Gets the parent dir (e.g., client-python-master)
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
# Ensure the directory exists before proceeding (optional but good practice)
if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"Model directory not found at calculated path: {MODEL_DIR}\nEnsure 'saved_models' exists in the parent directory of 'backtesting'.")

# MODEL_FILENAME = "random_forest_FF2022_model.pkl" # <<< REMOVED HARDCODED VALUE
# FEATURES_FILENAME = "random_forest_FF2022_feature_columns.pkl" # <<< REMOVED HARDCODED VALUE
# MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
# FEATURES_PATH = os.path.join(MODEL_DIR, FEATURES_FILENAME)

# Data fetching chunk size
CHUNK_SIZE = 10000
# ---------------------

# Define the schema for the target table
# Using DateTime64(9, 'UTC') for high precision timestamps compatible with nanoseconds
TARGET_SCHEMA = {
    'timestamp': "DateTime64(9, 'UTC')",
    'ticker': 'String',
    'prediction_raw': 'Float64', # The direct output from the model
    'prediction_cat': 'UInt8',    # Categorized prediction (0-5)
    'actual_target_5': 'UInt8',
    'actual_target_15': 'UInt8',
    'actual_target_30': 'UInt8',
    'actual_target_60': 'UInt8'
}

# Define table engine and sorting/primary key
# ReplacingMergeTree allows updating/replacing rows with the same sorting key
TARGET_ENGINE = "ENGINE = ReplacingMergeTree()"
TARGET_ORDER_BY = "ORDER BY (ticker, timestamp)" # Order matters for ReplacingMergeTree
TARGET_PRIMARY_KEY = "PRIMARY KEY (ticker, timestamp)"

def load_model_and_features(model_path: str, features_path: str) -> tuple:
    """Loads the pickled model and feature list."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found at: {features_path}")

    print(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"Loading feature columns from: {features_path}")
    with open(features_path, 'rb') as f:
        feature_columns = pickle.load(f)
        if not isinstance(feature_columns, list):
             raise TypeError(f"Expected feature_columns.pkl to contain a list, got {type(feature_columns)}")

    print(f"Loaded model and {len(feature_columns)} feature columns.")
    return model, feature_columns

def categorize_prediction(raw_prediction: float) -> int:
    """Categorizes raw prediction score into 0-5 based on defined ranges."""
    if raw_prediction < 0.5:
        return 0
    elif raw_prediction < 1.5: # 0.5 <= raw_prediction < 1.5
        return 1
    elif raw_prediction < 2.5: # 1.5 <= raw_prediction < 2.5
        return 2
    elif raw_prediction < 3.5: # 2.5 <= raw_prediction < 3.5
        return 3
    elif raw_prediction < 4.5: # 3.5 <= raw_prediction < 4.5
        return 4
    else: # raw_prediction >= 4.5
        return 5

def prepare_data_for_prediction(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Prepares the DataFrame for prediction (subsetting, cleaning)."""
    # Ensure all required columns are present
    missing_features = set(feature_columns) - set(df.columns)
    if missing_features:
        # This should ideally not happen if the source table is correct
        raise ValueError(f"Missing required feature columns in source data: {missing_features}")

    X = df[feature_columns].copy()

    # Basic cleaning (similar to model_feed.py, adapt as needed)
    for col in X.columns:
        # Convert to numeric, coercing errors
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Handle NaNs - Replace with 2.5
    if X.isna().any().any():
        print(f"Warning: NaNs found in prediction features. Filling with 2.5.")
        X.fillna(2.5, inplace=True) # Fill all NaNs in the dataframe X with 2.5

    # Handle infinities (replace with NaN first, then fill)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    if X.isna().any().any(): # Re-check NaNs after infinity replacement
        print(f"Warning: NaNs found after replacing infinities. Filling with 2.5.")
        X.fillna(2.5, inplace=True) # Fill any remaining NaNs with 2.5

    # Optional: Clip extreme values if necessary
    # Example: clip_threshold = 1e9
    # X = X.clip(-clip_threshold, clip_threshold)

    return X

def run_predictions(start_date: str | None = None, end_date: str | None = None, model_filename: str | None = None, features_filename: str | None = None):
    """Main function to fetch data, run predictions, and store results."""
    db_client = None
    try:
        # Validate filenames provided
        if not model_filename:
            raise ValueError("model_filename must be provided.")
        if not features_filename:
            raise ValueError("features_filename must be provided.")

        # Construct paths using provided filenames
        model_path = os.path.join(MODEL_DIR, model_filename)
        features_path = os.path.join(MODEL_DIR, features_filename)

        # 1. Load Model and Features
        model, feature_columns = load_model_and_features(model_path, features_path)

        # 2. Initialize Database Connection
        db_client = ClickHouseClient()

        # 3. Drop and Recreate Target Table (ensures fresh start and correct schema)
        print(f"Dropping table {TARGET_TABLE} if it exists...")
        db_client.drop_table_if_exists(TARGET_TABLE)
        print(f"Creating table {TARGET_TABLE}...")
        # Use the existing create_table_if_not_exists method
        db_client.create_table_if_not_exists(
            TARGET_TABLE,
            TARGET_SCHEMA,
            TARGET_ENGINE,
            TARGET_ORDER_BY,
            TARGET_PRIMARY_KEY
        )
        # Add a check to confirm creation or handle failure
        if not db_client.table_exists(TARGET_TABLE):
            print(f"Error: Failed to create table {TARGET_TABLE}. Aborting.")
            raise RuntimeError(f"Failed to create target table {TARGET_TABLE}")
        print(f"Table {TARGET_TABLE} is ready.")

        # 4. Fetch data from source table in chunks
        print(f"Fetching data from {SOURCE_TABLE} in chunks of {CHUNK_SIZE}...")
        total_rows_processed = 0
        offset = 0

        # Define standard target columns to fetch
        actual_target_columns = ['target_5', 'target_15', 'target_30', 'target_60']
        print(f"Will fetch actual target columns: {actual_target_columns}")

        # Ensure the actual target column is included in the features if not already
        columns_to_select = list(set(feature_columns + ['timestamp', 'ticker'] + actual_target_columns))
        # Create aliases for target columns in the SELECT statement
        select_aliases = [f"`{col}`" for col in columns_to_select if col not in actual_target_columns]
        select_aliases.extend([f"`{col}` AS actual_{col}" for col in actual_target_columns])
        select_columns_str = ", ".join(select_aliases)

        # Build WHERE clause for date filtering
        where_clauses = []
        if start_date:
            # Assuming start_date is 'YYYY-MM-DD'. Need to include the whole day.
            start_dt_str = f"toDateTime('{start_date} 00:00:00', 'UTC')"
            where_clauses.append(f"timestamp >= {start_dt_str}") # Ensure UTC
        if end_date:
            # Assuming end_date is 'YYYY-MM-DD'. Need to include the whole day up to 23:59:59.999...
            # ClickHouse toDateTime handles 'YYYY-MM-DD' and interprets it as the start of the day.
            # To include the end date, we should compare against the start of the *next* day.
            end_dt_next_day_str = f"toDateTime('{end_date}', 'UTC') + INTERVAL 1 DAY"
            where_clauses.append(f"timestamp < {end_dt_next_day_str}") # Ensure UTC

        where_clause = ""
        if where_clauses:
            where_clause = "WHERE " + " AND ".join(where_clauses)
            print(f"Applying date filter to SELECT: {where_clause}")

        while True:
            # Construct the query with optional WHERE clause, selecting specific columns
            # Alias the dynamic actual target column for easier access later
            query = f"""
            SELECT {select_columns_str}
            FROM `{db_client.database}`.`{SOURCE_TABLE}`
            {where_clause}
            ORDER BY ticker, timestamp # Ensure consistent order for chunking
            LIMIT {CHUNK_SIZE} OFFSET {offset}
            """
            print(f"Fetching rows from offset {offset}...")
            source_df = db_client.query_dataframe(query)

            if source_df is None or source_df.empty:
                print("No more data to fetch.")
                break

            print(f"Fetched {len(source_df)} rows.")

            # Ensure 'timestamp' is in UTC datetime format
            # The db_utils._ensure_utc can handle various input types
            source_df['timestamp'] = source_df['timestamp'].apply(db_client._ensure_utc)
            # Drop rows where timestamp conversion failed
            source_df.dropna(subset=['timestamp'], inplace=True)
            if source_df.empty:
                print("Skipping chunk due to timestamp conversion issues.")
                offset += CHUNK_SIZE
                continue

            # 5. Prepare Data
            X = prepare_data_for_prediction(source_df, feature_columns)

            # 6. Make Predictions
            print(f"Making predictions for {len(X)} rows...")
            try:
                raw_predictions = model.predict(X)
            except Exception as e:
                print(f"Error during model prediction: {e}")
                # Option: Skip chunk or stop? Stopping for safety.
                raise

            # 7. Prepare Results for Insertion
            results = []
            for i, index in enumerate(X.index):
                raw_pred = float(raw_predictions[i])
                category = categorize_prediction(raw_pred)
                result_row = {
                    'timestamp': source_df.loc[index, 'timestamp'],
                    'ticker': source_df.loc[index, 'ticker'],
                    'prediction_raw': raw_pred,
                    'prediction_cat': category
                }
                # Add all fetched actual target categories
                for target_col in actual_target_columns:
                    source_col_alias = f'actual_{target_col}' # Use the alias from the query
                    value = source_df.loc[index, source_col_alias]
                    # Convert to int, default to 2 if NaN/missing or conversion fails
                    try:
                        result_row[source_col_alias] = int(value) if pd.notna(value) else 2
                    except (ValueError, TypeError):
                        result_row[source_col_alias] = 2 # Default on conversion error

                results.append(result_row)

            # 8. Insert Results into Target Table
            if results:
                print(f"Inserting {len(results)} predictions into {TARGET_TABLE}...")
                db_client.insert(TARGET_TABLE, results, column_names=list(TARGET_SCHEMA.keys()))
                total_rows_processed += len(results)
            else:
                print("No results to insert for this chunk.")

            # Move to the next chunk
            offset += CHUNK_SIZE
            # Small safety break for testing, remove for full run
            # if offset >= CHUNK_SIZE * 2: # Process only 2 chunks for testing
            #     print("Stopping after 2 chunks for testing.")
            #     break

        print(f"\nPrediction process finished. Total rows processed: {total_rows_processed}")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure model and feature files exist at expected paths.")
    except ValueError as e:
        print(f"Configuration or data error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if db_client:
            db_client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate historical predictions based on normalized stock data.")
    parser.add_argument("--start-date", type=str, help="Start date for prediction range (YYYY-MM-DD). Filters source data.")
    parser.add_argument("--end-date", type=str, help="End date for prediction range (YYYY-MM-DD). Filters source data.")
    # Add arguments for standalone execution (optional, but good practice)
    parser.add_argument("--model-file", type=str, default="random_forest_FF2022_model.pkl", help="Filename of the model pickle file in saved_models.")
    parser.add_argument("--features-file", type=str, default="random_forest_FF2022_feature_columns.pkl", help="Filename of the feature columns pickle file in saved_models.")

    args = parser.parse_args()

    # Basic validation (can be improved)
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

    print("=== Starting Historical Prediction Generation ===")
    # Pass dates and filenames to the main function
    run_predictions(
        start_date=args.start_date,
        end_date=args.end_date,
        model_filename=args.model_file,
        features_filename=args.features_file
    )
    print("=================================================") 