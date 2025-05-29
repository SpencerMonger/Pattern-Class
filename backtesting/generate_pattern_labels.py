import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import argparse
import sys
import yaml

# Restoring original sys.path manipulations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add parent dir for db_utils
OHLC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ohlc_detection')
if OHLC_DIR not in sys.path:
    sys.path.insert(0, OHLC_DIR) # Insert ohlc_detection path

try:
    from db_utils import ClickHouseClient # Reverted to direct import
    from ohlc_detection.label_shapes import generate_labels_from_db
except ImportError as e:
    print(f"Error importing required modules in generate_pattern_labels.py: {e}")
    # Adjusted error message to reflect restored import style
    print("Ensure db_utils.py can be found (possibly via parent directory in sys.path) and label_shapes.py is in 'ohlc_detection' directory (also in sys.path).")
    sys.exit(1)

# --- Configuration --- 
# Target table for storing generated labels
TARGET_TABLE = "stock_historical_labels" # <<< New table name

# Data insertion chunk size (how many rows to insert at once)
# Applied *after* all labels are generated
INSERT_CHUNK_SIZE = 50000
# ---------------------

# Define table engine and sorting/primary key
# ReplacingMergeTree allows updating/replacing rows with the same sorting key
TARGET_ENGINE = "ENGINE = ReplacingMergeTree()"
TARGET_ORDER_BY = "ORDER BY (ticker, timestamp)" # Order matters for ReplacingMergeTree
TARGET_PRIMARY_KEY = "PRIMARY KEY (ticker, timestamp)"

# Helper function to map pandas dtypes to ClickHouse types (simplified)
def map_pandas_to_clickhouse(dtype):
    if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
        # Assume high precision UTC datetime
        return "DateTime64(9, 'UTC')"
    elif pd.api.types.is_integer_dtype(dtype):
        # Use Int64 as a general default for integers
        return 'Int64' # Could use UInt64 if known non-negative
    elif pd.api.types.is_float_dtype(dtype):
        return 'Float64'
    elif pd.api.types.is_bool_dtype(dtype):
        return 'UInt8' # Booleans often stored as 0/1
    # Default to String for object or other types
    return 'String'

def run_label_generation(config: Dict, start_date: str | None = None, end_date: str | None = None):
    """Main function to fetch OHLCV data, generate pattern labels, and store results."""
    db_client = None
    try:
        # 1. Initialize Database Connection
        db_client = ClickHouseClient()
        db_name = db_client.database # Get database name

        # --- Get pattern_directions from config --- #
        label_gen_config = config.get('label_generation', {})
        pattern_directions = label_gen_config.get('pattern_directions', {})
        if not pattern_directions:
            print("Warning: 'pattern_directions' not found or empty in config. All patterns will be attempted.")
        else:
            print(f"Using pattern_directions from config: {len(pattern_directions)} entries.")

        # 2. Generate Labels using the refactored function
        print("Generating pattern labels from database...")
        # Returns the full OHLCV dataframe with 'pattern_label' column added
        labels_df = generate_labels_from_db(db_client, config, pattern_directions) # <<< Pass pattern_directions

        if labels_df.empty:
            print("Label generation returned no data. Nothing to insert.")
            return

        # --- Dynamically Create Target Schema --- #
        labels_df_reset = labels_df.reset_index() # Get timestamp as column
        dynamic_target_schema = {}
        print("\nDynamically determining target schema from DataFrame:")
        for col_name, dtype in labels_df_reset.dtypes.items():
            ch_type = map_pandas_to_clickhouse(dtype)
            dynamic_target_schema[col_name] = ch_type
            print(f"  - Column: '{col_name}', Pandas dtype: {dtype}, ClickHouse type: {ch_type}")
        # Ensure essential columns are present
        for key_col in ['timestamp', 'ticker']:
             if key_col not in dynamic_target_schema:
                  raise ValueError(f"Essential column '{key_col}' missing from generated DataFrame.")
        if 'pattern_label' not in dynamic_target_schema:
             print("Warning: 'pattern_label' column missing, adding as String.")
             dynamic_target_schema['pattern_label'] = 'String'
        
        # --- Add macd_allowed to the schema if not present --- #
        # This will be the target column for the MACD predicate True/False
        # Defaulting to 0 (False) initially. It will be updated by a later step.
        if 'macd_allowed' not in dynamic_target_schema:
            print("Adding 'macd_allowed' (UInt8) to target schema, default 0.")
            dynamic_target_schema['macd_allowed'] = 'UInt8'
        # --- End Schema Creation --- #

        # 3. Create or Alter Target Table using dynamic schema
        if not db_client.table_exists(TARGET_TABLE):
            print(f"Table {TARGET_TABLE} does not exist. Creating with dynamic schema...")
            db_client.create_table_if_not_exists(
                TARGET_TABLE,
                dynamic_target_schema, 
                TARGET_ENGINE,
                TARGET_ORDER_BY,
                TARGET_PRIMARY_KEY
            )
            if not db_client.table_exists(TARGET_TABLE):
                raise RuntimeError(f"Failed to create target table {TARGET_TABLE}")
            print(f"Table {TARGET_TABLE} created successfully.")
        else:
            print(f"Table {TARGET_TABLE} already exists. Checking for 'macd_allowed' column...")
            # Check if macd_allowed column exists
            table_columns = db_client.get_table_columns(TARGET_TABLE)
            if 'macd_allowed' not in table_columns:
                print(f"Column 'macd_allowed' not found in {TARGET_TABLE}. Adding column...")
                alter_query = f"ALTER TABLE `{db_name}`.`{TARGET_TABLE}` ADD COLUMN macd_allowed UInt8 DEFAULT 0"
                db_client.execute(alter_query)
                print("Column 'macd_allowed' added successfully with default 0.")
            else:
                print("Column 'macd_allowed' already exists.")
        
        print(f"Table {TARGET_TABLE} is ready.")

        # 4. Prepare data for insertion
        # Add macd_allowed column to the DataFrame if it's not there from label generation step
        if 'macd_allowed' not in labels_df_reset.columns:
            labels_df_reset['macd_allowed'] = 0 # Default to 0 (False)
        else:
            # If it somehow came from generate_labels_from_db, ensure it's int for ClickHouse UInt8
            labels_df_reset['macd_allowed'] = labels_df_reset['macd_allowed'].fillna(0).astype(int)

        # Ensure columns match the dynamic schema keys (for correct order)
        columns_to_insert = list(dynamic_target_schema.keys())
        labels_df_final = labels_df_reset[columns_to_insert] # Select/reorder based on dynamic schema

        # Convert DataFrame rows to list of dictionaries for insertion
        data_to_insert = labels_df_final.to_dict('records')

        # 5. Insert Results into Target Table in Chunks
        total_rows_inserted = 0
        num_records = len(data_to_insert)
        print(f"\nPreparing to insert {num_records} rows into {TARGET_TABLE}...")

        for i in range(0, num_records, INSERT_CHUNK_SIZE):
            chunk = data_to_insert[i : i + INSERT_CHUNK_SIZE]
            if chunk:
                print(f"Inserting chunk {i // INSERT_CHUNK_SIZE + 1}/{(num_records + INSERT_CHUNK_SIZE - 1) // INSERT_CHUNK_SIZE} ({len(chunk)} rows)...")
                # Pass the actual column names used for insertion
                db_client.insert(TARGET_TABLE, chunk, column_names=columns_to_insert)
                total_rows_inserted += len(chunk)

        print(f"\nLabel generation and insertion finished. Total rows inserted: {total_rows_inserted}")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure required files exist.") # Less likely now
    except ValueError as e:
        print(f"Configuration or data error during label generation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during label generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if db_client:
            db_client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate historical pattern labels based on OHLC stock data.") # <<< Updated description
    # Arguments needed for standalone execution match run_backtest.py
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
        help="Path to the YAML configuration file."
    )
    parser.add_argument("--start-date", type=str, help="Start date for label generation range (YYYY-MM-DD). Overrides config.")
    parser.add_argument("--end-date", type=str, help="End date for label generation range (YYYY-MM-DD). Overrides config.")

    args = parser.parse_args()

    # --- Load config for standalone run --- #
    config = {}
    try:
        print(f"Loading configuration from: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        if config is None: config = {}
    except FileNotFoundError:
        print(f"Error: Config file not found at '{args.config}'")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file '{args.config}': {e}")
        sys.exit(1)

    # --- Determine dates --- #
    start_date_arg = args.start_date or config.get('start_date')
    end_date_arg = args.end_date or config.get('end_date')

    # Basic validation
    if start_date_arg and not end_date_arg:
        parser.error("Config or --start-date requires --end-date.")
    if end_date_arg and not start_date_arg:
        parser.error("Config or --end-date requires --start-date.")
    if start_date_arg and end_date_arg:
        try:
            datetime.strptime(start_date_arg, '%Y-%m-%d')
            datetime.strptime(end_date_arg, '%Y-%m-%d')
            if start_date_arg > end_date_arg:
                 parser.error("Start date cannot be after end date.")
            # Add validated dates back to config for the run function
            config['start_date'] = start_date_arg
            config['end_date'] = end_date_arg
        except ValueError:
            parser.error("Invalid date format. Please use YYYY-MM-DD.")

    print("=== Starting Historical Pattern Label Generation ===")
    # Pass the whole config dictionary
    run_label_generation(
        config=config,
        start_date=config.get('start_date'), # Pass dates explicitly too for clarity
        end_date=config.get('end_date')
    )
    print("===================================================") 