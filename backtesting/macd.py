import pandas as pd
from typing import Dict, Optional
import os

from db_utils import ClickHouseClient

# Constants for the MACD logic
HARDCODED_MACD_SOURCE_TABLE = "stock_master"
HARDCODED_MACD_VALUE_COL = "macd_value_d"
HARDCODED_MACD_SIGNAL_COL = "macd_signal_d"
TARGET_LABELS_TABLE = "stock_historical_labels"

def update_labels_with_macd_info(
    db_client: ClickHouseClient,
    config: Dict, # For use_macd_predicate flag and potentially dates if not passed directly
    start_date: Optional[str],
    end_date: Optional[str]
) -> None:
    """    Fetches MACD data from stock_master, calculates the MACD condition,
    and updates the 'macd_allowed' column in stock_historical_labels.
    Assumes stock_historical_labels is a ReplacingMergeTree table.
    """
    use_macd_predicate = config.get('label_generation', {}).get('use_macd_predicate', False)

    if not use_macd_predicate:
        print("MACD predicate is not enabled in config. Skipping MACD update to labels table.")
        return

    if not db_client.table_exists(TARGET_LABELS_TABLE):
        print(f"Error: Target labels table '{TARGET_LABELS_TABLE}' does not exist. Cannot update MACD info.")
        return
    
    # Ensure macd_allowed column exists, if not, something went wrong in generate_pattern_labels
    labels_table_cols = db_client.get_table_columns(TARGET_LABELS_TABLE)
    if 'macd_allowed' not in labels_table_cols:
        print(f"Error: Column 'macd_allowed' not found in '{TARGET_LABELS_TABLE}'. MACD update cannot proceed.")
        print("Please ensure generate_pattern_labels.py ran correctly and added the column.")
        return

    print(f"Fetching MACD data from '{HARDCODED_MACD_SOURCE_TABLE}' to update '{TARGET_LABELS_TABLE}'.")
    db_name = db_client.database

    # 1. Construct date filter for querying both tables
    date_filters = []
    if start_date:
        date_filters.append(f"lbl.timestamp >= toDateTime('{start_date} 00:00:00', 'UTC')")
    if end_date:
        date_filters.append(f"lbl.timestamp < (toDate('{end_date}') + INTERVAL 1 DAY)")
    
    # Combined WHERE conditions
    combined_where_conditions = []
    if date_filters:
        combined_where_conditions.extend(date_filters)
    
    combined_where_conditions.append(f"m.`{HARDCODED_MACD_VALUE_COL}` IS NOT NULL")
    combined_where_conditions.append(f"m.`{HARDCODED_MACD_SIGNAL_COL}` IS NOT NULL")

    where_clause_str = ""
    if combined_where_conditions:
        where_clause_str = "WHERE " + " AND ".join(combined_where_conditions)
    
    # 2. Query to get (ticker, timestamp, macd_value, macd_signal) from stock_master
    #    and join with stock_historical_labels to get all existing columns from labels table
    #    for the records that will be updated. We only care about rows that exist in stock_historical_labels.
    #    This ensures we only update existing label entries.
    
    # Get all columns from TARGET_LABELS_TABLE to select them for re-insertion
    # We need to preserve all other data in the labels table.
    select_cols_from_labels = [f"lbl.`{col_name}`" for col_name in labels_table_cols if col_name != 'macd_allowed']
    select_cols_str = ", ".join(select_cols_from_labels)

    # Note: Casting macd_value and macd_signal to Float64 to handle potential type issues during comparison or if they are Nullable.
    # The primary keys (ticker, timestamp) will ensure ReplacingMergeTree updates correctly.
    update_query = f"""
    SELECT
        {select_cols_str},
        CAST(m.`{HARDCODED_MACD_VALUE_COL}` AS Float64) >= CAST(m.`{HARDCODED_MACD_SIGNAL_COL}` AS Float64) AS macd_allowed_calc
    FROM `{db_name}`.`{HARDCODED_MACD_SOURCE_TABLE}` AS m
    INNER JOIN `{db_name}`.`{TARGET_LABELS_TABLE}` AS lbl
      ON m.ticker = lbl.ticker AND m.timestamp = lbl.timestamp
    {where_clause_str} -- Applies to relevant tables based on column qualification
    """
    # The WHERE clause above ensures we only process rows where MACD values are available
    # and optionally filters by date on the labels table.

    print(f"Executing query to fetch data for MACD update:\n{update_query}")
    try:
        data_for_update_df = db_client.query_dataframe(update_query)

        if data_for_update_df is None or data_for_update_df.empty:
            print(f"No (ticker, timestamp) pairs found in '{TARGET_LABELS_TABLE}' that have corresponding MACD data in '{HARDCODED_MACD_SOURCE_TABLE}' for the date range, or MACD values were NULL.")
            print("No labels will be updated with MACD information.")
            return

        print(f"Found {len(data_for_update_df)} label entries to update with MACD information.")

        # Rename columns from 'lbl.column_name' to 'column_name' if they exist
        cols_to_rename_prefix = {col: col.replace('lbl.', '') for col in data_for_update_df.columns if col.startswith('lbl.')}
        if cols_to_rename_prefix:
            print(f"Removing 'lbl.' prefix from DataFrame columns: {cols_to_rename_prefix}")
            data_for_update_df.rename(columns=cols_to_rename_prefix, inplace=True)

        # Rename calculated column 'macd_allowed_calc' to 'macd_allowed'
        data_for_update_df.rename(columns={'macd_allowed_calc': 'macd_allowed'}, inplace=True)
        # Convert boolean macd_allowed to UInt8 (0 or 1)
        data_for_update_df['macd_allowed'] = data_for_update_df['macd_allowed'].astype(int)

        # Get the order of columns as they exist in the target table for consistent insertion
        target_table_column_order = db_client.get_table_columns(TARGET_LABELS_TABLE)
        if not target_table_column_order:
            raise ValueError(f"Could not retrieve column order for target table {TARGET_LABELS_TABLE}.")

        # Ensure data_for_update_df has all columns required by the target table
        # and reorder them to match the target table's column order.
        missing_cols_in_df = [col for col in target_table_column_order if col not in data_for_update_df.columns]
        if missing_cols_in_df:
            # Attempt to be more specific if only case is an issue for some df columns
            df_cols_lower = {c.lower(): c for c in data_for_update_df.columns}
            # remapped_target_cols = [] # Not strictly needed for this logic block
            identified_missing = []
            for t_col in target_table_column_order: # Iterate through all target columns to ensure all are checked
                if t_col in data_for_update_df.columns:
                    # remapped_target_cols.append(t_col) # Column exists with correct case
                    pass
                elif t_col.lower() in df_cols_lower:
                    original_df_col_name = df_cols_lower[t_col.lower()]
                    print(f"Warning: Column case mismatch. Target table expects '{t_col}', DataFrame has '{original_df_col_name}'. Renaming for consistency.")
                    data_for_update_df.rename(columns={original_df_col_name: t_col}, inplace=True)
                    # remapped_target_cols.append(t_col)
                elif t_col not in data_for_update_df.columns: # If not found even after potential case rename
                    identified_missing.append(t_col)
            
            if identified_missing:
                 raise ValueError(f"DataFrame for update is missing columns required by target table '{TARGET_LABELS_TABLE}': {identified_missing}. DataFrame columns available: {data_for_update_df.columns.tolist()}")
        
        # Reorder DataFrame columns to match the target table schema exactly for insertion
        data_to_insert_df_ordered = data_for_update_df[target_table_column_order]
        
        # 3. Re-insert these rows into TARGET_LABELS_TABLE.
        # ReplacingMergeTree will handle updating based on primary key (ticker, timestamp).
        print(f"Re-inserting {len(data_to_insert_df_ordered)} rows into '{TARGET_LABELS_TABLE}' to update 'macd_allowed' column...")
        
        # Convert DataFrame to list of dicts for insertion
        records_to_insert = data_to_insert_df_ordered.to_dict('records')

        # Consider chunking if data_for_update_df can be very large
        INSERT_CHUNK_SIZE = 50000 # Can be adjusted
        num_records = len(records_to_insert)
        for i in range(0, num_records, INSERT_CHUNK_SIZE):
            chunk = records_to_insert[i : i + INSERT_CHUNK_SIZE]
            if chunk:
                print(f"Inserting chunk {i // INSERT_CHUNK_SIZE + 1}/{(num_records + INSERT_CHUNK_SIZE - 1) // INSERT_CHUNK_SIZE} ({len(chunk)} rows)...")
                db_client.insert(TARGET_LABELS_TABLE, chunk, column_names=target_table_column_order)

        print(f"Successfully updated 'macd_allowed' in '{TARGET_LABELS_TABLE}'.")

    except Exception as e:
        print(f"Error during MACD update to '{TARGET_LABELS_TABLE}': {e}")
        import traceback
        traceback.print_exc()
    return

# Original get_macd_predicate_df is removed as its functionality is merged into update_labels_with_macd_info

if __name__ == "__main__":
    import argparse
    import yaml
    import sys
    from datetime import datetime as dt # Alias to avoid conflict with module name

    parser = argparse.ArgumentParser(description="Update stock_historical_labels with MACD predicate information.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"), # Assumes config is in the same dir
        help="Path to the YAML configuration file."
    )
    parser.add_argument("--start-date", type=str, help="Start date for processing (YYYY-MM-DD). Overrides config.")
    parser.add_argument("--end-date", type=str, help="End date for processing (YYYY-MM-DD). Overrides config.")

    args = parser.parse_args()

    # Load main configuration
    config_main = {}
    try:
        print(f"Loading configuration from: {args.config}")
        with open(args.config, 'r') as f:
            config_main = yaml.safe_load(f)
        if config_main is None: 
            print(f"Warning: Config file '{args.config}' is empty or invalid.")
            config_main = {}
    except FileNotFoundError:
        print(f"Error: Config file not found at '{args.config}'")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file '{args.config}': {e}")
        sys.exit(1)

    # Determine start and end dates (command line overrides config)
    # The update_labels_with_macd_info function expects these to be passed if filtering by date is desired.
    start_date_str = args.start_date or config_main.get('start_date')
    end_date_str = args.end_date or config_main.get('end_date')

    # Validate date range logic if both are provided
    if start_date_str and end_date_str:
        try:
            dt_start = dt.strptime(start_date_str, '%Y-%m-%d')
            dt_end = dt.strptime(end_date_str, '%Y-%m-%d')
            if dt_start > dt_end:
                parser.error(f"Start date ({start_date_str}) cannot be after end date ({end_date_str}).")
        except ValueError:
            parser.error("Invalid date format. Please use YYYY-MM-DD.")
    elif (start_date_str and not end_date_str) or (not start_date_str and end_date_str):
        # If providing one, should provide both for a range, though the function can handle one or none.
        print("Warning: For date range filtering, both start and end dates are typically provided. Proceeding with provided dates.")

    db_client_standalone = None
    try:
        print("=== Standalone MACD Label Update Test ===")
        db_client_standalone = ClickHouseClient() # Assumes DB connection details are configured in ClickHouseClient
        
        # Manually ensure the predicate is treated as 'true' for this test run, 
        # or rely on the config value. For direct testing, forcing it might be useful.
        # Forcing use_macd_predicate to true for this test script if not set in the general config for testing purposes
        if 'label_generation' not in config_main:
            config_main['label_generation'] = {}
        if not config_main['label_generation'].get('use_macd_predicate', False):
            print("Warning: 'use_macd_predicate' is false in config. Forcing to true for this standalone test.")
            config_main['label_generation']['use_macd_predicate'] = True

        update_labels_with_macd_info(
            db_client=db_client_standalone,
            config=config_main, # Pass the full config
            start_date=start_date_str,
            end_date=end_date_str
        )
        print("Standalone MACD label update test finished.")

    except Exception as e_standalone:
        print(f"Error during standalone MACD update test: {e_standalone}")
        import traceback
        traceback.print_exc()
    finally:
        if db_client_standalone:
            db_client_standalone.close()
            print("Closed ClickHouse connection for standalone test.")
    print("=========================================") 