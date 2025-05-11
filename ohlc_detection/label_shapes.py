import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from typing import Dict, List

# --- Adjust sys.path --- #
# Add project root directory (parent of 'backtesting' and 'ohlc_detection') to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    print(f"Adding project root {PROJECT_ROOT} to sys.path")
    sys.path.insert(0, PROJECT_ROOT) # Add project root first

# Ensure the ohlc_detection directory itself is also findable if needed
OHLC_DIR_ABS = os.path.dirname(os.path.abspath(__file__))
if OHLC_DIR_ABS not in sys.path:
     sys.path.insert(1, OHLC_DIR_ABS) # Add after root

# Assuming db_utils is in the parent directory ('backtesting')
try:
    # Allow importing from the parent directory (backtesting)
    # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Covered by PROJECT_ROOT
    from backtesting.db_utils import ClickHouseClient # Be more explicit
except ImportError as e:
    print(f"Error importing ClickHouseClient from backtesting.db_utils: {e}")
    print("Ensure db_utils.py is in the 'backtesting' directory relative to project root.")
    sys.exit(1)


# --- Configuration ---
# Removed file paths

# --- Import Candlestick Module ---
# CANDLESTICK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'candlestick'))
# if CANDLESTICK_DIR not in sys.path:
#     print(f"Adding {CANDLESTICK_DIR} to sys.path")
#     sys.path.insert(0, CANDLESTICK_DIR) # Insert at beginning

try:
    # Try importing the package/module directly, assuming project root setup is sufficient
    from ohlc_detection.candlestick import candlestick as cs

    print("Successfully imported candlestick module.") # Updated message
except ImportError as e:
    print(f"Error importing candlestick module from ohlc_detection.candlestick: {e}")
    print("Please ensure the 'candlestick' directory/file exists within 'ohlc_detection' and is structured correctly.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during candlestick import: {e}")
    sys.exit(1)
# --- End Import ---

# --- Define Patterns to Detect ---
# (Keeping the same pattern list as before)
PATTERNS_TO_DETECT: Dict[str, callable] = {
    'bearish_engulfing': cs.bearish_engulfing,
    'bearish_harami': cs.bearish_harami,
    'bullish_engulfing': cs.bullish_engulfing,
    'bullish_harami': cs.bullish_harami,
    'dark_cloud_cover': cs.dark_cloud_cover,
    'doji': cs.doji,
    'doji_star': cs.doji_star,
    'dragonfly_doji': cs.dragonfly_doji,
    'evening_star': cs.evening_star,
    'evening_star_doji': cs.evening_star_doji,
    'gravestone_doji': cs.gravestone_doji,
    'hammer': cs.hammer,
    'hanging_man': cs.hanging_man,
    'inverted_hammer': cs.inverted_hammer,
    'morning_star': cs.morning_star,
    'morning_star_doji': cs.morning_star_doji,
    'piercing_pattern': cs.piercing_pattern,
    'rain_drop': cs.rain_drop,
    'rain_drop_doji': cs.rain_drop_doji,
    'shooting_star': cs.shooting_star,
    'star': cs.star,
    # Add new multi-candle patterns
    'double_top': cs.is_double_top,
    'double_bottom': cs.double_bottom,
    'triple_top': cs.is_triple_top,
    'triple_bottom': cs.is_triple_bottom,
    'breakout': cs.breakout,
}
print(f"Configured to detect {len(PATTERNS_TO_DETECT)} patterns.")

def load_and_prepare_data_from_db(db_client: ClickHouseClient, source_table: str, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    """Loads OHLCV data from ClickHouse, prepares columns, ensures sorted datetime index."""
    required_cols = ['timestamp', 'ticker', 'open', 'high', 'low', 'close']
    db_name = db_client.database

    print(f"Loading data from {db_name}.{source_table}...")

    where_clauses = []
    if start_date:
        start_dt_str = f"toDateTime('{start_date} 00:00:00', 'UTC')"
        where_clauses.append(f"timestamp >= {start_dt_str}")
    if end_date:
        end_dt_next_day_str = f"toDateTime('{end_date}', 'UTC') + INTERVAL 1 DAY"
        where_clauses.append(f"timestamp < {end_dt_next_day_str}")

    where_clause = ""
    if where_clauses:
        where_clause = "WHERE " + " AND ".join(where_clauses)
        print(f"Applying date filter: {where_clause}")

    query = f"""
    SELECT *
    FROM `{db_name}`.`{source_table}`
    {where_clause}
    ORDER BY ticker, timestamp
    """

    try:
        df = db_client.query_dataframe(query)

        if df is None or df.empty:
            print(f"No data returned from query on {source_table}. Check table and date range.")
            return pd.DataFrame(columns=required_cols)

        print(f"Loaded {len(df)} rows from database.")

        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)
        if df.empty:
             print("DataFrame empty after timestamp conversion/dropna.")
             return pd.DataFrame(columns=required_cols)

        # <<< Re-implement duplicate check based on (ticker, timestamp) combination >>>
        # Reset index temporarily to check columns
        df_reset = df.reset_index()
        initial_rows = len(df_reset)
        duplicates_exist = df_reset.duplicated(subset=['ticker', 'timestamp']).any()

        if duplicates_exist:
            print("Warning: Duplicate (ticker, timestamp) combinations found. Keeping first occurrence.")
            df_reset.drop_duplicates(subset=['ticker', 'timestamp'], keep='first', inplace=True)
            rows_dropped = initial_rows - len(df_reset)
            print(f"Dropped {rows_dropped} duplicate (ticker, timestamp) rows.")
            if df_reset.empty:
                print("DataFrame empty after dropping duplicates.")
                return pd.DataFrame(columns=required_cols)
            # Set timestamp back as index from the cleaned DataFrame
            df = df_reset.set_index('timestamp')
        else:
            # If no duplicates, df index is already set correctly
            print("No duplicate (ticker, timestamp) combinations found.")
            # df remains as it was after the initial set_index

        # Ensure index is sorted after potential modifications
        df.sort_index(inplace=True)
        # At this point, df.index should be unique across (ticker, timestamp) pairs
        # although the same timestamp might exist for different tickers.

        rename_map = {
            col: col.lower() for col in df.columns if col.lower() in ['open', 'high', 'low', 'close']
        }
        if 'ticker' not in df.columns:
             found_ticker = None
             for col in df.columns:
                 if col.lower() == 'ticker':
                     found_ticker = col
                     break
             if found_ticker and found_ticker != 'ticker':
                 rename_map[found_ticker] = 'ticker'
             else:
                 raise ValueError("Missing required 'ticker' column in source table.")
        df.rename(columns=rename_map, inplace=True)

        expected_final_cols = ['ticker', 'open', 'high', 'low', 'close']
        missing_cols = [col for col in expected_final_cols if col not in df.columns]
        if missing_cols:
             raise ValueError(f"Missing required columns after processing: {missing_cols}")

        for col in ['open', 'high', 'low', 'close']:
            if not pd.api.types.is_numeric_dtype(df[col]):
                 print(f"Warning: Column '{col}' is not numeric. Attempting conversion...")
                 df[col] = pd.to_numeric(df[col], errors='coerce')
                 if df[col].isnull().any():
                      raise ValueError(f"Conversion failed for column '{col}', contains non-numeric data or NaNs.")

        print("Data loaded and prepared successfully.")
        return df

    except ValueError as ve:
         print(f"Data validation error: {ve}")
         return pd.DataFrame(columns=required_cols)
    except Exception as e:
        print(f"An error occurred loading data from DB: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=required_cols)

def generate_labels_from_db(db_client: ClickHouseClient, config: Dict, pattern_directions: Dict[str, str]) -> pd.DataFrame:
    """ Loads data, detects patterns, assigns labels, and returns the enriched DataFrame."""
    label_config = config.get('label_generation', {})
    source_table = label_config.get('source_table')
    start_date = config.get('start_date')
    end_date = config.get('end_date')

    if not source_table:
        raise ValueError("Configuration missing 'label_generation.source_table'")

    # 1. Load and Prepare Data from DB
    # Returns df with unique (timestamp, ticker) index and OHLCV columns
    ohlc_df = load_and_prepare_data_from_db(db_client, source_table, start_date, end_date)
    if ohlc_df.empty:
        print("No OHLC data loaded, returning empty DataFrame.")
        return pd.DataFrame(columns=['ticker', 'pattern_label'])

    # --- Filter patterns to detect based on pattern_directions ---
    active_patterns_to_detect = {
        name: func for name, func in PATTERNS_TO_DETECT.items()
        if pattern_directions.get(name, '').lower() != 'ignore'
    }
    print(f"Original patterns: {len(PATTERNS_TO_DETECT)}, Active patterns after 'ignore' filter: {len(active_patterns_to_detect)}")
    if not active_patterns_to_detect:
        print("No active patterns to detect after filtering 'ignore' directives. Returning OHLC data with 'no_pattern' labels.")
        ohlc_df['pattern_label'] = 'no_pattern'
        # Ensure 'ticker' column exists if we are to return here.
        # load_and_prepare_data_from_db ensures 'ticker' is in ohlc_df if data is loaded.
        # If ohlc_df was empty and returned early, this path isn't hit.
        # If ohlc_df is not empty, it should have 'ticker'.
        return ohlc_df[['ticker', 'pattern_label']].reset_index() # return timestamp, ticker, pattern_label

    # 2. Detect All Patterns
    # Returns df with cols: timestamp, ticker, pattern_bool_cols...
    # Index is RangeIndex here
    pattern_results_with_ticker = detect_patterns_on_df(ohlc_df, active_patterns_to_detect)
    if pattern_results_with_ticker.empty:
         print("No pattern results generated, returning empty DataFrame.")
         return pd.DataFrame(columns=['ticker', 'pattern_label'])

    # 3. Assign Single Label per Row using idxmax
    # Returns Series with labels, indexed by RangeIndex
    label_series = assign_single_label(pattern_results_with_ticker)
    if label_series.empty:
         print("No labels assigned, returning empty DataFrame.")
         ohlc_df['pattern_label'] = 'no_pattern'
    # Assign the label series to the pattern results dataframe
    pattern_results_with_ticker['pattern_label'] = label_series

    # 4. Merge labels back into the original ohlc_df based on timestamp and ticker
    # Keep the original ohlc_df index (timestamp)
    ohlc_with_labels = pd.merge(
        ohlc_df.reset_index(),
        pattern_results_with_ticker[['timestamp', 'ticker', 'pattern_label']],
        on=['timestamp', 'ticker'],
        how='left'
    )

    # Fill any rows that didn't get a match with 'no_pattern'
    ohlc_with_labels['pattern_label'] = ohlc_with_labels['pattern_label'].fillna('no_pattern')

    # Set the timestamp back as the index
    ohlc_with_labels.set_index('timestamp', inplace=True)

    print(f"Generated labels for {len(ohlc_with_labels)} rows.")
    # Return the original DataFrame enriched with the pattern_label column
    return ohlc_with_labels

def detect_patterns_on_df(ohlc_df: pd.DataFrame, patterns_to_run: Dict[str, callable]) -> pd.DataFrame:
    """
    Applies specified candlestick pattern detection functions to the OHLC DataFrame,
    grouped by ticker.

    Args:
        ohlc_df: DataFrame with OHLC data, indexed by timestamp, and a 'ticker' column.
        patterns_to_run: Dictionary of pattern names to their detection functions.

    Returns:
        DataFrame with timestamp, ticker, and boolean columns for each detected pattern,
        indexed by the original timestamp.
    """
    if ohlc_df.empty:
        # print("Input OHLC DataFrame is empty. Cannot detect patterns.") # Commented out
        return pd.DataFrame()

    all_results = []
    total_tickers = ohlc_df['ticker'].nunique()
    # print(f"Detecting patterns for {total_tickers} unique tickers...") # Commented out

    required_cols = ['ticker', 'open', 'high', 'low', 'close']
    ohlc_df_processed = ohlc_df.copy()
    ohlc_df_processed.columns = map(str.lower, ohlc_df_processed.columns)
    if not all(col in ohlc_df_processed.columns for col in required_cols):
        missing = [col for col in required_cols if col not in ohlc_df_processed.columns]
        raise ValueError(f"Missing required columns in DataFrame: {missing}")

    if not isinstance(ohlc_df_processed.index, pd.DatetimeIndex):
        # print("Warning: DataFrame index is not a DatetimeIndex. Attempting to set 'timestamp' column as index.") # Commented out
        if 'timestamp' in ohlc_df_processed.columns:
            ohlc_df_processed['timestamp'] = pd.to_datetime(ohlc_df_processed['timestamp'], utc=True)
            ohlc_df_processed.set_index('timestamp', inplace=True)
        else:
            raise ValueError("Cannot set timestamp index: 'timestamp' column not found.")

    grouped = ohlc_df_processed.groupby('ticker')
    # print("Debug: About to start iterating through grouped tickers...") # Commented out

    for ticker, group_df in tqdm(grouped, total=total_tickers, desc="Processing Tickers"): 
        # print(f"\nStarting processing for ticker: {ticker}, Number of rows: {len(group_df)}") # Commented out
        if len(group_df) < 3: 
            continue

        ticker_results = group_df[[]].copy() 
        ticker_results['ticker'] = ticker 

        ohlc_list_for_mcp_prepared = False
        local_ohlc_list_for_mcp = [] 

        for pattern_name, pattern_func in patterns_to_run.items():
            # print(f"  Applying pattern '{pattern_name}' for ticker '{ticker}'...") # Commented out
            try:
                is_old_style_mcp = pattern_name in ['double_top', 'triple_top', 'triple_bottom']

                if is_old_style_mcp:
                    if not ohlc_list_for_mcp_prepared:
                        local_ohlc_list_for_mcp = []
                        for ts_idx, row_data in group_df.iterrows():
                            ohlc_item = {'timestamp': ts_idx}
                            for col in ['open', 'high', 'low', 'close']:
                                if col in row_data:
                                    ohlc_item[col] = row_data[col]
                                else: 
                                    ohlc_item[col] = np.nan 
                            local_ohlc_list_for_mcp.append(ohlc_item)
                        ohlc_list_for_mcp_prepared = True

                    if not local_ohlc_list_for_mcp:
                        ticker_results[pattern_name] = pd.Series(False, index=group_df.index)
                        continue

                    found, details = pattern_func(local_ohlc_list_for_mcp, config=None) # Corrected variable name
                    pattern_series = pd.Series(False, index=group_df.index) 
                    if found and details and 'breakout' in details and 'timestamp' in details['breakout'] and details['breakout']['timestamp'] is not None:
                        breakout_timestamp = pd.Timestamp(details['breakout']['timestamp'])
                        if breakout_timestamp.tzinfo is None and group_df.index.tz is not None: 
                           breakout_timestamp = breakout_timestamp.tz_localize(group_df.index.tz)
                        elif breakout_timestamp.tzinfo is not None and group_df.index.tz is not None and breakout_timestamp.tzinfo != group_df.index.tzinfo:
                           breakout_timestamp = breakout_timestamp.tz_convert(group_df.index.tz)

                        if breakout_timestamp in pattern_series.index:
                            pattern_series.loc[breakout_timestamp] = True
                    ticker_results[pattern_name] = pattern_series
                else: 
                    df_for_pattern = group_df.rename(columns={
                        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'
                    })
                    df_for_pattern.index = df_for_pattern.index.tz_localize(None) 

                    pattern_output = pattern_func(df_for_pattern)
                                                                
                    class_name_from_pattern = ''.join(word.capitalize() for word in pattern_name.split('_'))
                    bool_series_from_pattern = None

                    if isinstance(pattern_output, pd.DataFrame) and class_name_from_pattern in pattern_output.columns:
                        bool_series_from_pattern = pattern_output[class_name_from_pattern]
                    elif isinstance(pattern_output, pd.Series):
                        bool_series_from_pattern = pattern_output
                    else:
                        # print(f"\n  Warning: Pattern '{pattern_name}' for ticker '{ticker}' returned unexpected type: {type(pattern_output)}. Setting to False.") # Commented out
                        bool_series_from_pattern = pd.Series(False, index=df_for_pattern.index) 

                    # # --- DEBUGGING bool_series_from_pattern DIRECTLY AFTER CREATION (WORKING VERSION) ---
                    # if pattern_name == 'breakout' and ticker == 'META':
                    #     print(f"  DEBUG LABEL_SHAPES (META, breakout) - Initial bool_series_from_pattern:")
                    #     print(f"    Index type: {type(bool_series_from_pattern.index)}")
                    #     if isinstance(bool_series_from_pattern.index, pd.DatetimeIndex):
                    #         print(f"    Index tz: {bool_series_from_pattern.index.tzinfo}")
                    #     print(f"    Length of series: {len(bool_series_from_pattern)}")
                    #     print(f"    Number of True values: {bool_series_from_pattern.sum()}")
                    #     target_naive_ts_for_check = pd.Timestamp('2023-09-11 17:17:00') 
                    #     if isinstance(bool_series_from_pattern.index, pd.DatetimeIndex):
                    #         if target_naive_ts_for_check in bool_series_from_pattern.index:
                    #             val_at_target = bool_series_from_pattern.loc[target_naive_ts_for_check]
                    #             print(f"    Value at {target_naive_ts_for_check} (if DatetimeIndex): {val_at_target}", flush=True)
                    #         else:
                    #             print(f"    Target naive ts {target_naive_ts_for_check} NOT IN current DatetimeIndex.", flush=True)
                    #     elif isinstance(bool_series_from_pattern.index, pd.RangeIndex) and len(bool_series_from_pattern) == len(df_for_pattern):
                    #         try:
                    #             iloc_in_df_for_pattern = df_for_pattern.index.get_loc(target_naive_ts_for_check)
                    #             if iloc_in_df_for_pattern < len(bool_series_from_pattern):
                    #                 val_at_iloc = bool_series_from_pattern.iloc[iloc_in_df_for_pattern]
                    #                 print(f"    Value at iloc {iloc_in_df_for_pattern} (corresponds to {target_naive_ts_for_check} in df_for_pattern): {val_at_iloc}", flush=True)
                    #             else:
                    #                 print(f"    Iloc {iloc_in_df_for_pattern} out of bounds for bool_series_from_pattern (len {len(bool_series_from_pattern)}).", flush=True)
                    #         except KeyError:
                    #             print(f"    Target naive ts {target_naive_ts_for_check} not found in df_for_pattern.index to get iloc.", flush=True)
                    # # --- END DEBUGGING ---

                    if not isinstance(bool_series_from_pattern.index, pd.DatetimeIndex):
                        if len(bool_series_from_pattern) == len(df_for_pattern):
                            bool_series_from_pattern.index = df_for_pattern.index 
                        else:
                            # print(f"\n  ERROR: Length mismatch for {pattern_name} on {ticker}. Cannot align RangeIndex result with df_for_pattern. Setting to False.") # Commented out
                            bool_series_from_pattern = pd.Series(False, index=df_for_pattern.index)
                    
                    if bool_series_from_pattern.index.tzinfo is None: 
                        bool_series_from_pattern = bool_series_from_pattern.tz_localize('UTC', ambiguous='infer')
                    elif bool_series_from_pattern.index.tzinfo != ticker_results.index.tzinfo: 
                        bool_series_from_pattern = bool_series_from_pattern.tz_convert(ticker_results.index.tzinfo)

                    bool_series_from_pattern = bool_series_from_pattern.astype(bool)
                    
                    final_bool_series = bool_series_from_pattern.reindex(ticker_results.index, fill_value=False)
                    ticker_results[pattern_name] = final_bool_series.astype(bool) 

            except Exception as e:
                print(f"\n  Error applying pattern '{pattern_name}' to ticker '{ticker}': {e}")
                import traceback
                traceback.print_exc()
                ticker_results[pattern_name] = pd.Series(False, index=ticker_results.index)
            # print(f"  Finished pattern '{pattern_name}' for ticker '{ticker}'.") # Commented out

        all_results.append(ticker_results.reset_index())
    # print(f"Finished processing for ticker: {ticker}") # Commented out

    if not all_results:
        # print("No pattern results generated for any ticker.") # Commented out
        return pd.DataFrame()

    final_results_df = pd.concat(all_results, ignore_index=True)
    # print("Pattern detection finished.") # Commented out
    return final_results_df

def assign_single_label(pattern_results_df: pd.DataFrame) -> pd.Series:
    """Assigns a single label per row based on detected patterns (alphabetical priority).
       Input DF expected index: RangeIndex. Columns: timestamp, ticker, pattern_bool_cols...
       Returns pandas Series containing the assigned label, indexed by timestamp."""

    if pattern_results_df.empty:
        print("Pattern results DataFrame is empty, cannot assign labels.")
        return pd.Series(dtype=str)

    print("Assigning single label per row (using idxmax for first True)...")

    # Identify pattern boolean columns (exclude timestamp, ticker)
    pattern_bool_columns = sorted([col for col in pattern_results_df.columns if col not in ['timestamp', 'ticker']])

    # Ensure boolean columns exist before proceeding
    if not pattern_bool_columns: # No specific pattern columns mean no patterns were run/detected.
        print("Warning: No pattern boolean columns found in results. Assigning 'no_pattern' to all.")
        # If pattern_results_df is not empty but has no pattern columns (e.g., only timestamp, ticker),
        # we need to create an index for the 'no_pattern' series that matches.
        return pd.Series('no_pattern', index=pattern_results_df.index)

    # Select only the boolean columns for idxmax
    print(f"DEBUG assign_single_label: Columns for idxmax: {pattern_bool_columns}") # Debug print
    bool_df = pattern_results_df[pattern_bool_columns]

    has_pattern_mask = bool_df.any(axis=1)

    # Initialize label series
    # Use the same RangeIndex as pattern_results_df
    final_labels = pd.Series('no_pattern', index=pattern_results_df.index)

    # Apply idxmax only to rows that have at least one pattern
    if has_pattern_mask.any():
        first_true_label = bool_df.loc[has_pattern_mask].idxmax(axis=1)
        final_labels.loc[has_pattern_mask] = first_true_label

    print("Label assignment complete.")
    print("Final label distribution:")
    print(final_labels.value_counts())

    # Return only the Series of labels, indexed by RangeIndex
    return final_labels

if __name__ == "__main__":
    print("This script is intended to be imported and called via generate_labels_from_db.")
    print("To test standalone, you would need to:")
    print("1. Set up DB connection details (e.g., environment variables for ClickHouseClient)")
    print("2. Create a sample config dictionary")
    print("3. Instantiate ClickHouseClient")
    print("4. Call generate_labels_from_db(client, config)")
    pass 