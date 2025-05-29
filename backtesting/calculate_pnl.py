import os
import sys
import datetime as dt_module
from datetime import timedelta, datetime
import pandas as pd
import argparse
import logging
from typing import Dict, Optional
import yaml # Added missing import for standalone execution

# Assuming db_utils is in the same directory
from db_utils import ClickHouseClient # Reverted to direct import

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Database table names
LABELS_TABLE = "stock_historical_labels"
QUOTES_TABLE = "stock_quotes"
PNL_TABLE = "stock_pnl"
PREDICTIONS_TABLE = "stock_historical_predictions"

# Share size for P&L calculation
SHARE_SIZE = 100

# Time offsets for entry and exit relative to label timestamp
# Entry: 1 minute + 12 seconds after label timestamp
# Exit: 15 minutes after entry time (configurable implicitly via offset sum)
ENTRY_OFFSET_SECONDS = 72  # 1 * 60 + 12
# EXIT_OFFSET_SECONDS = ENTRY_OFFSET_SECONDS + (15 * 60) # <<< This will be replaced by a local variable
# ---------------------

# Define the schema for the PnL table
# Using DateTime64(9, 'UTC') for timestamps to match labels and quotes
PNL_SCHEMA = {
    'label_timestamp': "DateTime64(9, 'UTC')", # Timestamp of the label event
    'ticker': 'String',
    'pattern_label': 'String',   # The detected pattern label
    'pos_long': 'UInt8',       # 1 if long trade based on pattern direction
    'pos_short': 'UInt8',      # 1 if short trade based on pattern direction
    'entry_timestamp': "DateTime64(9, 'UTC')",  # Timestamp for entry quote
    'exit_timestamp': "DateTime64(9, 'UTC')",   # Timestamp for exit quote
    'entry_bid_price': 'Float64', # Bid price at entry time
    'entry_ask_price': 'Float64', # Ask price at entry time
    'exit_bid_price': 'Float64',  # Bid price at exit time
    'exit_ask_price': 'Float64',  # Ask price at exit time
    'price_diff_per_share': 'Float64', # P&L per share based on long/short logic
    'share_size': 'UInt32',
    'pnl_total': 'Float64'      # Total P&L for the trade
}

# Define table engine and sorting key for PnL table
PNL_ENGINE = "ENGINE = MergeTree()"
# Sort by ticker, then the original label timestamp
PNL_ORDER_BY = "ORDER BY (ticker, label_timestamp)"

# Batching configuration for inner loop (labels per ticker)
LABEL_CHUNK_SIZE = 10000

# Buffer added to the min/max quote time range for the quote filter
QUOTE_TIME_BUFFER = timedelta(hours=1, minutes=10)

def run_pnl_calculation(
    pattern_directions: Dict[str, str], 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    model_predicate_threshold: Optional[float] = None,
    hold_time_seconds: Optional[int] = None,
    config: Optional[Dict] = None
):
    """Connects to DB, calculates P&L based on pattern labels and directions, and stores it."""
    db_client = None
    try:
        # --- Determine effective hold time and calculate exit offset ---
        default_hold_seconds = 15 * 60 # 15 minutes
        effective_hold_time_seconds = default_hold_seconds
        if hold_time_seconds is not None:
            if isinstance(hold_time_seconds, int) and hold_time_seconds > 0:
                effective_hold_time_seconds = hold_time_seconds
                logger.info(f"Using configured hold time: {effective_hold_time_seconds} seconds.")
            else:
                logger.warning(f"Invalid hold_time_seconds value ('{hold_time_seconds}') received. Using default: {default_hold_seconds}s")
        else:
            logger.info(f"No hold_time_seconds provided. Using default: {default_hold_seconds}s")
        
        # Calculate exit_offset based on entry_offset and effective_hold_time
        calculated_exit_offset_seconds = ENTRY_OFFSET_SECONDS + effective_hold_time_seconds
        logger.info(f"Entry offset: {ENTRY_OFFSET_SECONDS}s, Exit offset: {calculated_exit_offset_seconds}s (Hold: {effective_hold_time_seconds}s)")

        # 1. Initialize Database Connection
        db_client = ClickHouseClient()
        db_name = db_client.database

        # --- Validate pattern_directions --- #
        if not pattern_directions or not isinstance(pattern_directions, dict):
            logger.error("Invalid or missing 'pattern_directions' dictionary. Cannot determine trades.")
            return
        # Create sets for quick lookup, converting labels to lowercase for robustness
        long_patterns = {label.lower() for label, direction in pattern_directions.items() if direction.lower() == 'long'}
        short_patterns = {label.lower() for label, direction in pattern_directions.items() if direction.lower() == 'short'}
        ignored_patterns = {label.lower() for label, direction in pattern_directions.items() if direction.lower() == 'ignore'}
        
        logger.info(f"Pattern configuration summary:")
        logger.info(f"  LONG patterns ({len(long_patterns)}): {sorted(long_patterns) if long_patterns else 'None'}")
        logger.info(f"  SHORT patterns ({len(short_patterns)}): {sorted(short_patterns) if short_patterns else 'None'}")
        logger.info(f"  IGNORED patterns ({len(ignored_patterns)}): {sorted(ignored_patterns) if ignored_patterns else 'None'}")
        
        if not long_patterns and not short_patterns:
            logger.warning("No patterns configured for LONG or SHORT trades in pattern_directions.")

        # Check if source tables exist
        if not db_client.table_exists(LABELS_TABLE):
            logger.error(f"Labels table '{LABELS_TABLE}' not found. Please run generate_pattern_labels.py first.")
            return
        if not db_client.table_exists(QUOTES_TABLE):
            logger.error(f"Quotes table '{QUOTES_TABLE}' not found. Ensure quote data is available.")
            return

        # --- Handle Model Predicate Threshold ---
        effective_model_predicate_threshold = None
        if model_predicate_threshold is not None:
            if not db_client.table_exists(PREDICTIONS_TABLE):
                logger.warning(f"Model predicate threshold is {model_predicate_threshold}, but predictions table '{PREDICTIONS_TABLE}' not found. Predicate will be IGNORED.")
            else:
                logger.info(f"Model predicate ENABLED: prediction_raw >= {model_predicate_threshold} from table '{PREDICTIONS_TABLE}'")
                effective_model_predicate_threshold = model_predicate_threshold
        else:
            logger.info("Model predicate DISABLED: No threshold provided.")

        # 2. Get list of distinct tickers to process from the LABELS table
        logger.info(f"Fetching distinct tickers from {LABELS_TABLE}...")
        # Filter tickers based on labels that actually trigger trades
        tradable_labels = list(long_patterns.union(short_patterns))
        if not tradable_labels:
            logger.warning("No patterns configured for trading. No tickers to process.")
            return
        # Format labels for SQL IN clause
        tradable_labels_sql = ", ".join([f"'{label}'" for label in tradable_labels])
        ticker_query = f"SELECT DISTINCT ticker FROM `{db_name}`.`{LABELS_TABLE}` WHERE lower(pattern_label) IN ({tradable_labels_sql}) ORDER BY ticker"

        ticker_df = db_client.query_dataframe(ticker_query)
        if ticker_df is None or ticker_df.empty:
            logger.warning(f"No tickers found with tradable patterns in {LABELS_TABLE}. Nothing to process.")
            return
        tickers_to_process = ticker_df['ticker'].tolist()
        logger.info(f"Found {len(tickers_to_process)} tickers with tradable patterns to process.")

        # 3. Drop and Recreate the PnL Table
        logger.info(f"Dropping existing PnL table '{PNL_TABLE}' (if it exists)...")
        db_client.drop_table_if_exists(PNL_TABLE)
        logger.info(f"Creating empty PnL table '{PNL_TABLE}' with new schema...")
        create_table_query = f"""
        CREATE TABLE `{db_name}`.`{PNL_TABLE}` (
            {', '.join([f'`{col}` {dtype}' for col, dtype in PNL_SCHEMA.items()])}
        )
        {PNL_ENGINE}
        {PNL_ORDER_BY}
        """
        create_result = db_client.execute(create_table_query)
        if not create_result and not db_client.table_exists(PNL_TABLE):
            logger.error(f"Failed to create table '{PNL_TABLE}'. Aborting.")
            return
        logger.info(f"Successfully created or confirmed table '{PNL_TABLE}'.")

        # 4. Process P&L Calculation and Insertion in Batches
        total_rows_inserted_overall = 0
        start_time_all = dt_module.datetime.now()

        # Build WHERE clause for date filtering on label timestamp
        label_where_clauses = []
        if start_date:
            label_where_clauses.append(f"timestamp >= toDateTime('{start_date} 00:00:00', 'UTC')")
        if end_date:
            label_where_clauses.append(f"timestamp < toDateTime('{end_date}', 'UTC') + INTERVAL 1 DAY")

        # Also filter for only tradable patterns here to reduce count/query load
        label_where_clauses.append(f"lower(pattern_label) IN ({tradable_labels_sql})")

        label_date_filter = " AND ".join(label_where_clauses)
        logger.info(f"Applying label filter: {label_date_filter}")

        for i, ticker in enumerate(tickers_to_process):
            logger.info(f"\n--- Processing ticker {i+1}/{len(tickers_to_process)}: {ticker} ---")
            start_time_ticker = dt_module.datetime.now()

            # Get total labels count for this ticker for chunking (WITH DATE/PATTERN FILTER)
            count_filter = f"ticker = {{ticker:String}} AND {label_date_filter}"
            count_query = f"SELECT count() FROM `{db_name}`.`{LABELS_TABLE}` WHERE {count_filter}"
            count_result = db_client.execute(count_query, params={'ticker': ticker})

            if not count_result or not count_result.result_rows:
                logger.warning(f"Could not get label count for ticker {ticker}. Skipping...")
                continue
            total_labels_for_ticker = count_result.result_rows[0][0]
            if total_labels_for_ticker == 0:
                logger.warning(f"No tradable labels found for ticker {ticker} within date range. Skipping...")
                continue
            logger.info(f"Found {total_labels_for_ticker} tradable labels for {ticker}. Processing in chunks of {LABEL_CHUNK_SIZE}...")

            # Inner loop for label chunks within the current ticker
            offset = 0
            while offset < total_labels_for_ticker:
                logger.info(f"  Processing chunk: offset {offset}, limit {LABEL_CHUNK_SIZE}")
                start_time_chunk = dt_module.datetime.now()

                # --- Step 1: Fetch target times for the current chunk --- 
                # Add date/pattern filter to the subquery here as well
                target_times_subquery_filter = f"ticker = {{ticker:String}} AND {label_date_filter}"

                target_times_query = f"""
                SELECT
                    min(timestamp + INTERVAL {ENTRY_OFFSET_SECONDS} SECOND) AS min_entry_time,
                    max(timestamp + INTERVAL {calculated_exit_offset_seconds} SECOND) AS max_exit_time
                FROM (
                    SELECT timestamp
                    FROM `{db_name}`.`{LABELS_TABLE}`
                    WHERE {target_times_subquery_filter} -- Apply ticker, date, and pattern filter here
                    ORDER BY timestamp
                    LIMIT {LABEL_CHUNK_SIZE} OFFSET {{offset:UInt64}}
                )
                """
                params_chunk_range = {'ticker': ticker, 'offset': offset}
                target_times_df = db_client.query_dataframe(target_times_query, params=params_chunk_range)

                if target_times_df is None or target_times_df.empty or target_times_df.iloc[0]['min_entry_time'] is None:
                    logger.warning(f"  Warning: Could not fetch valid time range for chunk (offset {offset}). Skipping chunk.")
                    offset += LABEL_CHUNK_SIZE
                    continue

                min_target_time = pd.to_datetime(target_times_df.iloc[0]['min_entry_time'], utc=True)
                max_target_time = pd.to_datetime(target_times_df.iloc[0]['max_exit_time'], utc=True)

                # --- Step 2: Determine time range for filtering quotes --- 
                quote_range_start = min_target_time - QUOTE_TIME_BUFFER
                quote_range_end = max_target_time + QUOTE_TIME_BUFFER
                logger.info(f"    Quote Time Range Filter: {quote_range_start} to {quote_range_end}")

                # --- Step 3: Construct and Execute INSERT query using ASOF JOIN --- #
                quote_start_str = quote_range_start.strftime('%Y-%m-%d %H:%M:%S.%f')
                quote_end_str = quote_range_end.strftime('%Y-%m-%d %H:%M:%S.%f')

                # Add date/pattern filter to the labels_chunk CTE definition
                labels_chunk_filter = f"ticker = {{ticker:String}} AND {label_date_filter}"

                # Construct CASE statements for pos_long/pos_short based on pattern_directions
                long_case_conditions = " OR ".join([f"lower(pattern_label) = '{p}'" for p in long_patterns]) if long_patterns else "0"
                short_case_conditions = " OR ".join([f"lower(pattern_label) = '{p}'" for p in short_patterns]) if short_patterns else "0"

                # --- Build model predicate SQL filter ---
                model_predicate_filter_sql = ""
                if effective_model_predicate_threshold is not None:
                    model_predicate_filter_sql = f"AND shp.timestamp IS NOT NULL AND shp.prediction_raw >= {effective_model_predicate_threshold}"

                # --- Build MACD predicate SQL filter (directly on labels_chunk.macd_allowed) ---
                macd_predicate_condition_sql = ""
                if config and config.get('label_generation', {}).get('use_macd_predicate', False):
                    macd_predicate_condition_sql = f"AND lc.macd_allowed = 1"
                else:
                    logger.info("MACD predicate is disabled in config or config not provided to P&L; MACD filter will not be applied in P&L query.")

                insert_pnl_query = f"""
                INSERT INTO `{db_name}`.`{PNL_TABLE}`
                WITH labels_chunk AS (
                    -- Select labels for the current chunk, calculate target entry/exit times
                    SELECT
                        timestamp AS label_timestamp,
                        ticker,
                        pattern_label,
                        macd_allowed, # <<< Select existing macd_allowed from labels_chunk (stock_historical_labels)
                        label_timestamp + INTERVAL {ENTRY_OFFSET_SECONDS} SECOND AS target_entry_time,
                        label_timestamp + INTERVAL {calculated_exit_offset_seconds} SECOND AS target_exit_time,
                        -- Determine position based on pattern label direction config
                        if({long_case_conditions}, 1, 0) AS pos_long_calc,
                        if({short_case_conditions}, 1, 0) AS pos_short_calc
                    FROM `{db_name}`.`{LABELS_TABLE}`
                    WHERE {labels_chunk_filter} -- Apply ticker, date, and pattern filter here
                    ORDER BY label_timestamp
                    LIMIT {LABEL_CHUNK_SIZE} OFFSET {{offset:UInt64}}
                ),
                quotes_filtered AS (
                    -- Pre-filter quotes AND ensure correct ORDER BY for ASOF JOIN
                    SELECT sip_timestamp, ticker, bid_price, ask_price
                    FROM `{db_name}`.`{QUOTES_TABLE}`
                    WHERE ticker = {{ticker:String}}
                      AND sip_timestamp >= '{quote_start_str}'
                      AND sip_timestamp <= '{quote_end_str}'
                    ORDER BY ticker, sip_timestamp -- Crucial for ASOF JOIN
                )
                SELECT
                    lc.label_timestamp,
                    lc.ticker,
                    lc.pattern_label,
                    lc.pos_long_calc AS pos_long,
                    lc.pos_short_calc AS pos_short,

                    q_entry.sip_timestamp AS entry_timestamp,
                    q_exit.sip_timestamp AS exit_timestamp,

                    q_entry.bid_price AS entry_bid_price,
                    q_entry.ask_price AS entry_ask_price,
                    q_exit.bid_price AS exit_bid_price,
                    q_exit.ask_price AS exit_ask_price,

                    -- P&L Calculation based on determined long/short position
                    multiIf(
                        pos_long = 1, exit_ask_price - entry_bid_price,  -- Long: Exit Ask - Entry Bid
                        pos_short = 1, entry_ask_price - exit_bid_price, -- Short: Entry Ask - Exit Bid
                        0
                    ) AS price_diff_per_share,
                    {SHARE_SIZE} AS share_size,
                    price_diff_per_share * share_size AS pnl_total

                FROM labels_chunk lc
                -- Join with predictions table to get prediction_raw
                LEFT JOIN `{db_name}`.`{PREDICTIONS_TABLE}` AS shp
                  ON lc.ticker = shp.ticker AND lc.label_timestamp = shp.timestamp
                -- Find last quote AT or BEFORE target_entry_time
                ASOF LEFT JOIN quotes_filtered q_entry ON lc.ticker = q_entry.ticker AND lc.target_entry_time >= q_entry.sip_timestamp
                -- Find last quote AT or BEFORE target_exit_time
                ASOF LEFT JOIN quotes_filtered q_exit ON lc.ticker = q_exit.ticker AND lc.target_exit_time >= q_exit.sip_timestamp

                WHERE -- Filter out rows where ASOF JOIN didn't find a match or prices are invalid
                    entry_timestamp IS NOT NULL AND exit_timestamp IS NOT NULL AND
                    entry_bid_price > 0 AND entry_ask_price > 0 AND
                    exit_bid_price > 0 AND exit_ask_price > 0 AND
                    (pos_long = 1 OR pos_short = 1) -- Ensure it was actually a trade
                    {model_predicate_filter_sql} -- <<< Apply model predicate filter
                    {macd_predicate_condition_sql} -- <<< Apply MACD predicate filter directly on lc.macd_allowed
                """

                # Add debug information about the INSERT query parameters and results
                logger.info(f"    Target time range: {min_target_time} to {max_target_time}")
                logger.info(f"    Label filter includes: {tradable_labels}")
                if effective_model_predicate_threshold is not None:
                    logger.info(f"    Model predicate threshold: >= {effective_model_predicate_threshold}")
                else:
                    logger.info(f"    Model predicate: DISABLED")

                # --- Optional Debug: Count expected rows before insert ---
                # NOTE: Debug query disabled due to memory issues with large table JOINs
                logger.info(f"    Debug stats: Skipping detailed pre-insert statistics due to memory constraints with large tables")
                
                # try:
                #     # Build the debug query with proper JOINs
                #     debug_model_join = ""
                #     debug_model_filter = ""
                #     if effective_model_predicate_threshold is not None:
                #         debug_model_join = f"""
                #         LEFT JOIN `{db_name}`.`{PREDICTIONS_TABLE}` shp 
                #             ON l.ticker = shp.ticker AND l.timestamp = shp.timestamp"""
                #         debug_model_filter = f"AND shp.timestamp IS NOT NULL AND shp.prediction_raw >= {effective_model_predicate_threshold}"
                #     
                #     count_debug_query = f"""
                #     SELECT 
                #         COUNT() as total_rows,
                #         COUNT(DISTINCT l.pattern_label) as distinct_patterns,
                #         COUNT(CASE WHEN eh.bid_price IS NOT NULL THEN 1 END) as entry_quotes_found,
                #         COUNT(CASE WHEN ex.ask_price IS NOT NULL THEN 1 END) as exit_quotes_found
                #     FROM `{db_name}`.`{LABELS_TABLE}` l
                #     LEFT JOIN `{db_name}`.`{QUOTES_TABLE}` eh 
                #         ON eh.ticker = l.ticker 
                #         AND eh.sip_timestamp = l.timestamp + INTERVAL {ENTRY_OFFSET_SECONDS} SECOND
                #     LEFT JOIN `{db_name}`.`{QUOTES_TABLE}` ex 
                #         ON ex.ticker = l.ticker 
                #         AND ex.sip_timestamp = l.timestamp + INTERVAL {calculated_exit_offset_seconds} SECOND
                #     {debug_model_join}
                #     WHERE l.ticker = %(ticker)s
                #         AND l.timestamp + INTERVAL {ENTRY_OFFSET_SECONDS} SECOND >= %(quote_range_start)s
                #         AND l.timestamp + INTERVAL {calculated_exit_offset_seconds} SECOND <= %(quote_range_end)s
                #         AND lower(l.pattern_label) IN ({tradable_labels_sql})
                #         {debug_model_filter}
                #         {macd_predicate_condition_sql}
                #     """
                #     debug_params = {
                #         'ticker': ticker, 
                #         'quote_range_start': quote_range_start,
                #         'quote_range_end': quote_range_end
                #     }
                #     if effective_model_predicate_threshold is not None:
                #         debug_params['model_threshold'] = effective_model_predicate_threshold
                #     
                #     debug_result = db_client.query_dataframe(count_debug_query, params=debug_params)
                #     if debug_result is not None and not debug_result.empty:
                #         debug_row = debug_result.iloc[0]
                #         logger.info(f"    Pre-insert stats: {debug_row['total_rows']} total rows, "
                #                   f"{debug_row['distinct_patterns']} patterns, "
                #                   f"{debug_row['entry_quotes_found']}/{debug_row['total_rows']} entry quotes found "
                #                   f"({100*debug_row['entry_quotes_found']/max(debug_row['total_rows'],1):.1f}%), "
                #                   f"{debug_row['exit_quotes_found']}/{debug_row['total_rows']} exit quotes found "
                #                   f"({100*debug_row['exit_quotes_found']/max(debug_row['total_rows'],1):.1f}%)")
                # except Exception as debug_e:
                #     logger.warning(f"    Debug query failed: {debug_e}")

                # Execute the query
                params = {'ticker': ticker, 'offset': offset}
                logger.info(f"  Executing calculation and insertion for {ticker} chunk (offset {offset})...")
                insert_result = db_client.execute(insert_pnl_query, params=params)
                end_time_chunk = dt_module.datetime.now()

                if insert_result:
                    logger.info(f"  Successfully processed chunk for {ticker} (offset {offset}). Time: {end_time_chunk - start_time_chunk}")
                else:
                    logger.warning(f"  Failed to process chunk for {ticker} (offset {offset}). Check query and logs. Skipping chunk...")

                offset += LABEL_CHUNK_SIZE

            end_time_ticker = dt_module.datetime.now()
            logger.info(f"--- Finished processing ticker {ticker}. Total Time: {end_time_ticker - start_time_ticker} --- ")

        end_time_all = dt_module.datetime.now()
        logger.info(f"\n--- Finished processing all tickers --- ")
        logger.info(f"Total batch processing time: {end_time_all - start_time_all}")

        # Final count
        count_result = db_client.execute(f"SELECT count() FROM `{db_name}`.`{PNL_TABLE}`")
        if count_result and count_result.result_rows:
            total_rows_inserted_overall = count_result.result_rows[0][0]
            logger.info(f"Total rows inserted into '{PNL_TABLE}': {total_rows_inserted_overall}")
        else:
            logger.warning(f"Could not retrieve final row count for '{PNL_TABLE}'.")

    except FileNotFoundError as e: # Should not happen if tables exist
        logger.error(f"Error: {e}.")
    except ValueError as e:
        logger.error(f"Configuration or data error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during P&L calculation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if db_client:
            db_client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate P&L based on historical pattern labels and quotes.")
    parser.add_argument("--start-date", type=str, help="Start date for P&L calculation range (YYYY-MM-DD). Filters labels used.")
    parser.add_argument("--end-date", type=str, help="End date for P&L calculation range (YYYY-MM-DD). Filters labels used.")
    # Argument for pattern directions - needed for standalone run
    parser.add_argument("--pattern-directions-yaml", type=str, default="config.yaml", help="Path to YAML config file containing label_generation.pattern_directions.")

    args = parser.parse_args()

    # --- Load pattern directions from config for standalone run --- #
    pattern_directions_map = {}
    try:
        # import yaml # No longer needed here, moved to top
        logger.info(f"Loading pattern directions from: {args.pattern_directions_yaml}")
        with open(args.pattern_directions_yaml, 'r') as f:
            config_full = yaml.safe_load(f)
        if config_full and 'label_generation' in config_full and 'pattern_directions' in config_full['label_generation']:
            pattern_directions_map = config_full['label_generation']['pattern_directions']
        else:
            logger.warning(f"Warning: Could not find 'label_generation.pattern_directions' in {args.pattern_directions_yaml}")
    except FileNotFoundError:
        logger.error(f"Error: Config file '{args.pattern_directions_yaml}' not found.")
        sys.exit(1)
    except ImportError:
        logger.error("Error: PyYAML is required to load config for standalone run. Please install it.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file '{args.pattern_directions_yaml}': {e}")
        sys.exit(1)

    # Basic date validation
    if args.start_date and not args.end_date:
        parser.error("--start-date requires --end-date.")
    if args.end_date and not args.start_date:
        parser.error("--end-date requires --start-date.")
    if args.start_date and args.end_date:
        try:
            # Correctly use the imported datetime class for parsing
            datetime_obj_start = dt_module.datetime.strptime(args.start_date, '%Y-%m-%d')
            datetime_obj_end = dt_module.datetime.strptime(args.end_date, '%Y-%m-%d')
            
            # Reformat to ensure YYYY-MM-DD with leading zeros for consistency 
            # and for ClickHouse if these args were to be used directly in queries here.
            start_date_formatted = datetime_obj_start.strftime('%Y-%m-%d')
            end_date_formatted = datetime_obj_end.strftime('%Y-%m-%d')

            if datetime_obj_start > datetime_obj_end: # Compare datetime objects
                 parser.error("Start date cannot be after end date.")
            
            # Update args with formatted dates to be passed to run_pnl_calculation
            args.start_date = start_date_formatted
            args.end_date = end_date_formatted
        except ValueError:
            parser.error("Invalid date format. Please use YYYY-MM-DD.")

    logger.info("=== Starting P&L Calculation (Pattern Based) ===")
    run_pnl_calculation(
        pattern_directions=pattern_directions_map,
        start_date=args.start_date,
        end_date=args.end_date,
        model_predicate_threshold=None,
        hold_time_seconds=None,
        config=config_full if 'config_full' in locals() else {}
    )
    logger.info("===============================================")
