import os
import pandas as pd
import csv
from datetime import datetime, time
from zoneinfo import ZoneInfo
import argparse
from collections import defaultdict
from typing import Set, Tuple, Dict # Added Dict

# Assuming db_utils is in the same directory
from db_utils import ClickHouseClient

# --- Configuration ---
# Database table name to read PNL results from
PNL_TABLE = "stock_pnl"

# Output CSV file path
OUTPUT_DIR = r"C:\Users\spenc\Downloads\Dev Files\datahead_v3\backtesting\files" # Escaped backslashes for Windows path
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV header (remains the same)
CSV_HEADER = ["Date", "Time", "Symbol", "Quantity", "Price", "Side"]

# Chunk size (removed, fetching all at once now)
# DB_CHUNK_SIZE = 50000 
# ---------------------

def format_timestamp(ts: pd.Timestamp) -> tuple:
    """Formats pandas Timestamp into Date (MM/DD/YYYY) and Time (HH:MM:SS)."""
    if pd.isna(ts):
        return None, None
    if ts.tzinfo is None:
        ts_utc = ts.tz_localize('UTC')
    else:
        ts_utc = ts.tz_convert('UTC')
    date_str = ts_utc.strftime('%m/%d/%Y')
    time_str = ts_utc.strftime('%H:%M:%S')
    return date_str, time_str

# <<< Updated signature slightly (removed export_config as filters are handled upstream) >>>
def export_pnl_to_csv(
    size_multiplier: float = 1.0,
    use_time_filter: bool = False,
    start_date_str: str | None = None,
    end_date_str: str | None = None,
    use_random_sample: bool = False,
    sample_fraction: float = 0.1,
    max_concurrent_positions: int | None = None
) -> Set[Tuple[str, pd.Timestamp]]: # Return type remains the same (ticker, label_timestamp)
    """
    Fetches P&L data from ClickHouse (generated based on patterns),
    applies time filter, sampling, concurrency limits, sorts globally,
    and exports results to CSV.
    Returns a set of (ticker, label_timestamp) tuples for exported trades.
    """
    db_client = None
    base_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_path = os.path.join(OUTPUT_DIR, f"backtest_results_{base_timestamp_str}.csv")
    exported_trade_identifiers = set() # Initialize set

    try:
        # 1. Initialize Database Connection
        db_client = ClickHouseClient()
        db_name = db_client.database

        # 2. Check if PNL table exists
        if not db_client.table_exists(PNL_TABLE):
            print(f"Error: PnL table '{PNL_TABLE}' not found. Please run calculate_pnl.py first.")
            return exported_trade_identifiers # Return empty set

        # --- Filters --- 
        # SQL filters based on prediction scores/categories are removed.
        # Filtering based on pattern labels happens in calculate_pnl.py (via pattern_directions).
        # We only apply date filters here.
        active_sql_filters = []
        if start_date_str:
            # Filter based on the original label timestamp stored in the PNL table
            active_sql_filters.append(f"toDate(label_timestamp) >= toDate('{start_date_str}')")
            print(f"Date filter: Starting from {start_date_str}")
        if end_date_str:
            active_sql_filters.append(f"toDate(label_timestamp) <= toDate('{end_date_str}')")
            print(f"Date filter: Ending at {end_date_str}")

        combined_sql_filter_clause = " AND ".join(f"({f})" for f in active_sql_filters)
        where_clause = f"WHERE {combined_sql_filter_clause}" if combined_sql_filter_clause else ""
        # --- End Filters ---

        # Time filter info
        if use_time_filter:
            print("Time filter ENABLED (will be applied after fetching).")
        else:
            print("Time filter is DISABLED.")

        # --- Construct Base Query Parts --- 
        base_table_expression = f"`{db_name}`.`{PNL_TABLE}`"

        # --- Handle Sampling (Post-fetch) --- 
        if use_random_sample:
            print(f"Random sampling ENABLED: {sample_fraction * 100:.1f}% (Applied after fetch)")
            if not (0 < sample_fraction <= 1):
                print("Error: Sample fraction must be between 0 (exclusive) and 1 (inclusive). Aborting.")
                return exported_trade_identifiers
        else:
            print("Random sampling DISABLED.")

        # --- Handle Max Concurrent Positions --- 
        if max_concurrent_positions is not None:
            if max_concurrent_positions <= 0:
                print("Error: Max concurrent positions must be a positive integer. Aborting.")
                return exported_trade_identifiers
            print(f"Max concurrent positions per symbol ENABLED: {max_concurrent_positions}")
        else:
            print("Max concurrent positions per symbol DISABLED.")

        # --- Final Query Construction (Fetch data based ONLY on date filters) --- 
        count_query = f"SELECT count() FROM {base_table_expression} {where_clause}"
        print(f"Counting rows with date filter: {combined_sql_filter_clause if combined_sql_filter_clause else 'None'}...")
        count_result = db_client.execute(count_query)
        if not count_result or not count_result.result_rows:
            print(f"Error: Could not get row count from {PNL_TABLE} with applied filters.")
            return exported_trade_identifiers
        total_rows_found = count_result.result_rows[0][0]
        if total_rows_found == 0:
            print(f"No PNL data found matching date filters to export.")
            return exported_trade_identifiers

        print(f"Total P&L records matching date filters found: {total_rows_found}. Fetching all...")

        # 4. Fetch all data matching date filters, ordered by label timestamp
        # Select all columns needed, including the label_timestamp for identifiers
        query = f"""
        SELECT * 
        FROM {base_table_expression} {where_clause}
        ORDER BY label_timestamp -- Order by label time initially for consistency
        """
        pnl_df = db_client.query_dataframe(query)
        print(f"Fetched {len(pnl_df)} records.")

        if pnl_df is None or pnl_df.empty:
            print("No data fetched or an error occurred. Aborting.")
            return exported_trade_identifiers

        # --- Apply Sampling in Pandas (if enabled) --- 
        if use_random_sample and not pnl_df.empty:
            original_size = len(pnl_df)
            pnl_df = pnl_df.sample(frac=sample_fraction)
            print(f"Applied pandas sampling ({sample_fraction * 100:.1f}%): {original_size} -> {len(pnl_df)} rows.")
            if pnl_df.empty:
                print("DataFrame became empty after sampling. No data to export.")
                return exported_trade_identifiers
        # --- End Sampling ---

        print("Transforming data...")

        # Ensure timestamp columns are pandas Timestamps (UTC)
        # Include label_timestamp here as it's used for identifiers
        for col in ['label_timestamp', 'entry_timestamp', 'exit_timestamp']:
            if col in pnl_df.columns:
                pnl_df[col] = pd.to_datetime(pnl_df[col], utc=True)
            else:
                print(f"Warning: Expected timestamp column '{col}' not found in PNL data.")
                # Handle missing column -> return empty identifiers as we can't proceed safely
                return exported_trade_identifiers

        # 5. Generate Trade Legs
        all_trade_legs = []
        # Use a dictionary to map original PNL table index to the trade identifier
        # This avoids repeatedly accessing the DataFrame row inside the loop
        trade_id_map = {
            idx: (row['ticker'], row['label_timestamp'])
            for idx, row in pnl_df[['ticker', 'label_timestamp']].iterrows()
        }

        for idx, row in pnl_df.iterrows():
            # Use idx to lookup the identifier
            if idx not in trade_id_map:
                 continue # Should not happen
            trade_identifier = trade_id_map[idx]
            symbol = trade_identifier[0] # Get ticker from identifier

            # Use original share_size from DB, apply multiplier
            original_quantity = row['share_size']
            quantity = int(round(original_quantity * size_multiplier))

            entry_date, entry_time = format_timestamp(row['entry_timestamp'])
            exit_date, exit_time = format_timestamp(row['exit_timestamp'])

            if not entry_date or not exit_date:
                continue

            try:
                entry_bid = float(row['entry_bid_price'])
                entry_ask = float(row['entry_ask_price'])
                exit_bid = float(row['exit_bid_price'])
                exit_ask = float(row['exit_ask_price'])
            except (ValueError, TypeError):
                continue

            entry_dt = row['entry_timestamp']
            exit_dt = row['exit_timestamp']

            if pd.isna(entry_dt) or pd.isna(exit_dt): continue

            # Add legs with trade_id (original index), leg_type, and identifier
            if row['pos_long'] == 1:
                # Entry Leg (Buy) - Use ENTRY BID price for CSV
                all_trade_legs.append({'datetime': entry_dt, 'date': entry_date, 'time': entry_time, 'symbol': symbol, 'quantity': quantity, 'price': f"{entry_bid:.2f}", 'side': "Buy", 'trade_id': idx, 'leg_type': 'entry', 'identifier': trade_identifier})
                # Exit Leg (Sell) - Use EXIT ASK price for CSV
                all_trade_legs.append({'datetime': exit_dt, 'date': exit_date, 'time': exit_time, 'symbol': symbol, 'quantity': quantity, 'price': f"{exit_ask:.2f}", 'side': "Sell", 'trade_id': idx, 'leg_type': 'exit', 'identifier': trade_identifier})
            elif row['pos_short'] == 1:
                # Entry Leg (Sell) - Use ENTRY ASK price for CSV
                all_trade_legs.append({'datetime': entry_dt, 'date': entry_date, 'time': entry_time, 'symbol': symbol, 'quantity': quantity, 'price': f"{entry_ask:.2f}", 'side': "Sell", 'trade_id': idx, 'leg_type': 'entry', 'identifier': trade_identifier})
                # Exit Leg (Buy) - Use EXIT BID price for CSV
                all_trade_legs.append({'datetime': exit_dt, 'date': exit_date, 'time': exit_time, 'symbol': symbol, 'quantity': quantity, 'price': f"{exit_bid:.2f}", 'side': "Buy", 'trade_id': idx, 'leg_type': 'exit', 'identifier': trade_identifier})

        print(f"Generated {len(all_trade_legs)} potential trade legs.")

        # 5.5 Apply Time Filter (if enabled)
        if use_time_filter:
            print("Applying market hours filter based on ENTRY time (9:30:00 to 16:00:00 US/Eastern Time, DST aware)...")
            valid_trade_ids_time = set()
            try:
                eastern_tz = ZoneInfo("America/New_York")
                market_open_time = time(9, 30, 0)
                market_close_time = time(16, 0, 0)
            except Exception as tz_err:
                print(f"  - ERROR initializing timezone or times: {tz_err}. Skipping time filter.")
                valid_trade_ids_time = {leg['trade_id'] for leg in all_trade_legs}
            else:
                for leg in all_trade_legs:
                    if leg['leg_type'] == 'entry':
                        utc_dt = leg['datetime']
                        if utc_dt.tzinfo is None: utc_dt = utc_dt.tz_localize('UTC')
                        elif str(utc_dt.tzinfo) != 'UTC': utc_dt = utc_dt.tz_convert('UTC')

                        try:
                            eastern_dt = utc_dt.tz_convert(eastern_tz)
                        except Exception: continue

                        leg_time_eastern = eastern_dt.time()
                        if market_open_time <= leg_time_eastern <= market_close_time:
                            valid_trade_ids_time.add(leg['trade_id'])

                original_leg_count = len(all_trade_legs)
                filtered_legs_time = [leg for leg in all_trade_legs if leg['trade_id'] in valid_trade_ids_time]
                all_trade_legs = filtered_legs_time
                print(f"Finished Python market hours filtering. Kept {len(all_trade_legs)} legs from {len(valid_trade_ids_time)} trades (originally {original_leg_count} legs).")

        # 6. Sort all trade legs chronologically
        print("Sorting all trade legs chronologically...")
        all_trade_legs.sort(key=lambda x: (x['datetime'], x['leg_type'] == 'exit'))
        print("Sorting complete.")

        # 7. Apply Max Concurrent Positions Filter (if enabled)
        final_output_rows = [CSV_HEADER]
        if max_concurrent_positions is not None:
            print(f"Applying max concurrent positions limit ({max_concurrent_positions})...")
            open_positions = defaultdict(int)
            included_trade_ids_concurrency = set()
            final_filtered_legs = []

            for leg in all_trade_legs:
                symbol = leg['symbol']
                trade_id = leg['trade_id']
                leg_type = leg['leg_type']
                identifier = leg['identifier'] # Get identifier (ticker, label_timestamp)

                if leg_type == 'entry':
                    if open_positions[symbol] < max_concurrent_positions:
                        open_positions[symbol] += 1
                        included_trade_ids_concurrency.add(trade_id)
                        final_filtered_legs.append(leg)
                        exported_trade_identifiers.add(identifier) # Add identifier to the set
                    else:
                        pass # Skip entry leg
                elif leg_type == 'exit':
                    if trade_id in included_trade_ids_concurrency:
                        open_positions[symbol] = max(0, open_positions[symbol] - 1)
                        final_filtered_legs.append(leg)
                        # Do NOT add identifier again here
                    else:
                        pass # Skip exit leg

            for leg in final_filtered_legs:
                 final_output_rows.append([leg['date'], leg['time'], leg['symbol'], leg['quantity'], leg['price'], leg['side']])
            print(f"Finished applying concurrency limit. {len(final_output_rows) - 1} trade legs remain.")

        else:
            # If no concurrency limit, format all sorted legs and collect identifiers
            print("No concurrency limit applied. Formatting all legs...")
            trade_ids_in_output = set()
            for leg in all_trade_legs:
                final_output_rows.append([leg['date'], leg['time'], leg['symbol'], leg['quantity'], leg['price'], leg['side']])
                if leg['trade_id'] not in trade_ids_in_output:
                    exported_trade_identifiers.add(leg['identifier']) # identifier = (ticker, label_timestamp)
                    trade_ids_in_output.add(leg['trade_id'])
            print(f"Formatted {len(final_output_rows) - 1} trade legs.")

        # 8. Write the final data to partitioned CSV files
        MAX_ROWS_PER_FILE = 100000
        data_rows = final_output_rows[1:]
        num_data_rows = len(data_rows)

        if num_data_rows > 0:
            num_files = (num_data_rows + (MAX_ROWS_PER_FILE - 1) - 1) // (MAX_ROWS_PER_FILE - 1)
            print(f"Total data rows: {num_data_rows}. Splitting into {num_files} file(s) with max {MAX_ROWS_PER_FILE} rows each...")

            rows_written_total = 0
            for i in range(num_files):
                part_num = i + 1
                part_output_csv_path = os.path.join(OUTPUT_DIR, f"backtest_results_{base_timestamp_str}_part{part_num}.csv")
                start_index = i * (MAX_ROWS_PER_FILE - 1)
                end_index = min(start_index + (MAX_ROWS_PER_FILE - 1), num_data_rows)
                rows_in_this_part = [CSV_HEADER] + data_rows[start_index:end_index]
                num_legs_this_part = len(rows_in_this_part) - 1

                print(f"Writing part {part_num}/{num_files} ({num_legs_this_part} legs) to {part_output_csv_path}...")
                try:
                    with open(part_output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows(rows_in_this_part)
                    print(f"Successfully wrote part {part_num}")
                    rows_written_total += num_legs_this_part
                except IOError as e:
                    print(f"Error writing CSV file {part_output_csv_path}: {e}")
                    break

            print(f"Finished writing. Total legs written across all parts: {rows_written_total}")
        else:
            print("No trade legs to write after filtering.")

    except Exception as e:
        print(f"An unexpected error occurred during CSV export: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if db_client:
            db_client.close()

    return exported_trade_identifiers # Return the set of (ticker, label_timestamp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export P&L data (pattern-based) to CSV, sorted, with filters.") # Updated description
    parser.add_argument(
        "--size-multiplier", type=float, default=1.0,
        help="Multiplier for share_size. Default=1.0."
    )
    parser.add_argument(
        "--use-time-filter", action='store_true',
        help="Enable filtering for US market hours (9:30-16:00 Eastern)."
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Start date filter (inclusive), YYYY-MM-DD."
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="End date filter (inclusive), YYYY-MM-DD."
    )
    parser.add_argument(
        "--use-random-sample", action='store_true',
        help="Enable random sampling."
    )
    parser.add_argument(
        "--sample-fraction", type=float, default=0.1,
        help="Fraction for random sample (0 < f <= 1). Default=0.1."
    )
    parser.add_argument(
        "--max-concurrent-positions", type=int, default=None,
        help="Max concurrent open positions per symbol. Default=None (no limit)."
    )

    args = parser.parse_args()

    # Validations
    if args.size_multiplier <= 0:
        parser.error("Size multiplier must be positive.")
    if args.use_random_sample and not (0 < args.sample_fraction <= 1):
         parser.error(f"Sample fraction must be > 0 and <= 1, got: {args.sample_fraction}")
    if args.max_concurrent_positions is not None and args.max_concurrent_positions <= 0:
        parser.error(f"Max concurrent positions must be positive, got: {args.max_concurrent_positions}")

    # --- Removed config loading here - filters are applied upstream --- #

    print("=== Starting Backtest Results CSV Export (Pattern Based) ===")
    exported_ids = export_pnl_to_csv(
        # Pass args directly
        size_multiplier=args.size_multiplier,
        use_time_filter=args.use_time_filter,
        start_date_str=args.start_date,
        end_date_str=args.end_date,
        use_random_sample=args.use_random_sample,
        sample_fraction=args.sample_fraction,
        max_concurrent_positions=args.max_concurrent_positions
    )
    print(f"Export function returned {len(exported_ids)} unique trade identifiers ((ticker, label_timestamp)).")
    print("================================================================") 