import pandas as pd
import sys
import os

# Adjust path to import db_utils, which is in the same directory as this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is .../img-class/backtesting
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR) # Add .../img-class/backtesting to the path

try:
    from db_utils import ClickHouseClient # Should now find .../img-class/backtesting/db_utils.py
except ImportError as e:
    print(f"Error importing ClickHouseClient: {e}")
    print(f"Ensure db_utils.py is in the same directory as this script: {SCRIPT_DIR}")
    sys.exit(1)

def apply_breakout_logic_for_candle(df, candle_timestamp_utc_str, high_lookback_period=30):
    """
    Applies the breakout logic to a specific candle in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with OHLC data, lowercase columns, DatetimeIndex (UTC).
        candle_timestamp_utc_str (str): The timestamp of the candle to check (e.g., '2023-09-11 17:17:00').
        high_lookback_period (int): Number of preceding candles for high lookback.

    Returns:
        None. Prints debug information.
    """
    target_timestamp = pd.Timestamp(candle_timestamp_utc_str, tz='UTC')

    if target_timestamp not in df.index:
        print(f"Target timestamp {target_timestamp} not found in DataFrame index.")
        return

    # Get integer position of the target candle
    idx = df.index.get_loc(target_timestamp)

    if idx < high_lookback_period:
        print(f"Not enough preceding data for candle at {target_timestamp} (index {idx}) to satisfy lookback of {high_lookback_period}.")
        return

    # --- Replicated logic from breakout.py's logic() method ---
    start_lookback_idx = max(0, idx - high_lookback_period)
    preceding_highs = df['high'].iloc[start_lookback_idx:idx] # Slice up to, but not including, idx

    if preceding_highs.empty:
        print(f"For candle {target_timestamp}: Preceding highs slice is empty.")
        return
    
    valid_preceding_highs = preceding_highs.dropna()
    if valid_preceding_highs.empty:
        print(f"For candle {target_timestamp}: No valid preceding highs after dropna().")
        return
        
    thirty_min_high = valid_preceding_highs.max()

    current_open = df['open'].iloc[idx]
    current_high = df['high'].iloc[idx]
    current_low = df['low'].iloc[idx]
    current_close = df['close'].iloc[idx]
    # --- End replicated logic ---

    print(f"\n--- DEBUG FOR {target_timestamp} ---")
    print(f"  Lookback Period: {high_lookback_period} candles")
    print(f"  Preceding Window (index positions): {start_lookback_idx} to {idx-1}")
    if not preceding_highs.empty:
        print(f"  Preceding Window (timestamps): {preceding_highs.index[0]} to {preceding_highs.index[-1]}")
    print(f"  Calculated Peak of Preceding {len(preceding_highs)} Highs: {thirty_min_high:.4f}") # Format for readability
    print(f"  Snippet of preceding 5 highs: {preceding_highs.tail(5).round(4).to_list()}")
    print(f"  Current Candle ({target_timestamp}):")
    print(f"    Open:  {current_open:.4f}")
    print(f"    High:  {current_high:.4f}")
    print(f"    Low:   {current_low:.4f}")
    print(f"    Close: {current_close:.4f}")
    
    is_breakout = (
        current_open > thirty_min_high or
        current_high > thirty_min_high or
        current_low > thirty_min_high or 
        current_close > thirty_min_high
    )
    print(f"  Breakout Condition (any OHLC > {thirty_min_high:.4f}): {is_breakout}")
    if is_breakout:
        if current_open > thirty_min_high: print("    Breakout due to Open")
        if current_high > thirty_min_high: print("    Breakout due to High")
        if current_low > thirty_min_high: print("    Breakout due to Low")
        if current_close > thirty_min_high: print("    Breakout due to Close")
    print("---")

def main():
    db_client = None
    try:
        db_client = ClickHouseClient() # Assumes your DB connection details are handled by ClickHouseClient defaults
        
        ticker_to_test = 'META'
        # Query a slightly wider window to ensure enough preceding data for the first candle we test
        start_query_dt_str = "2023-09-11 16:40:00" # 30 mins before 17:10 approx
        end_query_dt_str = "2023-09-11 17:25:00"   # A bit after our target range

        query = f"""
        SELECT timestamp, ticker, open, high, low, close, volume
        FROM {db_client.database}.stock_bars
        WHERE ticker = '{ticker_to_test}'
          AND timestamp >= toDateTime('{start_query_dt_str}', 'UTC')
          AND timestamp < toDateTime('{end_query_dt_str}', 'UTC')
        ORDER BY timestamp ASC
        """
        print(f"Executing query for {ticker_to_test} from {start_query_dt_str} to {end_query_dt_str}...")
        df = db_client.query_dataframe(query)

        if df is None or df.empty:
            print("No data returned from query.")
            return

        print(f"Loaded {len(df)} rows for {ticker_to_test}.")

        # Prepare DataFrame: ensure datetime index and lowercase columns
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df.set_index('timestamp', inplace=True)
        df.columns = [col.lower() for col in df.columns]
        required_ohlc = ['open', 'high', 'low', 'close']
        for col in required_ohlc:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
            df[col] = pd.to_numeric(df[col], errors='raise')

        # Test specific candles
        candles_to_test = [
            '2023-09-11 17:16:00',
            '2023-09-11 17:17:00',
            '2023-09-11 17:18:00'
        ]

        for ts_str in candles_to_test:
            apply_breakout_logic_for_candle(df, ts_str)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if db_client:
            db_client.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main() 