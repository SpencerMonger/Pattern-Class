import os
import pandas as pd
from datetime import timedelta

# Assuming db_utils is in the same directory
from db_utils import ClickHouseClient

# --- Configuration ---
# Database table name to read from
PNL_TABLE = "stock_pnl"

# Filter for prediction categories
CATEGORY_FILTER = "prediction_cat IN (0, 1, 4, 5)"

# Time window for concurrency
CONCURRENCY_WINDOW = timedelta(minutes=15)
# ---------------------

def count_concurrent_trades():
    """Fetches relevant trade timestamps and counts trades concurrent within a time window."""
    db_client = None
    print(f"Counting concurrent trades from '{PNL_TABLE}'...")
    print(f"Filter: {CATEGORY_FILTER}")
    print(f"Concurrency Window: {CONCURRENCY_WINDOW}")
    try:
        # 1. Initialize Database Connection
        db_client = ClickHouseClient()
        db_name = db_client.database

        # 2. Check if PNL table exists
        if not db_client.table_exists(PNL_TABLE):
            print(f"Error: PnL table '{PNL_TABLE}' not found. Please run calculate_pnl.py first.")
            return

        # 3. Fetch filtered prediction timestamps
        query = f"""
        SELECT prediction_timestamp
        FROM `{db_name}`.`{PNL_TABLE}`
        WHERE {CATEGORY_FILTER}
        ORDER BY prediction_timestamp -- Crucial for comparison
        """
        print("Fetching timestamps...")
        pnl_df = db_client.query_dataframe(query)

        if pnl_df is None:
            print("Error fetching data.")
            return
        if pnl_df.empty:
            print("No trades found matching the filter criteria.")
            print("Number of concurrent trades: 0")
            return

        print(f"Fetched {len(pnl_df)} relevant trade timestamps.")

        # 4. Ensure timestamps are datetime objects and sorted (redundant due to ORDER BY, but safe)
        pnl_df['prediction_timestamp'] = pd.to_datetime(pnl_df['prediction_timestamp'], utc=True)
        pnl_df.sort_values('prediction_timestamp', inplace=True)
        pnl_df.reset_index(drop=True, inplace=True) # Reset index for easier iloc access

        # 5. Identify indices of concurrent trades
        concurrent_trade_indices = set()
        if len(pnl_df) > 1:
            print("Checking for concurrency...")
            # Iterate up to the second-to-last row
            for i in range(len(pnl_df) - 1):
                t_current = pnl_df.iloc[i]['prediction_timestamp']
                t_next = pnl_df.iloc[i+1]['prediction_timestamp']

                # Check if the next trade is within the window
                if (t_next - t_current) <= CONCURRENCY_WINDOW:
                    # Both trades are involved in this concurrent pair
                    concurrent_trade_indices.add(i)
                    concurrent_trade_indices.add(i+1)
        else:
            print("Only one trade found, cannot have concurrency.")

        # 6. Report the count
        concurrent_count = len(concurrent_trade_indices)
        print(f"\nNumber of trades involved in concurrency: {concurrent_count}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if db_client:
            db_client.close()

if __name__ == "__main__":
    print("=== Starting Concurrent Trade Counter ===")
    count_concurrent_trades()
    print("========================================") 