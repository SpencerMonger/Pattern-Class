import os

# Assuming db_utils is in the same directory
from db_utils import ClickHouseClient

# --- Configuration ---
# Database table name to drop
PNL_TABLE = "stock_pnl"
# ---------------------

def drop_pnl_table():
    """Connects to the database and drops the PNL_TABLE if it exists."""
    db_client = None
    print(f"Attempting to drop table: {PNL_TABLE}...")
    try:
        # 1. Initialize Database Connection
        db_client = ClickHouseClient()
        db_name = db_client.database # For confirmation message

        # 2. Drop the table
        db_client.drop_table_if_exists(PNL_TABLE)
        # The drop_table_if_exists method in db_utils already prints a confirmation
        print(f"Operation completed for table '{PNL_TABLE}' in database '{db_name}'.")

    except Exception as e:
        print(f"An unexpected error occurred while trying to drop table '{PNL_TABLE}': {e}")
        import traceback
        traceback.print_exc()
    finally:
        if db_client:
            db_client.close()

if __name__ == "__main__":
    print("=== Starting Drop PnL Table Script ===")
    drop_pnl_table()
    print("=====================================") 