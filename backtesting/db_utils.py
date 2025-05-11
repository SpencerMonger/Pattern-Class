import os
import clickhouse_connect
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import pytz # Add pytz for timezone handling

class ClickHouseClient:
    """Handles synchronous connection and operations with ClickHouse."""

    def __init__(self):
        # Load environment variables from .env in the same directory as this script
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        if not os.path.exists(dotenv_path):
            raise FileNotFoundError(f"Could not find .env file at {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path)

        self.host = os.getenv("CLICKHOUSE_HOST")
        self.port = int(os.getenv("CLICKHOUSE_HTTP_PORT", "8443"))
        self.user = os.getenv("CLICKHOUSE_USER")
        self.password = os.getenv("CLICKHOUSE_PASSWORD")
        self.database = os.getenv("CLICKHOUSE_DATABASE")
        self.secure = os.getenv("CLICKHOUSE_SECURE", "true").lower() == "true"
        self.client: Optional[clickhouse_connect.driver.Client] = None

        if not all([self.host, self.user, self.password, self.database]):
            raise ValueError("Missing required ClickHouse credentials in .env file")

        self.connect()

    def connect(self):
        """Establishes the connection to ClickHouse."""
        try:
            self.client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                username=self.user,
                password=self.password,
                database=self.database,
                secure=self.secure
            )
            self.client.command("SELECT 1") # Test connection
            print("Successfully connected to ClickHouse.")
        except Exception as e:
            print(f"Error connecting to ClickHouse: {e}")
            self.client = None
            raise  # Re-raise the exception after printing

    def close(self):
        """Closes the connection to ClickHouse."""
        if self.client:
            try:
                self.client.close()
                print("Closed ClickHouse connection.")
            except Exception as e:
                print(f"Error closing ClickHouse connection: {e}")
            finally:
                self.client = None

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Executes a query and returns the result summary."""
        if not self.client:
            print("Error: ClickHouse client is not connected.")
            return None
        try:
            # Use query for commands, query_df for selects returning dataframes if needed later
            result = self.client.query(query, parameters=params)
            return result
        except Exception as e:
            print(f"Error executing query: {e}\nQuery: {query}")
            return None

    def query_dataframe(self, query: str, params: Optional[Dict[str, Any]] = None):
        """Executes a query and returns the result as a pandas DataFrame."""
        if not self.client:
            print("Error: ClickHouse client is not connected.")
            return None
        try:
            df = self.client.query_df(query, parameters=params)
            return df
        except Exception as e:
            print(f"Error executing query for DataFrame: {e}\nQuery: {query}")
            return None

    def insert(self, table_name: str, data: List[Dict[str, Any]], column_names: Optional[List[str]] = None) -> None:
        """Inserts data into the specified table."""
        if not self.client:
            print("Error: ClickHouse client is not connected.")
            return
        if not data:
            print(f"No data provided for insertion into {table_name}.")
            return

        # Infer column names if not provided
        if not column_names:
            column_names = list(data[0].keys())

        # Prepare values list
        values = []
        for row in data:
            values.append([row.get(col) for col in column_names])

        try:
            self.client.insert(f"`{self.database}`.`{table_name}`", values, column_names=column_names)
            print(f"Successfully inserted {len(values)} rows into {table_name}.")
        except Exception as e:
            print(f"Error inserting data into {table_name}: {e}")

    def table_exists(self, table_name: str) -> bool:
        """Checks if a table exists in the database."""
        if not self.client:
            print("Error: ClickHouse client is not connected.")
            return False # Assume doesn't exist if not connected
        try:
            # Use a query that returns nothing but fails if table doesn't exist
            self.client.command(f"SELECT 1 FROM `{self.database}`.`{table_name}` WHERE 1=0")
            return True
        except Exception as e:
            # Check specific error message for non-existence
            err_str = str(e).lower()
            if (
                ("table" in err_str and ("doesn't exist" in err_str or "does not exist" in err_str)) or
                "unknown table" in err_str or
                "code: 60" in err_str # Explicitly check for error code 60 string
            ):
                return False
            else:
                # Log other unexpected errors
                print(f"Error checking table existence for {table_name}: {e}")
                return False

    def drop_table_if_exists(self, table_name: str) -> None:
        """Drops a table if it exists, ensuring the operation is synchronous."""
        if not self.client:
            print("Error: ClickHouse client is not connected.")
            return
        try:
            drop_command = f"DROP TABLE IF EXISTS `{self.database}`.`{table_name}` SYNC"
            print(f"Executing: {drop_command}") # Added print for visibility
            self.client.command(drop_command)
            print(f"Table {table_name} dropped SYNC (if it existed).")
        except Exception as e:
            print(f"Error dropping table {table_name} SYNC: {e}")

    def create_table_if_not_exists(self, table_name: str, schema: Dict[str, str], engine_clause: str, order_by_clause: str, primary_key_clause: Optional[str] = None) -> None:
        """Creates a table if it doesn't already exist."""
        if not self.client:
            print("Error: ClickHouse client is not connected.")
            return
        if self.table_exists(table_name):
            print(f"Table {table_name} already exists.")
            return

        columns_def = ", ".join([f"`{col}` {dtype}" for col, dtype in schema.items()])
        pk_clause = f"{primary_key_clause}" if primary_key_clause else ""
        # Add settings only if engine is not Memory
        settings_clause = "SETTINGS index_granularity = 8192" if "Memory" not in engine_clause else ""
        query = f"""
        CREATE TABLE IF NOT EXISTS `{self.database}`.`{table_name}` (
            {columns_def}
        ) {engine_clause}
        {pk_clause}
        {order_by_clause}
        {settings_clause}
        """
        try:
            self.client.command(query)
            print(f"Successfully created table {table_name}.")
        except Exception as e:
            print(f"Error creating table {table_name}: {e}\nQuery: {query}")
            raise # Re-raise to signal failure

    # Helper to ensure UTC similar to endpoints/db.py but synchronous
    def _ensure_utc(self, timestamp: Any) -> Optional[datetime]:
        if timestamp is None: return None
        dt = None
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None: dt = pytz.UTC.localize(timestamp)
            else: dt = timestamp.astimezone(pytz.UTC)
        elif isinstance(timestamp, (int, float)):
            try:
                # Attempt to handle seconds, milliseconds, and nanoseconds
                ts_val = float(timestamp)
                if len(str(int(ts_val))) >= 17: # Likely nanoseconds
                    dt = datetime.fromtimestamp(ts_val / 1e9, tz=pytz.UTC)
                elif len(str(int(ts_val))) >= 11: # Likely milliseconds
                    dt = datetime.fromtimestamp(ts_val / 1e3, tz=pytz.UTC)
                else: # Likely seconds
                    dt = datetime.fromtimestamp(ts_val, tz=pytz.UTC)
            except (ValueError, OSError) as e:
                print(f"Warning: Could not convert numeric timestamp {timestamp} to datetime: {e}")
                return None
        elif isinstance(timestamp, str):
            # Add more formats if needed, handle potential timezone info in string
            formats = ['%Y-%m-%d %H:%M:%S.%f%z', '%Y-%m-%d %H:%M:%S%z',
                       '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
                       '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ',
                       '%Y-%m-%d']
            for fmt in formats:
                try:
                    dt_naive_or_aware = datetime.strptime(timestamp, fmt)
                    if dt_naive_or_aware.tzinfo is None:
                        dt = pytz.UTC.localize(dt_naive_or_aware)
                    else:
                        dt = dt_naive_or_aware.astimezone(pytz.UTC)
                    break # Success
                except ValueError:
                    continue
            # Final attempt with fromisoformat for robust ISO 8601 parsing
            if dt is None:
                 try:
                     dt_aware = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                     dt = dt_aware.astimezone(pytz.UTC)
                 except ValueError:
                      print(f"Warning: Could not parse string timestamp '{timestamp}' into UTC datetime.")
                      return None

        # Attempt to preserve nanoseconds if original was high precision numeric
        if dt is not None and isinstance(timestamp, (int, float)) and len(str(int(timestamp))) >= 17:
             nanos = int(timestamp % 1e9)
             # ClickHouse DateTime64(9) stores nanoseconds
             # Python datetime microsecond precision is enough for this conversion
             dt = dt.replace(microsecond=nanos // 1000)

        return dt

# Example usage (optional - can be removed or commented out)
# if __name__ == "__main__":
#     try:
#         db = ClickHouseClient()
#         # Example: Check if 'stock_quotes' table exists
#         exists = db.table_exists("stock_quotes")
#         print(f"Table 'stock_quotes' exists: {exists}")
#         # Example: Execute a simple query
#         result = db.execute("SELECT COUNT(*) FROM stock_quotes")
#         if result:
#             print(f"Count result: {result.result_rows}")
#         # Example: Query to DataFrame
#         df = db.query_dataframe("SELECT * FROM stock_quotes LIMIT 5")
#         if df is not None:
#             print("Sample data from stock_quotes:")
#             print(df)
#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         if 'db' in locals() and db.client:
#             db.close() 