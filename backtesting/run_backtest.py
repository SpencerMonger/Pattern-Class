#!/usr/bin/env python3
import argparse
import yaml
import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np
# Removed sklearn imports as confusion matrix/accuracy are no longer used for labels
# from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error
import re
from typing import Set, Tuple, Dict, Optional # Added Dict and Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the backtesting directory is in the path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
# Ensure ohlc_detection is also in the path for label_shapes import within generate_pattern_labels
OHLC_DIR = os.path.join(os.path.dirname(script_dir), 'ohlc_detection')
if OHLC_DIR not in sys.path:
    sys.path.insert(0, OHLC_DIR)

# Import the main functions from the other scripts using direct imports
# as sys.path.append(script_dir) should make them available
try:
    from generate_pattern_labels import run_label_generation
    from calculate_pnl import run_pnl_calculation, SHARE_SIZE
    from export_pnl import export_pnl_to_csv
    from db_utils import ClickHouseClient
    from macd import update_labels_with_macd_info
except ImportError as e:
    print(f"Error importing required functions: {e}")
    print("Ensure generate_pattern_labels.py, calculate_pnl.py, export_pnl.py, db_utils.py, macd.py (all in backtesting directory) and label_shapes.py (in ohlc_detection) are accessible and sys.path is correct.")
    sys.exit(1)

def load_config(config_path: str) -> dict:
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None:
            print(f"Warning: Config file '{config_path}' is empty or invalid.")
            return {}
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at '{config_path}'")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file '{config_path}': {e}")
        sys.exit(1)

def validate_date(date_str: Optional[str]) -> Optional[str]: # Changed to Optional[str]
    """Validates date string format YYYY-MM-DD and ensures leading zeros."""
    if date_str is None:
        return None
    try:
        dt_obj = datetime.strptime(date_str, '%Y-%m-%d')
        # Reformat to ensure YYYY-MM-DD with leading zeros for ClickHouse
        return dt_obj.strftime('%Y-%m-%d')
    except ValueError:
        print(f"Error: Invalid date format '{date_str}'. Please use YYYY-MM-DD.")
        sys.exit(1)

# --- Metrics Calculation Function (Modified for Pattern Labels + Accuracy) --- #
def calculate_and_print_metrics(
    exported_trade_ids: Set[Tuple[str, pd.Timestamp]],
    config: Dict, # Pass config for context if needed, e.g., size multiplier
    size_multiplier: float # Explicitly pass size multiplier used in export
):
    """Queries PNL table for exported trades and calculates P&L and accuracy metrics per pattern label."""
    db_client = None
    temp_table_name = None # Initialize temporary table name
    if not exported_trade_ids:
        print("\n--- Metrics Calculation --- ")
        print("Warning: No trade identifiers were provided from the export step. Cannot calculate metrics.")
        print("--- End of Metrics Calculation ---")
        return

    print(f"\n--- Metrics Calculation (Based on {len(exported_trade_ids)} Exported Trades) --- ")
    try:
        db_client = ClickHouseClient()
        db_name = db_client.database
        pnl_table = "stock_pnl"

        if not db_client.table_exists(pnl_table):
            print(f"Error: PNL table '{pnl_table}' not found. Cannot calculate metrics.")
            return

        # --- Use Temporary Table for Identifiers ---
        timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S%f")
        temp_table_name = f"temp_exported_ids_{timestamp_str}"
        print(f"Creating temporary table: {temp_table_name} for {len(exported_trade_ids)} identifiers...")

        temp_table_schema = {
            'ticker': 'String',
            'label_timestamp': "DateTime64(9, 'UTC')"
        }
        # Using Memory engine for temporary table
        temp_engine = "ENGINE = Memory"
        temp_order_by = "" # No specific order needed for Memory engine join key
        temp_primary_key = ""

        # Drop if it somehow exists (shouldn't, due to timestamp)
        db_client.drop_table_if_exists(temp_table_name)

        # Create the temporary table
        db_client.create_table_if_not_exists(
            temp_table_name,
            temp_table_schema,
            temp_engine,
            temp_order_by,
            temp_primary_key
        )

        # Prepare data for insertion
        ids_to_insert = []
        for ticker, ts in exported_trade_ids:
            # Ensure timestamp is UTC for ClickHouse DateTime64('UTC')
            if ts.tzinfo is None: ts_utc = ts.tz_localize('UTC')
            else: ts_utc = ts.tz_convert('UTC')
            ids_to_insert.append({'ticker': ticker, 'label_timestamp': ts_utc})

        # Insert data into the temporary table
        # Note: ClickHouseClient's insert handles batching if needed
        db_client.insert(temp_table_name, ids_to_insert, column_names=list(temp_table_schema.keys()))
        print(f"Successfully inserted {len(ids_to_insert)} identifiers into {temp_table_name}.")

        # Query necessary columns: pattern_label, pnl_total, share_size, and price_diff_per_share for accuracy
        query = f"""
        SELECT
            pnl.pattern_label,
            pnl.pnl_total,
            pnl.share_size,
            pnl.price_diff_per_share -- <<< Added for accuracy calculation
        FROM `{db_name}`.`{pnl_table}` AS pnl
        -- Use SEMI JOIN for filtering based on existence in the temporary table
        SEMI LEFT JOIN `{db_name}`.`{temp_table_name}` AS temp_ids
        ON pnl.ticker = temp_ids.ticker AND pnl.label_timestamp = temp_ids.label_timestamp
        """
        metrics_df = db_client.query_dataframe(query)

        if metrics_df is None or metrics_df.empty:
            print(f"No PNL data found matching the identifiers in temporary table {temp_table_name}. Cannot calculate metrics.")
            return

        print(f"Retrieved {len(metrics_df)} PNL rows matching exported trades for metrics.")

        # --- Calculate P&L Summary and Accuracy per Pattern Label --- #

        # Adjust PNL based on the size multiplier used in export
        # Ensure share_size is numeric, default to SHARE_SIZE constant if missing/invalid
        metrics_df['share_size'] = pd.to_numeric(metrics_df['share_size'], errors='coerce').fillna(SHARE_SIZE)
        # Calculate the multiplier relative to the base SHARE_SIZE constant
        # Avoid division by zero if SHARE_SIZE is 0
        base_share_size = float(SHARE_SIZE) if SHARE_SIZE > 0 else 1.0
        # metrics_df['effective_multiplier'] = (metrics_df['share_size'] * size_multiplier) / base_share_size
        # metrics_df['adjusted_pnl'] = metrics_df['pnl_total'] * metrics_df['effective_multiplier'] / (metrics_df['share_size'] / base_share_size)
        # Simpler adjustment: pnl_total in DB was base size * diff. We want (base size * multiplier) * diff
        # So, just multiply pnl_total by the size_multiplier used in export.
        metrics_df['adjusted_pnl'] = metrics_df['pnl_total'] * size_multiplier
        # Ensure price_diff_per_share is numeric for accuracy calculation
        metrics_df['price_diff_per_share'] = pd.to_numeric(metrics_df['price_diff_per_share'], errors='coerce')

        print("\n--- P&L and Accuracy Summary per Pattern Label (Exported Trades, Size Adjusted) ---")
        # Group by pattern label
        grouped_metrics = metrics_df.groupby('pattern_label')

        summary_stats = []
        for label, group in grouped_metrics:
            trade_count = len(group)
            total_pnl = group['adjusted_pnl'].sum()
            average_pnl = total_pnl / trade_count if trade_count > 0 else 0
            # Calculate accuracy: % of trades with positive price diff
            positive_trades = (group['price_diff_per_share'] > 0).sum()
            accuracy = (positive_trades / trade_count * 100) if trade_count > 0 else 0

            summary_stats.append({
                'Pattern Label': label,
                'Trade Count': trade_count,
                'Accuracy (%)': accuracy, # <<< Added Accuracy
                'Total P&L': total_pnl,
                'Average P&L': average_pnl
            })

        if not summary_stats:
            print("No pattern groups found for summary.")
        else:
            summary_df = pd.DataFrame(summary_stats)
            # Define column order
            column_order = ['Pattern Label', 'Trade Count', 'Accuracy (%)', 'Total P&L', 'Average P&L']
            summary_df = summary_df[column_order]
            summary_df.sort_values(by='Total P&L', ascending=False, inplace=True)

            # Format for printing
            summary_df['Accuracy (%)'] = summary_df['Accuracy (%)'].map('{:.2f}%'.format)
            summary_df['Total P&L'] = summary_df['Total P&L'].map('${:,.2f}'.format)
            summary_df['Average P&L'] = summary_df['Average P&L'].map('${:,.2f}'.format)

            # Print the summary table
            print(summary_df.to_string(index=False))

        print("---------------------------------------------------------------------")

        # --- Removed Accuracy/Confusion Matrix calculations --- #

        print("--- End of Metrics Calculation ---")

    except Exception as e:
        print(f"\nError during metrics calculation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if db_client:
            # --- Drop Temporary Table ---
            if temp_table_name:
                print(f"Dropping temporary table: {temp_table_name}")
                db_client.drop_table_if_exists(temp_table_name)
            db_client.close()

# --- End Metrics Calculation Function --- #

def main():
    parser = argparse.ArgumentParser(description="Run the pattern-based backtesting pipeline: generate labels -> calculate P&L -> export -> metrics.") # Updated description
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(script_dir, "config.yaml"),
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Start date for the backtest (YYYY-MM-DD). Overrides config."
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="End date for the backtest (YYYY-MM-DD). Overrides config."
    )

    args = parser.parse_args()

    start_time = datetime.now()
    logger.info("=== BACKTESTING PIPELINE STARTED ===")

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    export_config = config.get('export_pnl', {}) # Get export_pnl settings
    # <<< Get label generation config >>>
    label_config = config.get('label_generation', {})
    # <<< Get pnl_calculation_settings config >>>
    pnl_calc_config = config.get('pnl_calculation_settings', {})

    # Determine start and end dates (command line overrides config)
    start_date = validate_date(args.start_date or config.get('start_date'))
    end_date = validate_date(args.end_date or config.get('end_date'))
    # Add validated dates back to config for pipeline steps
    config['start_date'] = start_date
    config['end_date'] = end_date

    # Validate date range logic
    if start_date and not end_date: parser.error("If --start-date is provided, --end-date is required.")
    if end_date and not start_date: parser.error("If --end-date is provided, --start-date is required.")
    if start_date and end_date:
        dt_start = datetime.strptime(start_date, '%Y-%m-%d')
        dt_end = datetime.strptime(end_date, '%Y-%m-%d')
        if dt_start > dt_end:
            parser.error("Start date cannot be after end date.")

    # === Pre-Step 2: Ensure stock_pnl is clean ===
    logger.info("\n" + "="*10 + " Pre-Step 2: Ensuring stock_pnl table is clean " + "="*10)
    pre_step_2_db_client = None
    pnl_table_to_clear = "stock_pnl"  # Move outside try block
    try:
        pre_step_2_db_client = ClickHouseClient()
        logger.info(f"  Attempting to DROP table `{pre_step_2_db_client.database}`.`{pnl_table_to_clear}` SYNC before Step 2.")
        pre_step_2_db_client.drop_table_if_exists(pnl_table_to_clear) # drop_table_if_exists now uses SYNC
    except Exception as e_pre_step_2:
        logger.error(f"  Error during Pre-Step 2 cleanup of {pnl_table_to_clear}: {e_pre_step_2}")
    finally:
        if pre_step_2_db_client:
            pre_step_2_db_client.close()
            logger.info(f"  Closed ClickHouse connection for Pre-Step 2 cleanup.")
    logger.info("="*10 + " End Pre-Step 2 Cleanup " + "="*10 + "\n")

    # === Step 1: Generate Pattern Labels ===
    logger.info("\n" + "="*10 + " Step 1: Generating Pattern Labels " + "="*10)
    try:
        # Pass the full config, start/end dates are read within the function now
        run_label_generation(config=config, start_date=start_date, end_date=end_date)
        logger.info("\n" + " Label generation step completed successfully. " + "\n")
    except Exception as e:
        logger.error(f"\nError during label generation step: {e}")
        import traceback
        traceback.print_exc()
        logger.error("Aborting pipeline.")
        sys.exit(1)

    # === Step 1.5: Update Labels with MACD Info (if enabled) ===
    # This step runs AFTER label generation and BEFORE P&L calculation
    logger.info("\n" + "="*10 + " Step 1.5: Updating Labels with MACD Info " + "="*10)
    if config.get('label_generation', {}).get('use_macd_predicate', False):
        macd_update_db_client: Optional[ClickHouseClient] = None
        try:
            logger.info("MACD predicate is ENABLED. Attempting to update labels table...")
            macd_update_db_client = ClickHouseClient()
            update_labels_with_macd_info(
                db_client=macd_update_db_client,
                config=config,
                start_date=start_date, # Pass validated start_date
                end_date=end_date     # Pass validated end_date
            )
            logger.info("Label update with MACD info step completed.")
        except Exception as e_macd_update:
            logger.warning(f"\nError during MACD label update step: {e_macd_update}")
            import traceback
            traceback.print_exc()
            # Decide if this is fatal. For now, we'll print a warning and continue.
            logger.warning("Warning: MACD label update failed. P&L calculation might not use MACD filtering as expected.")
        finally:
            if macd_update_db_client:
                macd_update_db_client.close()
                logger.info("Closed ClickHouse connection for MACD label update.")
    else:
        logger.info("MACD predicate is DISABLED in config. Skipping label update with MACD info.")
    logger.info("="*10 + " End Step 1.5: MACD Label Update " + "="*10 + "\n")

    # === Step 2: Calculate P&L ===
    logger.info("="*10 + " Step 2: Calculating P&L (Pattern Based) " + "="*10)
    try:
        pattern_directions = label_config.get('pattern_directions', {})
        if not pattern_directions:
             logger.warning("Warning: 'pattern_directions' not found or empty in config under 'label_generation'. P&L step might not generate trades.")

        use_model_predicate = label_config.get('use_model_predicate', False)
        model_predicate_threshold_config = label_config.get('model_predicate_threshold', None)
        
        effective_model_predicate_threshold: Optional[float] = None
        if use_model_predicate:
            if model_predicate_threshold_config is not None:
                try:
                    effective_model_predicate_threshold = float(model_predicate_threshold_config)
                    logger.info(f"Model predicate ENABLED. Using threshold: {effective_model_predicate_threshold}")
                except ValueError:
                    logger.warning(f"Warning: Invalid 'model_predicate_threshold' value '{model_predicate_threshold_config}'. Model predicate will be DISABLED.")
            else:
                logger.warning("Warning: 'use_model_predicate' is true, but 'model_predicate_threshold' is not set. Model predicate will be DISABLED.")
        else:
            logger.info("Model predicate DISABLED by config ('use_model_predicate': false).")

        default_hold_time = 15 * 60 # 15 minutes
        hold_time_seconds_config = pnl_calc_config.get('hold_time_seconds', default_hold_time)
        hold_time_seconds: int = default_hold_time
        try:
            hold_time_seconds = int(hold_time_seconds_config)
            if hold_time_seconds <= 0:
                logger.warning(f"Warning: 'hold_time_seconds' ({hold_time_seconds_config}) must be positive. Using default: {default_hold_time}s")
                hold_time_seconds = default_hold_time
            else:
                logger.info(f"Using P&L hold time: {hold_time_seconds} seconds.")
        except ValueError:
            logger.warning(f"Warning: Invalid 'hold_time_seconds' value '{hold_time_seconds_config}'. Using default: {default_hold_time}s")
            hold_time_seconds = default_hold_time

        run_pnl_calculation(
            pattern_directions=pattern_directions,
            start_date=start_date,
            end_date=end_date,
            model_predicate_threshold=effective_model_predicate_threshold,
            hold_time_seconds=hold_time_seconds,
            config=config
            )
        logger.info("\n" + " P&L calculation step completed successfully. " + "\n")
    except Exception as e:
        logger.error(f"\nError during P&L calculation step: {e}")
        import traceback
        traceback.print_exc()
        logger.error("Aborting pipeline.")
        sys.exit(1)

    # === Post-Step 2: DIAGNOSTIC & Conditional TRUNCATE of stock_pnl ===
    logger.info("\n" + "="*10 + " DIAGNOSTIC & TRUNCATE: Inspecting stock_pnl after Step 2 " + "="*10)
    diag_db_client: Optional[ClickHouseClient] = None
    pnl_table_name = "stock_pnl"  # Move outside try block
    try:
        diag_db_client = ClickHouseClient()
        db_name = diag_db_client.database
        actual_row_count = -1 

        if diag_db_client.table_exists(pnl_table_name):
            count_query = f"SELECT count() FROM `{db_name}`.`{pnl_table_name}`"
            count_result_df = diag_db_client.query_dataframe(count_query)
            
            if count_result_df is not None and not count_result_df.empty:
                actual_row_count = count_result_df.iloc[0,0]
                logger.info(f"  DIAGNOSTIC: Total rows in `{db_name}`.`{pnl_table_name}` after Step 2: {actual_row_count}")
                
                if actual_row_count > 0:
                    pass 
            else:
                logger.warning(f"  DIAGNOSTIC: Could not get row count for `{db_name}`.`{pnl_table_name}` or table might be empty.")

            if diag_db_client.table_exists(pnl_table_name):
                count_query_for_distinct = f"SELECT count() FROM `{db_name}`.`{pnl_table_name}`"
                current_count_df_for_distinct = diag_db_client.query_dataframe(count_query_for_distinct)
                if current_count_df_for_distinct is not None and not current_count_df_for_distinct.empty and current_count_df_for_distinct.iloc[0,0] > 0:
                    distinct_labels_query = f"""
                    SELECT pattern_label, count() as label_count
                    FROM `{db_name}`.`{pnl_table_name}`
                    GROUP BY pattern_label
                    ORDER BY label_count DESC
                    LIMIT 10
                    """
                    distinct_labels_df = diag_db_client.query_dataframe(distinct_labels_query)
                    if distinct_labels_df is not None and not distinct_labels_df.empty:
                        logger.info(f"  DIAGNOSTIC: Distinct pattern_labels in `{db_name}`.`{pnl_table_name}` (top 10 after any TRUNCATE):")
                        logger.info(distinct_labels_df.to_string(index=False))
                    else:
                        logger.warning(f"  DIAGNOSTIC: Could not retrieve distinct pattern_labels from `{db_name}`.`{pnl_table_name}` (but table had rows). Df was None/empty.")
                elif actual_row_count == 0: # This case means table was empty before this check or became empty.
                     logger.warning(f"  DIAGNOSTIC: Table `{db_name}`.`{pnl_table_name}` is confirmed empty or just became empty.")
                else: # current_count_df_for_distinct is None, empty or count is 0, but actual_row_count was >0 or -1
                     logger.warning(f"  DIAGNOSTIC: Table `{db_name}`.`{pnl_table_name}` is inaccessible or became empty unexpectedly after initial count.")       
            elif actual_row_count == 0:
                 logger.warning(f"  DIAGNOSTIC: Table `{db_name}`.`{pnl_table_name}` was already empty after Step 2, as expected.")

        else:
            logger.warning(f"  DIAGNOSTIC: Table `{db_name}`.`{pnl_table_name}` does not exist after Step 2 claimed to create it! This is unexpected.")
        logger.info(f"  DIAGNOSTIC: Total rows in `{db_name}`.`{pnl_table_name}` after Step 2: {actual_row_count}")
        sys.stdout.flush()
    except Exception as diag_e:
        logger.error(f"  DIAGNOSTIC: Error during inspection/truncate of {pnl_table_name}: {diag_e}")
        import traceback
        traceback.print_exc()
    finally:
        if diag_db_client:
            diag_db_client.close()
            logger.info(f"  DIAGNOSTIC: Closed ClickHouse connection for inspection/truncate.")
    logger.info("="*10 + " END DIAGNOSTIC & TRUNCATE " + "="*10 + "\n")

    # === Step 3: Export P&L Results ===
    exported_trade_ids: Set[Tuple[str, pd.Timestamp]] = set()
    size_multiplier = 1.0 # Initialize default outside try block
    logger.info("="*10 + " Step 3: Exporting P&L Results " + "="*10)
    try:
        size_multiplier = float(export_config.get('size_multiplier', 1.0))
        use_time_filter = bool(export_config.get('use_time_filter', False))
        use_random_sample = bool(export_config.get('use_random_sample', False))
        sample_fraction = float(export_config.get('sample_fraction', 0.1))
        max_concurrent_positions_cfg = export_config.get('max_concurrent_positions', None)
        max_concurrent_positions: Optional[int] = None
        if max_concurrent_positions_cfg is not None:
            max_concurrent_positions = int(max_concurrent_positions_cfg)

        if size_multiplier <= 0: raise ValueError("export_pnl.size_multiplier must be positive.")
        if use_random_sample and not (0 < sample_fraction <= 1): raise ValueError(f"export_pnl.sample_fraction must be > 0 and <= 1, got: {sample_fraction}")
        if max_concurrent_positions is not None and max_concurrent_positions <= 0: raise ValueError(f"export_pnl.max_concurrent_positions must be positive integer if set, got: {max_concurrent_positions}")

        exported_trade_ids = export_pnl_to_csv(
            size_multiplier=size_multiplier,
            use_time_filter=use_time_filter,
            start_date_str=start_date,
            end_date_str=end_date,
            use_random_sample=use_random_sample,
            sample_fraction=sample_fraction,
            max_concurrent_positions=max_concurrent_positions
        )
        logger.info(f"\nExport step completed successfully. {len(exported_trade_ids)} trades included in the export.")

    except ValueError as ve:
        logger.error(f"\nConfiguration error for export step: {ve}")
        logger.error("Aborting pipeline.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nError during P&L export step: {e}")
        import traceback
        traceback.print_exc()
        logger.error("Aborting pipeline.")
        sys.exit(1)

    # === Step 4: Calculate and Print Performance Metrics ===
    logger.info("="*10 + " Step 4: Calculating Performance Metrics (Pattern P&L) " + "="*10)
    try:
        calculate_and_print_metrics(
            exported_trade_ids=exported_trade_ids,
            config=config, 
            size_multiplier=size_multiplier
        )
        logger.info("\nMetrics calculation step completed.")
    except Exception as e:
        logger.warning(f"\nError during Metrics Calculation step: {e}")
        import traceback
        traceback.print_exc()
        logger.warning("Warning: Metrics calculation failed, but prior steps may have succeeded.")

    # Calculate and display comprehensive final metrics
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    logger.info("\n" + "="*60)
    logger.info("           BACKTESTING PIPELINE COMPLETED")
    logger.info("="*60)
    logger.info(f"Total execution time: {total_duration}")
    logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get comprehensive final metrics from database
    try:
        final_metrics_db_client = ClickHouseClient()
        
        # Pattern label statistics
        logger.info("\n--- PATTERN LABEL STATISTICS ---")
        try:
            label_stats_query = """
            SELECT 
                pattern_label,
                COUNT() as count,
                COUNT() * 100.0 / (SELECT COUNT() FROM stock_historical_labels) as percentage
            FROM stock_historical_labels 
            GROUP BY pattern_label 
            ORDER BY count DESC
            """
            label_stats = final_metrics_db_client.query_dataframe(label_stats_query)
            if label_stats is not None and not label_stats.empty:
                total_labels = label_stats['count'].sum()
                logger.info(f"Total pattern labels generated: {total_labels:,}")
                logger.info("Pattern distribution:")
                for _, row in label_stats.iterrows():
                    logger.info(f"  {row['pattern_label']}: {int(row['count']):,} ({row['percentage']:.1f}%)")
            else:
                logger.warning("No pattern label statistics available")
        except Exception as e:
            logger.warning("Could not retrieve pattern label statistics")
        
        final_metrics_db_client.close()
        
    except Exception as e:
        logger.warning(f"Could not retrieve final statistics: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("Pipeline completed successfully!")
    logger.info("="*60)

if __name__ == "__main__":
    main()
