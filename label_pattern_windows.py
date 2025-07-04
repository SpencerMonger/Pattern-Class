import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# --- Configuration ---
OHLC_FILE_PATH = "training-data/ohlc.csv"
OUTPUT_LABEL_FILE = "training-data/pattern_labels.csv"
WINDOW_SIZE = 15 # Define the pattern window size in minutes (rows)

# Placeholder: Need to import the actual pattern finding function
# from candlestick.patterns.candlestick_finder import find_all_patterns # <<< ADJUST IMPORT PATH
# Placeholder: Need the list of pattern column names generated by the finder
# pattern_column_names = ['CDLDOJI', 'CDLHAMMER', ...] # <<< GET ACTUAL LIST

# Placeholder: Mapping from finder's column names to desired label strings
# pattern_mapping = { # <<< DEFINE ACTUAL MAPPINGS
#     'CDLDOJI': 'doji',
#     'CDLHAMMER': 'hammer',
#     'CDLBULLISHENGULFING': 'bullish_engulfing',
#     # ... add all patterns ...
# }


def label_windows_with_patterns(df_ohlc, window_size=WINDOW_SIZE):
    """ 
    Applies pattern detection and assigns a single label to each window.

    Args:
        df_ohlc (pd.DataFrame): DataFrame with OHLC data and datetime index.
        window_size (int): The number of rows (minutes) for the pattern window.

    Returns:
        pd.Series: A Series mapping window end timestamps to pattern label strings.
    """
    print(f"Running candlestick pattern detection (using MOCK data for now)...")
    # --- Step 1: Run the existing pattern finder --- 
    # This function should take df_ohlc and return a df with added pattern columns
    # df_with_patterns = find_all_patterns(df_ohlc.copy()) # <<< REPLACE MOCK WITH THIS

    # --- MOCK IMPLEMENTATION (Remove when integrating actual finder) ---
    df_with_patterns = df_ohlc.copy()
    # Example: Using dummy pattern columns and mapping for demonstration
    pattern_column_names = ['MOCK_CDLDOJI', 'MOCK_CDLHAMMER', 'MOCK_CDLBULLISHENGULFING']
    pattern_mapping = { 
        'MOCK_CDLDOJI': 'doji',
        'MOCK_CDLHAMMER': 'hammer',
        'MOCK_CDLBULLISHENGULFING': 'bullish_engulfing',
    }
    print(f"Using MOCK patterns: {list(pattern_mapping.values())}")
    for col in pattern_column_names:
         df_with_patterns[col] = np.random.choice([0, 100], size=len(df_with_patterns), p=[0.99, 0.01])
    print("Mock pattern detection complete.")
    # --- End MOCK --- 

    print(f"Labeling {window_size}-minute windows based on detected patterns...")
    window_labels = {} # Dictionary to store {end_timestamp: label_string}

    # Ensure the DataFrame index is sorted
    if not df_with_patterns.index.is_monotonic_increasing:
        print("Warning: DataFrame index is not sorted. Sorting...")
        df_with_patterns.sort_index(inplace=True)

    if len(df_with_patterns) < window_size:
         print(f"Error: Data length ({len(df_with_patterns)}) is less than window size ({window_size}). Cannot create labels.")
         return pd.Series(dtype=str)

    # Iterate through possible window end points
    for i in tqdm(range(window_size - 1, len(df_with_patterns))):
        window_start_index = i - window_size + 1
        window_end_index = i
        window_end_time = df_with_patterns.index[window_end_index]
        window_slice = df_with_patterns.iloc[window_start_index : window_end_index + 1]

        last_pattern_label = 'no_pattern' # Default label

        # Check pattern columns within the window slice 
        patterns_in_window = window_slice[pattern_column_names]

        # Find the index and column name of the *last* detected pattern signal in the window
        found_pattern = False
        # Iterate backwards through the window's rows
        for row_idx in range(len(patterns_in_window) - 1, -1, -1):
            row_timestamp = patterns_in_window.index[row_idx]
            row_signals = patterns_in_window.loc[row_timestamp]
            
            # Check for non-zero signals in this row
            detected_pattern_cols = row_signals[row_signals != 0].index
            
            if not detected_pattern_cols.empty:
                # Prioritize or just take the first detected column name if multiple occur simultaneously
                pattern_col = detected_pattern_cols[0] 
                if pattern_col in pattern_mapping:
                    last_pattern_label = pattern_mapping[pattern_col]
                    found_pattern = True
                    break # Stop searching once the latest pattern in the window is found
            
        # Assign the determined label (could be 'no_pattern' or a specific pattern)
        window_labels[window_end_time] = last_pattern_label

    print(f"Generated labels for {len(window_labels)} windows.")
    if not window_labels:
        print("Warning: No labels were generated.")
        return pd.Series(dtype=str)
        
    final_labels = pd.Series(window_labels, name="pattern_label")
    print("Label distribution:")
    print(final_labels.value_counts())
    
    return final_labels

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting pattern labeling process...")
    print(f"Input OHLC file: {OHLC_FILE_PATH}")
    print(f"Output label file: {OUTPUT_LABEL_FILE}")
    print(f"Window size: {WINDOW_SIZE} minutes")

    try:
        # 1. Load Data
        ohlc_df = pd.read_csv(OHLC_FILE_PATH, index_col=0, parse_dates=True)
        print(f"Loaded {len(ohlc_df)} rows from {OHLC_FILE_PATH}")
        ohlc_df.sort_index(inplace=True) # Ensure data is sorted

        # 2. Generate Labels
        pattern_labels = label_windows_with_patterns(ohlc_df, window_size=WINDOW_SIZE)

        # 3. Save Labels
        if not pattern_labels.empty:
            pattern_labels.to_csv(OUTPUT_LABEL_FILE, header=True)
            print(f"Successfully saved labels to {OUTPUT_LABEL_FILE}")
        else:
            print("Skipping saving as no labels were generated.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {OHLC_FILE_PATH}")
    except Exception as e:
        print(f"An error occurred during the labeling process: {e}")
        import traceback
        traceback.print_exc()

    print("Labeling process finished.") 