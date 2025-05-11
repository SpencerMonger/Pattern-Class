import math
import sys

DEFAULT_CONFIG = {
    "peak_trough_window": 10,  # Window size (W) for peak/trough detection
    "min_tops_bottoms_separation": 5,
    "max_tops_bottoms_separation": 40,
    "price_proximity_ratio": 0.03,
    "second_peak_max_higher_ratio": 0.015,
    "third_peak_max_higher_ratio": 0.02,
    "neckline_break_confirm_ratio": 0.002,
    "max_total_pattern_duration": 75,
    "min_total_pattern_duration": 10,
    "min_peak_neckline_drop_ratio": 0.01,
    "min_bottom_neckline_rise_ratio": 0.01
}

def _get_extrema(prices, W, is_peak_detector):
    """
    Finds local extrema (peaks or troughs) in a list of prices.
    An extremum at index i is found if prices[i] is the highest/lowest
    in the window prices[i-W:i+W+1], taking the first occurrence in case of a plateau.
    """
    extrema = []
    n = len(prices)
    if n == 0:
        return extrema

    # print(f"    _get_extrema: Processing {n} prices with W={W}, is_peak={is_peak_detector}") 
    # sys.stdout.flush()

    temp_extrema_indices = [] 

    for i in range(n):
        current_price = prices[i]
        is_extremum_in_window = True
        
        window_start_idx = max(0, i - W)
        window_end_idx = min(n - 1, i + W)

        for j in range(window_start_idx, window_end_idx + 1):
            if i == j:
                continue
            
            if is_peak_detector:
                if prices[j] > current_price:
                    is_extremum_in_window = False
                    break
                if prices[j] == current_price and j < i: 
                    is_extremum_in_window = False
                    break
            else: 
                if prices[j] < current_price:
                    is_extremum_in_window = False
                    break
                if prices[j] == current_price and j < i: 
                    is_extremum_in_window = False
                    break
        
        if is_extremum_in_window:
            is_turn = False # This logic was not strictly necessary for the original definition based on window
            if n == 1: is_turn = True
            elif i == 0: 
                if is_peak_detector: is_turn = current_price > prices[i+1] if n > 1 else True
                else: is_turn = current_price < prices[i+1] if n > 1 else True
            elif i == n - 1: 
                if is_peak_detector: is_turn = current_price > prices[i-1]
                else: is_turn = current_price < prices[i-1]
            else: 
                if is_peak_detector: is_turn = (current_price > prices[i-1] or current_price > prices[i+1])
                else: is_turn = (current_price < prices[i-1] or current_price < prices[i+1])
            
            # --- Robust price processing --- Start
            current_price_from_list = prices[i]
            processed_price = None
            try:
                processed_price = float(current_price_from_list)
            except TypeError:
                err_price_type = type(current_price_from_list).__name__
                if hasattr(current_price_from_list, '__next__'): # Is it an iterator?
                    try:
                        processed_price = float(next(current_price_from_list))
                    except (TypeError, StopIteration):
                        print(f"Error in _get_extrema: Price at index {i} (type {err_price_type}) is an iterator but couldn't resolve to float. Skipping point.")
                        continue # Skip this extremum point
                elif isinstance(current_price_from_list, list) and len(current_price_from_list) == 1:
                    try:
                        processed_price = float(current_price_from_list[0])
                    except (TypeError, ValueError):
                        print(f"Error in _get_extrema: Price at index {i} (type {err_price_type}) is a list but content couldn't resolve to float. Skipping point.")
                        continue # Skip this extremum point
                else:
                    print(f"Error in _get_extrema: Price at index {i} has unhandled type {err_price_type}. Value: {str(current_price_from_list)[:100]}. Skipping point.")
                    continue # Skip this extremum point
            except ValueError:
                 print(f"Error in _get_extrema: Price at index {i} could not be converted to float by ValueError. Value: {str(current_price_from_list)[:100]}. Skipping point.")
                 continue
            # --- Robust price processing --- End

            # --- Add detailed debug print before append --- Start
            # print(f"  _get_extrema INTERNAL DEBUG: index={i}, type(processed_price)={type(processed_price)}, value={processed_price}")
            # sys.stdout.flush()
            # --- Add detailed debug print before append --- End
            extrema.append({'index': i, 'price': processed_price})
            temp_extrema_indices.append(i)
            
    # print(f"    _get_extrema: Found {len(extrema)} extrema. Indices: {temp_extrema_indices[:10]}...{temp_extrema_indices[-10:] if len(temp_extrema_indices) > 20 else ''}")
    # sys.stdout.flush()
    return extrema 