import math
import sys
# import builtins # No longer needed
from .common_utils import DEFAULT_CONFIG, _get_extrema

def is_double_top(ohlc_data, config=None):
    cfg = config if config else DEFAULT_CONFIG.copy()
    n = len(ohlc_data)
    if n < cfg["min_total_pattern_duration"]:
        return False, None

    highs = [c['high'] for c in ohlc_data]
    lows = [c['low'] for c in ohlc_data]
    closes = [c['close'] for c in ohlc_data]

    peak_extrema = _get_extrema(highs, cfg["peak_trough_window"], is_peak_detector=True)
    trough_extrema = _get_extrema(lows, cfg["peak_trough_window"], is_peak_detector=False)

    for p2_item in peak_extrema:
        idx_p2 = p2_item['index']
        p2_price = p2_item['price']

        for nl_item in trough_extrema:
            idx_nl = nl_item['index']
            nl_price = nl_item['price']

            if not (idx_nl < idx_p2): continue
            if nl_price >= p2_price * (1 - cfg["min_peak_neckline_drop_ratio"]): continue

            for p1_item in peak_extrema:
                idx_p1 = p1_item['index']
                p1_price = p1_item['price']

                if not (idx_p1 < idx_nl): continue
                if nl_price >= p1_price * (1 - cfg["min_peak_neckline_drop_ratio"]): continue
                
                tops_separation = idx_p2 - idx_p1
                if not (cfg["min_tops_bottoms_separation"] <= tops_separation <= cfg["max_tops_bottoms_separation"]):
                    continue

                avg_peak_price = (p1_price + p2_price) / 2
                if avg_peak_price == 0: continue
                if abs(p1_price - p2_price) > avg_peak_price * cfg["price_proximity_ratio"]:
                    continue
                
                if p2_price > p1_price * (1 + cfg["second_peak_max_higher_ratio"]):
                    continue

                for idx_b in range(idx_p2 + 1, n):
                    total_duration = idx_b - idx_p1 + 1
                    if total_duration > cfg["max_total_pattern_duration"]: break 
                    if total_duration < cfg["min_total_pattern_duration"]:
                        continue

                    premature_break = False
                    for k_chk in range(idx_p2 + 1, idx_b):
                        if lows[k_chk] < nl_price * (1 - cfg["neckline_break_confirm_ratio"] / 2):
                            premature_break = True; break
                    if premature_break: continue

                    if closes[idx_b] < nl_price * (1 - cfg["neckline_break_confirm_ratio"]):
                        details = {
                            "pattern_name": "Double Top",
                            "p1": {"index": idx_p1, "price": p1_price, "timestamp": ohlc_data[idx_p1].get('timestamp')},
                            "nl": {"index": idx_nl, "price": nl_price, "timestamp": ohlc_data[idx_nl].get('timestamp')},
                            "p2": {"index": idx_p2, "price": p2_price, "timestamp": ohlc_data[idx_p2].get('timestamp')},
                            "breakout": {"index": idx_b, "price": closes[idx_b], "timestamp": ohlc_data[idx_b].get('timestamp')},
                            "entry_price_target": nl_price,
                            "ohlc_data_length": n
                        }
                        return True, details
    return False, None 