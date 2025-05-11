import math
import sys
# import builtins # No longer needed
from .common_utils import DEFAULT_CONFIG, _get_extrema

def is_triple_top(ohlc_data, config=None):
    cfg = config if config else DEFAULT_CONFIG.copy()
    n = len(ohlc_data)
    if n < cfg["min_total_pattern_duration"]:
        return False, None

    # print(f"  is_triple_top: Started for {n} candles.") 
    # sys.stdout.flush()

    highs = [c['high'] for c in ohlc_data]
    lows = [c['low'] for c in ohlc_data]
    closes = [c['close'] for c in ohlc_data]

    peak_extrema = _get_extrema(highs, cfg["peak_trough_window"], is_peak_detector=True)
    trough_extrema = _get_extrema(lows, cfg["peak_trough_window"], is_peak_detector=False)

    # print(f"  is_triple_top: Found {len(peak_extrema)} peaks, {len(trough_extrema)} troughs.") 
    # sys.stdout.flush()
    # p3_count = 0

    try:
        for p3_item in peak_extrema:
            # p3_count += 1
            # if p3_count % 100 == 0 : 
            #     print(f"    is_triple_top: Processing p3 peak {p3_count}/{len(peak_extrema)}, index: {p3_item['index']}")
            #     sys.stdout.flush()

            idx_p3 = p3_item['index']
            p3_price = p3_item['price']

            for nl2_item in trough_extrema:
                idx_nl2 = nl2_item['index']
                nl2_price = nl2_item['price']

                if not (idx_nl2 < idx_p3): continue
                if nl2_price >= p3_price * (1 - cfg["min_peak_neckline_drop_ratio"]): continue

                for p2_item in peak_extrema:
                    idx_p2 = p2_item['index']
                    p2_price = p2_item['price']

                    if not (idx_p2 < idx_nl2): continue
                    if idx_p2 == idx_p3 : continue
                    if nl2_price >= p2_price * (1 - cfg["min_peak_neckline_drop_ratio"]): continue
                    
                    p2_p3_separation = idx_p3 - idx_p2
                    if not (cfg["min_tops_bottoms_separation"] <= p2_p3_separation <= cfg["max_tops_bottoms_separation"]):
                        continue
                    
                    for nl1_item in trough_extrema:
                        idx_nl1 = nl1_item['index']
                        nl1_price = nl1_item['price']

                        if not (idx_nl1 < idx_p2): continue
                        if idx_nl1 == idx_nl2 : continue
                        if nl1_price >= p2_price * (1 - cfg["min_peak_neckline_drop_ratio"]): continue

                        for p1_item in peak_extrema:
                            idx_p1 = p1_item['index']
                            p1_price = p1_item['price']

                            if not (idx_p1 < idx_nl1): continue
                            if idx_p1 == idx_p2 or idx_p1 == idx_p3 : continue
                            if nl1_price >= p1_price * (1 - cfg["min_peak_neckline_drop_ratio"]): continue

                            p1_p2_separation = idx_p2 - idx_p1
                            if not (cfg["min_tops_bottoms_separation"] <= p1_p2_separation <= cfg["max_tops_bottoms_separation"]):
                                continue

                            overall_neckline_price = min(nl1_price, nl2_price)
                            
                            avg_3peaks_price = (p1_price + p2_price + p3_price) / 3
                            if avg_3peaks_price == 0: continue
                            if abs(p1_price - p2_price) > avg_3peaks_price * cfg["price_proximity_ratio"] or \
                               abs(p2_price - p3_price) > avg_3peaks_price * cfg["price_proximity_ratio"] or \
                               abs(p1_price - p3_price) > avg_3peaks_price * cfg["price_proximity_ratio"] * 1.5:
                                continue
                            
                            if p2_price > p1_price * (1 + cfg["second_peak_max_higher_ratio"]) or \
                               p3_price > p1_price * (1 + cfg["third_peak_max_higher_ratio"]): 
                                 continue
                            if p3_price > p2_price * (1 + cfg["second_peak_max_higher_ratio"]):
                                 continue

                            for idx_b in range(idx_p3 + 1, n):
                                total_duration = idx_b - idx_p1 + 1
                                if total_duration > cfg["max_total_pattern_duration"]: break
                                if total_duration < cfg["min_total_pattern_duration"]: continue
                                
                                premature_break = False
                                for k_chk in range(idx_p3 + 1, idx_b):
                                    if lows[k_chk] < overall_neckline_price * (1 - cfg["neckline_break_confirm_ratio"] / 2):
                                        premature_break = True; break
                                if premature_break: continue

                                if closes[idx_b] < overall_neckline_price * (1 - cfg["neckline_break_confirm_ratio"]):
                                    details = {
                                        "pattern_name": "Triple Top",
                                        "p1": {"index": idx_p1, "price": p1_price, "timestamp": ohlc_data[idx_p1].get('timestamp')},
                                        "nl1": {"index": idx_nl1, "price": nl1_price, "timestamp": ohlc_data[idx_nl1].get('timestamp')},
                                        "p2": {"index": idx_p2, "price": p2_price, "timestamp": ohlc_data[idx_p2].get('timestamp')},
                                        "nl2": {"index": idx_nl2, "price": nl2_price, "timestamp": ohlc_data[idx_nl2].get('timestamp')},
                                        "p3": {"index": idx_p3, "price": p3_price, "timestamp": ohlc_data[idx_p3].get('timestamp')},
                                        "breakout": {"index": idx_b, "price": closes[idx_b], "timestamp": ohlc_data[idx_b].get('timestamp')},
                                        "entry_price_target": overall_neckline_price,
                                        "ohlc_data_length": n
                                    }
                                    # print(f"  is_triple_top: Found pattern ending at breakout index {idx_b}.") 
                                    # sys.stdout.flush()
                                    return True, details
    except Exception as e_triple_top:
        print(f"!!! EXCEPTION in is_triple_top: {e_triple_top}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()

    # print(f"  is_triple_top: Finished processing all candidates, no pattern found.") 
    # sys.stdout.flush()
    return False, None 