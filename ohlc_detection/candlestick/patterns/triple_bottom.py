import math
import sys
# import builtins # No longer needed
from .common_utils import DEFAULT_CONFIG, _get_extrema

def is_triple_bottom(ohlc_data, config=None):
    cfg = config if config else DEFAULT_CONFIG.copy()
    n = len(ohlc_data)
    if n < cfg["min_total_pattern_duration"]:
        return False, None

    highs = [c['high'] for c in ohlc_data]
    lows = [c['low'] for c in ohlc_data]
    closes = [c['close'] for c in ohlc_data]

    trough_extrema = _get_extrema(lows, cfg["peak_trough_window"], is_peak_detector=False)
    peak_extrema = _get_extrema(highs, cfg["peak_trough_window"], is_peak_detector=True)

    try:
        for b3_item in trough_extrema: 
            idx_b3 = b3_item['index']
            b3_price = b3_item['price'] # Simplified

            for nl2_item in peak_extrema:
                idx_nl2 = nl2_item['index']
                nl2_price = nl2_item['price'] # Simplified

                if not (idx_nl2 < idx_b3): continue
                if nl2_price <= b3_price * (1 + cfg["min_bottom_neckline_rise_ratio"]): continue

                for b2_item in trough_extrema:
                    idx_b2 = b2_item['index']
                    b2_price = b2_item['price'] # Simplified

                    if not (idx_b2 < idx_nl2): continue
                    if idx_b2 == idx_b3 : continue
                    if nl2_price <= b2_price * (1 + cfg["min_bottom_neckline_rise_ratio"]): continue
                    
                    b2_b3_separation = idx_b3 - idx_b2
                    if not (cfg["min_tops_bottoms_separation"] <= b2_b3_separation <= cfg["max_tops_bottoms_separation"]):
                        continue

                    for nl1_item in peak_extrema:
                        idx_nl1 = nl1_item['index']
                        nl1_price = nl1_item['price'] # Simplified

                        if not (idx_nl1 < idx_b2): continue
                        if idx_nl1 == idx_nl2 : continue
                        if nl1_price <= b2_price * (1 + cfg["min_bottom_neckline_rise_ratio"]): continue
                        
                        for b1_item in trough_extrema:
                            idx_b1 = b1_item['index']
                            b1_price = b1_item['price'] # Simplified

                            if not (idx_b1 < idx_nl1): continue
                            if idx_b1 == idx_b2 or idx_b1 == idx_b3 : continue
                            if nl1_price <= b1_price * (1 + cfg["min_bottom_neckline_rise_ratio"]): continue

                            b1_b2_separation = idx_b2 - idx_b1
                            if not (cfg["min_tops_bottoms_separation"] <= b1_b2_separation <= cfg["max_tops_bottoms_separation"]):
                                continue

                            overall_neckline_price = max(nl1_price, nl2_price)
                            
                            avg_3bottoms_price = (b1_price + b2_price + b3_price) / 3
                            if avg_3bottoms_price == 0: continue
                            if abs(b1_price - b2_price) > avg_3bottoms_price * cfg["price_proximity_ratio"] or \
                               abs(b2_price - b3_price) > avg_3bottoms_price * cfg["price_proximity_ratio"] or \
                               abs(b1_price - b3_price) > avg_3bottoms_price * cfg["price_proximity_ratio"] * 1.5:
                                continue
                            
                            if b2_price < b1_price * (1 - cfg["second_peak_max_higher_ratio"]) or \
                               b3_price < b1_price * (1 - cfg["third_peak_max_higher_ratio"]):
                                continue
                            if b3_price < b2_price * (1 - cfg["second_peak_max_higher_ratio"]):
                                continue

                            for idx_b_break in range(idx_b3 + 1, n):
                                total_duration = idx_b_break - idx_b1 + 1
                                if total_duration > cfg["max_total_pattern_duration"]: break
                                if total_duration < cfg["min_total_pattern_duration"]: continue
                                
                                premature_break = False
                                for k_chk in range(idx_b3 + 1, idx_b_break):
                                    if highs[k_chk] > overall_neckline_price * (1 + cfg["neckline_break_confirm_ratio"] / 2):
                                        premature_break = True; break
                                if premature_break: continue

                                if closes[idx_b_break] > overall_neckline_price * (1 + cfg["neckline_break_confirm_ratio"]):
                                    details = {
                                        "pattern_name": "Triple Bottom",
                                        "b1": {"index": idx_b1, "price": b1_price, "timestamp": ohlc_data[idx_b1].get('timestamp')},
                                        "nl1": {"index": idx_nl1, "price": nl1_price, "timestamp": ohlc_data[idx_nl1].get('timestamp')},
                                        "b2": {"index": idx_b2, "price": b2_price, "timestamp": ohlc_data[idx_b2].get('timestamp')},
                                        "nl2": {"index": idx_nl2, "price": nl2_price, "timestamp": ohlc_data[idx_nl2].get('timestamp')},
                                        "b3": {"index": idx_b3, "price": b3_price, "timestamp": ohlc_data[idx_b3].get('timestamp')},
                                        "breakout": {"index": idx_b_break, "price": closes[idx_b_break], "timestamp": ohlc_data[idx_b_break].get('timestamp')},
                                        "entry_price_target": overall_neckline_price,
                                        "ohlc_data_length": n
                                    }
                                    return True, details
    except Exception as e_triple_bottom:
        print(f"!!! EXCEPTION in is_triple_bottom: {e_triple_bottom}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        
    return False, None 