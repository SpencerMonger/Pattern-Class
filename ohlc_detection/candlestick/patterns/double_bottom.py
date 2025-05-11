from candlestick.patterns.candlestick_finder import CandlestickFinder
import pandas as pd
import numpy as np # For NaN checks if any

class DoubleBottom(CandlestickFinder):
    def __init__(self, target=None):
        self.pattern_name = self.get_class_name()

        # --- Configuration for "W" Shape Double Bottom ---
        # B1: First significant low.
        # N:  Neckline peak after B1.
        # B2: Second significant low (retesting B1's level) after N.
        # Current candle 'idx' is the potential B2.

        self.b1_low_lookback = 15             # B1's low must be lower than preceding N candles.
        
        self.min_b1_to_n_len = 2              # Min candles from B1 (exclusive) to N (inclusive).
        self.max_b1_to_n_len = 10             # Max candles from B1 (exclusive) to N (inclusive).
        self.neckline_min_prominence_vs_b1 = 0.005 # Neckline peak (high_N) must be at least 0.5% > low_B1.

        self.min_n_to_b2_len = 2              # Min candles from N (exclusive) to B2 (inclusive = current idx).
        self.max_n_to_b2_len = 10             # Max candles from N (exclusive) to B2 (inclusive = current idx).
        self.b2_min_drop_from_n = 0.005       # B2's low must be at least 0.5% < high_N.
        
        self.retest_proximity_max_above_b1 = 0.05 # B2's low can be 0 to 0.05 (5 cents) *above* B1's low. Not below.

        self.min_total_pattern_len = 5        # Min candles from B1 (inclusive) to B2 (inclusive = current idx).
        self.max_total_pattern_len = 30       # Max candles from B1 (inclusive) to B2 (inclusive = current idx).

        # numbers_req: Max candles *prior* to current candle (idx) needed by the logic.
        # Max (B1-N len + N-B2 len) for structure, then + b1_low_lookback for B1 validation.
        # Max total pattern length already considers B1-N and N-B2 structure.
        # So, it's max_total_pattern_len (to find B1 relative to B2) + b1_low_lookback (to validate that B1).
        # Note: self.data access uses iloc, so indices are 0 to len-1.
        # 'idx' is the current candle index. B1 is at 'idx - total_len_from_b1_to_b2 + 1'.
        # The earliest candle needed for B1's lookback is 'idx_B1 - b1_low_lookback'.
        required_prior_candles = (self.max_total_pattern_len -1) + self.b1_low_lookback
        super().__init__(self.pattern_name, required_prior_candles, target=target)

    def logic(self, idx):
        # idx is the current candle, which is the potential second bottom (B2).

        if idx > 0 and idx % 2000 == 0: # Print progress every 2000 candles for this ticker
            try:
                total_candles_for_ticker = len(self.data)
                print(f"    DB W-Shape ({self.pattern_name}): Processing candle {idx}/{total_candles_for_ticker}...", flush=True)
            except AttributeError:
                print(f"    DB W-Shape ({self.pattern_name}): Processing candle {idx} (total N/A)...", flush=True)

        current_low_b2 = self.data[self.low_column].iloc[idx]

        # Iterate for overall pattern length (from B1 to B2 inclusive)
        for k_total_len in range(self.min_total_pattern_len, self.max_total_pattern_len + 1):
            idx_B1 = idx - k_total_len + 1
            if idx_B1 < self.b1_low_lookback: # Ensure enough data for B1's own lookback
                continue
            
            low_B1 = self.data[self.low_column].iloc[idx_B1]

            # Condition 0: B1 must be a 'b1_low_lookback'-minute low.
            is_B1_significant = False
            if self.b1_low_lookback > 0:
                start_lookback_B1 = idx_B1 - self.b1_low_lookback
                preceding_lows_B1 = self.data[self.low_column].iloc[start_lookback_B1:idx_B1]
                if not preceding_lows_B1.empty and low_B1 < preceding_lows_B1.min():
                    is_B1_significant = True
            else: # Should not happen if b1_low_lookback > 0
                is_B1_significant = True 
            
            if not is_B1_significant: continue

            # Condition for B2 retest (relative to B1)
            if not ((current_low_b2 >= low_B1) and ((current_low_b2 - low_B1) <= self.retest_proximity_max_above_b1)):
                continue

            # Iterate for N-B2 length (from N exclusive to B2 inclusive)
            # k_n_b2_len includes B2, does not include N.
            for k_n_b2_len in range(self.min_n_to_b2_len, self.max_n_to_b2_len + 1):
                if k_n_b2_len >= k_total_len: # N-B2 segment cannot be longer than total pattern
                    continue
                
                idx_N = idx - k_n_b2_len # idx_N is the candle *before* the N-B2 segment starts
                if idx_N < idx_B1 : # Neckline N must be after B1
                    continue
                
                # Segment for N-B2 decline (N is exclusive, B2 is inclusive)
                # Candles from (idx_N + 1) up to 'idx' (which is B2)
                # The peak N should be at idx_N or within the B1_to_N segment.
                # Let's find the B1_to_N segment first.
                
                # Length of B1-to-N segment (B1 inclusive, N inclusive)
                # k_total_len = (idx - idx_B1 + 1)
                # k_b1_n_len_inclusive = (idx_N - idx_B1 + 1)
                k_b1_n_len_inclusive = k_total_len - k_n_b2_len

                if not (self.min_b1_to_n_len <= k_b1_n_len_inclusive <= self.max_b1_to_n_len):
                    continue

                # Extract segment from B1 (inclusive) to N (inclusive)
                segment_b1_to_n = self.data.iloc[idx_B1 : idx_N + 1] # idx_N is inclusive here
                if segment_b1_to_n.empty:
                    continue
                
                high_N = segment_b1_to_n[self.high_column].max()
                # Optional: Find index of high_N if needed for stricter N definition (e.g. not B1 itself)
                # For now, just taking max high in the segment B1...idx_N.

                # Condition: Neckline prominence vs B1
                if not (high_N > low_B1 * (1 + self.neckline_min_prominence_vs_b1)):
                    continue
                
                # Condition: B2's low must be significantly lower than Neckline N's high
                if not (current_low_b2 < high_N * (1 - self.b2_min_drop_from_n)):
                    continue

                # All W-shape conditions met
                return True
        
        return False 