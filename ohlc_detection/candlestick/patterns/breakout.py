from candlestick.patterns.candlestick_finder import CandlestickFinder
import pandas as pd
import numpy as np

class Breakout(CandlestickFinder):
    def __init__(self, target=None):
        self.pattern_name = self.get_class_name()
        self.high_lookback_period = 30
        required_prior_candles = self.high_lookback_period
        # The second argument to super().__init__ is 'numbers_req_'
        # It also sets up self.open_column, self.high_column, etc. (defaulting to 'Open', 'High')
        super().__init__(self.pattern_name, required_prior_candles, target=target)

    def logic(self, idx):
        # self.data has a RangeIndex and CAPITALIZED columns here.
        # idx is an integer row index, likely starting from self.numbers_req_ (30 for this pattern).

        # # --- DEBUG FOR THE FIRST CALL TO logic() for each ticker (idx = 30) ---
        # if idx == self.high_lookback_period: 
        #     ticker_ctx = "UnknownTicker (self.target not set or not a string)"
        #     if hasattr(self, 'target') and isinstance(self.target, str) and self.target:
        #         ticker_ctx = self.target
            
        #     ts_at_0_str = "RangeIndex (no direct timestamp)"
        #     ts_at_idx_str = "RangeIndex (no direct timestamp)"
        #     if isinstance(self.data.index, pd.DatetimeIndex): 
        #         if len(self.data.index) > 0: ts_at_0_str = str(self.data.index[0])
        #         if idx < len(self.data.index): ts_at_idx_str = str(self.data.index[idx])
            
        #     print(f"\nDEBUG BREAKOUT.PY (TICKER: {ticker_ctx}, First call to logic(idx={idx})):")
        #     print(f"  self.data.columns: {self.data.columns.tolist()}")
        #     print(f"  self.high_column: '{self.high_column}', self.open_column: '{self.open_column}'")
        #     print(f"  self.data.index type: {type(self.data.index)}")
        #     print(f"  Number of rows in self.data: {len(self.data)}")
        #     print(f"  Timestamp of self.data.iloc[0] (via .index[0]): {ts_at_0_str}")
        #     print(f"  Timestamp of self.data.iloc[{idx}] (via .index[idx]): {ts_at_idx_str}")
        #     print(f"  First 5 rows of self.data using specific columns:\n{self.data[[self.open_column, self.high_column, self.low_column, self.close_column]].head(5).to_string()}", flush=True)
        #     context_start = max(0, idx - 2)
        #     context_end = min(len(self.data), idx + 3)
        #     print(f"  Context from self.data around idx {idx}:\n{self.data[[self.open_column, self.high_column, self.low_column, self.close_column]].iloc[context_start:context_end]}", flush=True)

        # --- Original logic ---
        start_lookback_idx = max(0, idx - self.high_lookback_period)
        preceding_highs = self.data[self.high_column].iloc[start_lookback_idx:idx]

        if preceding_highs.empty:
            return False
        
        valid_preceding_highs = preceding_highs.dropna()
        if valid_preceding_highs.empty:
            return False
            
        thirty_min_high = valid_preceding_highs.max()

        current_open = self.data[self.open_column].iloc[idx]
        current_high = self.data[self.high_column].iloc[idx]
        current_low = self.data[self.low_column].iloc[idx]
        current_close = self.data[self.close_column].iloc[idx]
        
        # if idx == self.high_lookback_period : 
        #     ticker_ctx_cont = "UnknownTarget"
        #     if hasattr(self, 'target') and isinstance(self.target, str) and self.target:
        #          ticker_ctx_cont = self.target
            
        #     if ticker_ctx_cont == 'META' or ticker_ctx_cont == "Breakout" or ticker_ctx_cont == "UnknownTarget":
        #         print(f"  (Cont.) DEBUG BREAKOUT.PY (TICKER: {ticker_ctx_cont}, idx: {idx}):")
        #         print(f"    Lookback Window (indices in self.data): {start_lookback_idx} to {idx-1}")
        #         print(f"    Calculated Peak of Preceding {len(preceding_highs)} Highs: {thirty_min_high}")
        #         print(f"    Current Candle (at idx {idx}): Open={current_open}, High={current_high}, Low={current_low}, Close={current_close}")
        #         print(f"    Breakout by High > Peak: {current_high > thirty_min_high}", flush=True)

        if (current_open > thirty_min_high or
            current_high > thirty_min_high or
            current_low > thirty_min_high or 
            current_close > thirty_min_high):
            return True
        
        return False 