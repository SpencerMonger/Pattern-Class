from candlestick.patterns.candlestick_finder import CandlestickFinder
import sys # For sys.stdout.flush()


class DoubleBottom(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 2, target=target)

    def logic(self, idx):
        # --- DEBUG PRINT --- (Removing this)
        # print(f"  DoubleBottom.logic(idx={idx}) called. self.data length: {len(self.data)}, current TS: {self.data.index[idx] if idx < len(self.data.index) else 'idx out of bounds for index'}")
        # sys.stdout.flush()
        # --- END DEBUG PRINT ---

        if idx < 1:
            return False
            
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx - 1]

        open = candle[self.open_column]
        prev_close = prev_candle[self.close_column]
        
        # --- Original Bullish Harami Logic (commented out) --- 
        # close = candle[self.close_column]
        # high = candle[self.high_column]
        # low = candle[self.low_column]
        # prev_open = prev_candle[self.open_column]
        # prev_high = prev_candle[self.high_column]
        # prev_low = prev_candle[self.low_column]
        
        # return (
        #     prev_open > prev_close and
        #     prev_close <= open < close <= prev_open and
        #     (close - open) < (prev_open - prev_close)
        # )

        return open > prev_close # New simple logic for testing 