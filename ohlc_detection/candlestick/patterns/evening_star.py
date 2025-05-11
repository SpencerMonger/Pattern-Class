from candlestick.patterns.candlestick_finder import CandlestickFinder


class EveningStar(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 3, target=target)

    def logic(self, idx):
        # Check if index is large enough for lookback
        if idx < 2: # Requires idx >= 2 for idx-1 and idx-2 access
            return False
            
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx - 1]
        b_prev_candle = self.data.iloc[idx - 2]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]
        prev_high = prev_candle[self.high_column]
        prev_low = prev_candle[self.low_column]

        b_prev_close = b_prev_candle[self.close_column]
        b_prev_open = b_prev_candle[self.open_column]
        b_prev_high = b_prev_candle[self.high_column]
        b_prev_low = b_prev_candle[self.low_column]

        # Check for zero range candles to avoid potential issues if logic uses division (even if commented out)
        if high - low == 0 or prev_high - prev_low == 0 or b_prev_high - b_prev_low == 0:
             return False

        # return (b_prev_close > b_prev_open and
        #         abs(b_prev_close - b_prev_open) / (b_prev_high - b_prev_low) >= 0.7 and
        #         0.3 > abs(prev_close - prev_open) / (prev_high - prev_low) >= 0.1 and
        #         close < open and
        #         abs(close - open) / (high - low) >= 0.7 and
        #         b_prev_close < prev_close and
        #         b_prev_close < prev_open and
        #         prev_close > open and
        #         prev_open > open and
        #         close < b_prev_close)

        return (min(prev_open, prev_close) > b_prev_close > b_prev_open and
                close < open < min(prev_open, prev_close))
