from candlestick.patterns.candlestick_finder import CandlestickFinder


class Star(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 2, target=target)

    def logic(self, idx):
        # Check if index is large enough for lookback
        if idx < 1: # Requires idx >= 1 for idx-1 access
            return False
            
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx - 1]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]
        prev_high = prev_candle[self.high_column]
        prev_low = prev_candle[self.low_column]

        # Check for zero range candles
        if high - low == 0 or prev_high - prev_low == 0:
            return False

        return (prev_close > prev_open and
                abs(prev_close - prev_open) / (prev_high - prev_low) >= 0.7 and
                0.3 > abs(close - open) / (high - low) >= 0.1 and
                prev_close < close and
                prev_close < open)
