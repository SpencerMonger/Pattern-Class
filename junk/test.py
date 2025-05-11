from flask import Flask, request, jsonify
import requests
import datetime
import pandas as pd
from datetime import datetime, timedelta
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import pytz 
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS
from candlestick import candlestick
from pattern_pred import get_pattern
app = Flask(__name__)
CORS(app) 

def get_stock_data(SYMB, Interval, Range):
    api_url = f"https://api.twelvedata.com/time_series?apikey=7a8e68b26a9d472180f530cbfe1b7dd8&interval={Interval}min&symbol={SYMB}&start_date={(datetime.now() - timedelta(days=Range)).strftime('%Y-%m-%dT%H:%M:%S.%f')}&format=JSON&timezone=US/Eastern"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        # print(data)
        time_series = data.get('values', [])
        df = pd.DataFrame(time_series)
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df.iloc[::-1].reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.loc[:, df.columns != 'Date'] = df.loc[:, df.columns != 'Date'].apply(pd.to_numeric, errors='coerce')
        df['Range'] = df['High'] - df['Low']     # High-low
        df['Change'] = df['Close'].diff()  # Calculates the price change between current and previous row's close price
        df['Typical Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        # if df['Volume'].cumsum() != 0:
        df['VWAP'] = (df['Typical Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = 100 - (100 / (1 + df['Close'].diff(1).apply(lambda x: x if x > 0 else 0).rolling(14).sum() / df['Close'].diff(1).apply(lambda x: abs(x) if x < 0 else 0).rolling(14).sum()))
        df['dt'] = pd.to_datetime(df['Date'])
        df['dt'] = df['dt'].dt.date
        last_date = df['dt'].max()
        today_data = df[df['dt'] == last_date]
        df["Today's high"]=today_data['Close'].max()
        df["Today's low"]=today_data['Close'].min()
        df.drop(columns="dt", inplace=True)
        df.dropna(inplace=True)
        # print(df.tail())
        return df
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")

def next_candle(SYMB, Interval, Range):
    start_time = time.time()

    df = get_stock_data(SYMB,Interval,Range)
    data = df[['Open', 'High', 'Low', 'Close','Volume', 'Range', 'Change', 'VWAP', 'SMA20','RSI',"Today's high", "Today's low"]].values
    print(df)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)
    time_steps = 7

    X, y = [], []
    for i in range(len(data_normalized) - time_steps):
        X.append(data_normalized[i:i+time_steps])
        y.append(data_normalized[i+time_steps])

    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_steps, 12)))
    model.add(LSTM(50))
    model.add(Dense(12))  # 4 output nodes for predicting 'Open', 'High', 'Low', 'Close'

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    y_pred = model.predict(X_test)

    # Accuracy
    y_pred_original = scaler.inverse_transform(y_pred)
    y_test_original = scaler.inverse_transform(y_test)
    
    def calculate_accuracy(actual, predicted):  
        actual = np.array(actual)
        predicted = np.array(predicted)
        accuracy = abs(100 - abs(actual - predicted) * 100)
        mask = np.abs(actual - predicted) <= 1
        accuracies = np.where(mask, accuracy, 0)
        mean_accuracy = np.mean(accuracies)
        return mean_accuracy

    # def mape(y_true, y_pred):
    #     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    accuracy = calculate_accuracy(y_test_original, y_pred_original)

    # Prediction
    last_time_step_data = data_normalized[-time_steps:]
    next_future_value = model.predict(np.array([last_time_step_data]))
    next_future_value = scaler.inverse_transform(next_future_value)

    df = df[['Date','Open','High','Low','Close', 'Volume']]
    df['Date'] = pd.to_datetime(df['Date'])
    curr_time = df['Date'].iloc[-1] + timedelta(minutes=int(Interval))
    target = pd.DataFrame(next_future_value, columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Range', 'Change', 'VWAP', 'SMA20','RSI',"Today's high", "Today's low"])
    target = candlestick.dragonfly_doji(target, target="Bullish_doji")
    target = candlestick.gravestone_doji(target, target="Bearish_doji")
    target['Date'] = curr_time
    print(curr_time)
    target['Color'] = 'gray'
    target['Volume'] = 0
    target['Accuracy'] = accuracy

    end_time = time.time()
    execution_time = end_time - start_time
    return df, target

@app.route('/predict/predict', methods=['POST'])
def predict_next_candle():
    try:
        data = request.get_json()
        SYMB = data['SYMB']
        Interval = data['Interval']
        Range = data['Range']
        
        df, target = next_candle(SYMB, Interval, Range)
        target = target[['Date','Open', 'High', 'Low', 'Close', 'Volume', 'Bullish_doji', 'Bearish_doji','Accuracy', 'Color']]
        target_json = target.to_dict(orient='records')
        print('target_json')
        print(target_json)
        response = target_json
        
        return jsonify(response)
    except Exception as e:
            print(str(e))
            return str(e)

@app.route('/predict/pattern', methods=['POST'])
def predict_pattern():
    try:
        data = request.get_json()
        SYMB = data['SYMB']
        Interval = data['Interval']
        Range = data['Range']
        response = get_pattern(SYMB, Interval, Range)   
        return jsonify(response)
    except Exception as e:
            print(str(e))
            return str(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5001)