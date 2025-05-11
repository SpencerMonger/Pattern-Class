import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import tensorflow as tf
import requests
from sklearn.metrics import r2_score
# from flask import Flask, request, jsonify
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app) 

def get_stock_data(SYMB, Interval, Range):
    from datetime import datetime, timedelta
    import time
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
        df.dropna(inplace=True)
        # print(df.tail())
        return df
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        
#checks if trend is double top
def isDoubleTop(sma10):
    trend = 'up'
    patternStarted = False
    lastPrice = sma10[5]
    #local extrema tuples are (Min/Max, index)
    firstLMax = secondLMax = (-1, -1)
    firstLMin = (float('inf'), -1)
    for i in range(5,len(sma10),5):
        print(i)
        trend = 'up' if sma10[i] > lastPrice else 'down'
        lastPrice = sma10[i]
        if not patternStarted:
            #trying to find first local max/resistance
            if trend=='up':
                firstLMax = (sma10[i], i)
            #check for signs of pattern starting
            elif -1 < firstLMax[1]:
                patternStarted = True
                firstLMin = (sma10[i], i)
        #check if pattern meets requirements
        else:
            #trying to find first local min/support
            if trend=='down' and secondLMax[1]<0:
                firstLMin = (sma10[i], i)
            elif trend=='up':
                #checks if broke resistance too early then break pattern
                if sma10[i]*1.10 > (firstLMax[0]):
                    firstLMax = secondLMax = (-1, -1)
                    firstLMin = (float('inf'), -1)
                    patternStarted = False
                else:
                    secondLMax = (sma10[i], i)
            elif trend=='down' and sma10[i] < firstLMin[0]:
                #meets pattern if breaks support
                return True
    return False

#test with PFE chart from 11/1/1999-05/1/200
#checks if trend is double bottom
def isDoubleBottom(sma10):
    trend = 'down'
    patternStarted = False
    lastPrice = sma10[5]
    #local extrema tuples are (Min/Max, index)
    firstLMin = secondLMin = (float('inf'), -1)
    firstLMax = (float('-inf'), -1)
    for i in range(5,len(sma10),5):
        print(i)
        trend = 'up' if sma10[i] > lastPrice else 'down'
        lastPrice = sma10[i]
        #trying to find firstLMin/support
        if not patternStarted:
            if trend=='down':
                firstLMin = (sma10[i], i)
            #check for signs of pattern starting
            elif firstLMin[1] > 0:
                patternStarted = True
                firstLMax = (sma10[i], i)
        #check if pattern meets requirements
        else:
            #try to find localMax/Resistance
            if trend=='up' and secondLMin[1]<0:
                firstLMax = (sma10[i], i)
            elif trend=='down':
                #checks if broke support too early then break pattern
                if sma10[i] < (firstLMin[0] * 0.9):
                    firstLMin = (float('inf'), -1)
                    secondLMin = (float('inf'), -1)
                    firstLMax = (float('-inf'), -1)
                    patternStarted = False
                else:
                    secondLMin = (sma10[i],i)

            #checks if second uptrend breaks resistance
            elif trend=='up' and sma10[i]>firstLMax[0]:
                #check if local mins are too far away
                print(firstLMin)
                print(firstLMax)
                print(secondLMin)
                if not (firstLMax[1] - firstLMin[1])*0.5 <= (secondLMin[1] - firstLMax[1]) <= (firstLMax[1] - firstLMin[1])*1.5:
                    firstLMin = (float('inf'), -1)
                    secondLMin = (float('inf'), -1)
                    firstLMax = (float('-inf'), -1)
                    patternStarted = False
                #found pattern
                else:
                    return True
    return False

Scale=StandardScaler()

def data_prep(df, lookback, future, Scale):
    date_train=pd.to_datetime(df['Date'])
    df_train=df[['Open','High','Low','Close','Volume']]
    df_train=df_train.astype(float)
    df_train_scaled=Scale.fit_transform(df_train)

    X, y =[],[]
    for i in range(lookback, len(df_train_scaled)-future+1):
        X.append(df_train_scaled[i-lookback:i, 0:df_train.shape[1]])
        y.append(df_train_scaled[i+future-1:i+future, 0])
        
    return np.array(X), np.array(y), df_train, date_train

def Lstm_model1(X, y):
    regressor = Sequential()

    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    regressor.fit(X, y, epochs = 10, validation_split=0.1, batch_size = 64, verbose=1, callbacks=[es])
    return regressor

def predict_open(model,df,Lstm_x,df_train, future, Scale, Interval):
    from datetime import datetime, timedelta
    start_date_timestamp = df['Date'].iloc[-1]
    start_date_timestamp = start_date_timestamp + timedelta(minutes=Interval)
    import datetime
    start_date = start_date_timestamp.to_pydatetime()
    forecasting_dates = [start_date + datetime.timedelta(minutes=i*Interval) for i in range(future)]
    predicted=model.predict(Lstm_x[-future:])
    predicted1=np.repeat(predicted, df_train.shape[1], axis=-1)
    predicted_descaled=Scale.inverse_transform(predicted1)[:,0]
    return predicted_descaled,forecasting_dates

def output_prep(forecasting_dates, predicted_descaled):
    df_final = pd.DataFrame(columns=['Date', 'Close'])
    df_final['Date'] = forecasting_dates
    df_final['Close'] = predicted_descaled
    
    return df_final

def get_pattern(SYMB, Interval, Range):
    df = get_stock_data(SYMB, Interval, Range)
    lookback = 30
    future = 1
    Lstm_x, Lstm_y, df_train, date_train = data_prep(df, lookback, future, Scale)
    model=Lstm_model1(Lstm_x,Lstm_y)
    loss=pd.DataFrame(model.history.history)
#     loss.plot()
    future=5
    predicted_descaled,forecasting_dates=predict_open(model,df,Lstm_x,df_train,future, Scale, Interval)
    results=output_prep(forecasting_dates,predicted_descaled)   
    actual_df = df[['Date','Close']]
    results = results[['Date','Close']]
    results = pd.concat([actual_df, results])
    results.reset_index(inplace=True)
    print(results)
    values = results.tail(6)
    results = results.set_index(results['Date'])
    sma10 = list(results['Close'].rolling(10).mean())
    import math
    sma10 = [x for x in sma10 if not math.isnan(x)]
    sma10 = sma10[-6:]
    DoubleBottom = isDoubleBottom(sma10)
    DoubleTop = isDoubleTop(sma10)
    predictions = model.predict(Lstm_x)
    r2 = r2_score(Lstm_y, predictions)
    if DoubleBottom:
        DoubleBottom_Accuracy = r2 * 100
    elif DoubleTop:
        DoubleTop_Accuracy = r2 * 100
    else:
        DoubleBottom_Accuracy = 0
        DoubleTop_Accuracy = 0   
    return {"Value": [[date, open] for date, open in zip(values["Date"].astype(str), values["Close"])], "DoubleBottom": DoubleBottom, "DoubleTop": DoubleTop, "DoubleTop_Accuracy": DoubleTop_Accuracy, "DoubleBottom_Accuracy": DoubleBottom_Accuracy }

# @app.route('/pattern', methods=['POST'])
# def predict_next_candle():
#     try:
#         data = request.get_json()
#         SYMB = data['SYMB']
#         Interval = data['Interval']
#         Range = data['Range']
#         response = get_pattern(SYMB, Interval, Range)   
#         return jsonify(response)
#     except Exception as e:
#             print(str(e))
#             return str(e)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True, port=5002)