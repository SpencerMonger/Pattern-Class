import requests
import datetime
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from datetime import datetime, timedelta
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import numpy as np
import pytz 
from sklearn.preprocessing import StandardScaler
import dash_table

# Create a Dash app
app = dash.Dash(__name__, url_base_pathname='/stock_market_2/')
Scale=StandardScaler()

# Define the layout of the app
app.layout = html.Div([
    html.H1("Candlestick Chart", style={'text-align': 'center', 'margin-bottom': '50px','font-family': "Rocher","src": "url(https://assets.codepen.io/9632/RocherColorGX.woff2)"}),
    html.Div([
        html.Label("Symbol:", style={'font-weight': 'bold'}),
        # dcc.Input(id="symbol-input", type="text", value="PAYTM.NS", style={'margin-right': '50px'}),
        dcc.Dropdown(
        id="symbol-input",
        options=[
            {'label': 'RELIANCE', 'value': 'RELIANCE.NS'},
            {'label': 'GOOGLE', 'value': 'GOOGL'},
            {'label': 'AAPLE', 'value': 'AAPL'},
            {'label': 'NVIDIA', 'value': 'NVDA'},
            {'label': 'TESLA', 'value': 'TSLA'},
            {'label': 'MICROSOFT', 'value': 'MSFT'},
            {'label': 'AMAZON', 'value': 'AMZN'},
            {'label': 'SPOTIFY', 'value': 'SPOT'},
            {'label': 'AIRBNB', 'value': 'ABNB'},
            {'label': 'AMD', 'value': 'AMD'},
            {'label': 'FACEBOOK', 'value': 'META'},
        ],
        value='',
        clearable=False,
        style={'margin-right': '50px', 'width': '130px'}
        ),
        html.Label("Interval : ", style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id="interval-dropdown",
            options=[
                {'label': '1 minute', 'value': 1},
                {'label': '5 minutes', 'value': 5},
                {'label': '15 minutes', 'value': 15}
            ],
            value=1,
            clearable=False,
            style={'margin-right': '50px'}
        ),
        html.Label("Range : ", style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id="range-dropdown",
            options=[
                {'label': '1 day', 'value': 1},
                {'label': '2 days', 'value': 2},
                {'label': '3 days', 'value': 3},
                {'label': '4 days', 'value': 4},
                {'label': '5 days', 'value': 5},
                {'label': '6 days', 'value': 6},
                {'label': '7 days', 'value': 7},
            ],
            value=1,
            clearable=False,
            style={'margin-right': '50px'}
        ),
        html.Button("Start", id="start-button", n_clicks=0, style={'margin-right': '10px'}),
        html.Button("Stop", id="stop-button", n_clicks=0),
        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'margin-bottom': '20px'}),
        dcc.Graph(id='live-graph'),
        html.Div(id='accuracy-prediction-time', style={'margin-bottom': '30px'}),
        dcc.Interval(
            id='interval-component',
            interval=1*60*1000,  
            n_intervals=0
        )],
        style={'margin': '50px auto', 'max-width': '1200px'})

# Initialize global variables
running = False
df = None

def get_stock_data(SYMB, Interval, Range):

    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{SYMB}?region=US&lang=en-US&includePrePost=false&interval={Interval}m&useYfid=true&range={Range}d&corsDomain=finance.yahoo.com&.tsrc=finance'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers, timeout=5)
    if response.status_code == 200:
        data = response.json()
    else:
        print(f"Error: Request failed with status code {response.status_code}")
        return None

    # Extract data from the JSON
    meta_data = data['chart']['result'][0]['meta']
    timestamp = data['chart']['result'][0]['timestamp']
    quote_data = data['chart']['result'][0]['indicators']['quote'][0]

    # Convert timestamp to Indian Standard Time (IST)
    utc = pytz.timezone('UTC')
    ist = pytz.timezone('US/Eastern') #US/Eastern Asia/Kolkata
    ist_time = [datetime.fromtimestamp(ts, tz=utc).astimezone(ist) for ts in timestamp]

    # Create a DataFrame from the data
    df = pd.DataFrame({
        'symbol': [meta_data['symbol']] * len(timestamp),
        'Date': ist_time,
        'Open': quote_data['open'],
        'High': quote_data['high'],
        'Low': quote_data['low'],
        'Close': quote_data['close'],
        'Volume': quote_data['volume']
    })
    
    df['Range'] = df['High'] - df['Low']     # High-low
    df['Change'] = df['Close'].diff()  # Calculates the price change between current and previous row's close price
    df['Typical Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['Typical Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff(1).apply(lambda x: x if x > 0 else 0).rolling(14).sum() / df['Close'].diff(1).apply(lambda x: abs(x) if x < 0 else 0).rolling(14).sum()))
    df.dropna(inplace=True)
    return df

past = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close','Volume', 'Range', 'Change', 'VWAP', 'SMA20','RSI'])

def next_candle(SYMB, Interval, Range):
    start_time = time.time()

    df = get_stock_data(SYMB,Interval,Range)
    data = df[['Open', 'High', 'Low', 'Close','Volume', 'Range', 'Change', 'VWAP', 'SMA20','RSI']].values
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
    model.add(LSTM(50, return_sequences=True, input_shape=(time_steps, 10)))
    model.add(LSTM(50))
    model.add(Dense(10))  # 4 output nodes for predicting 'Open', 'High', 'Low', 'Close'

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    y_pred = model.predict(X_test)

    # Accuracy
    y_pred_original = scaler.inverse_transform(y_pred)
    y_test_original = scaler.inverse_transform(y_test)

    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    accuracy = 100 - mape(y_test_original, y_pred_original)

    # Prediction
    last_time_step_data = data_normalized[-time_steps:]
    next_future_value = model.predict(np.array([last_time_step_data]))
    next_future_value = scaler.inverse_transform(next_future_value)

    df = df[['Date','Open','High','Low','Close', 'Volume']]
    df['Date'] = pd.to_datetime(df['Date'])
    curr_time = df['Date'].iloc[-1] + timedelta(minutes=int(Interval))
    target = pd.DataFrame(next_future_value, columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Range', 'Change', 'VWAP', 'SMA20','RSI'])
    print(target)
    target['Date'] = curr_time

    end_time = time.time()
    execution_time = end_time - start_time
    return df, target, accuracy, execution_time


def calculate_accuracy(actual, predicted):  
    actual = np.array(actual)
    predicted = np.array(predicted)
    accuracy = abs(100 - abs(actual - predicted) * 100)
    mask = np.abs(actual - predicted) <= 1
    print(mask)
    accuracies = np.where(mask, accuracy, 0)
    print("Individual Accuracies:", accuracies)
    mean_accuracy = np.mean(accuracies)
    return mean_accuracy


@app.callback(
    Output('interval-component', 'interval'),
    [Input('interval-dropdown', 'value')]
)

def update_interval(value):
    if value is None or value <= 0:
        return 1*60*1000 
    else:
        return value*60*1000 
    
# Define the callback to start and stop fetching data
@app.callback(
    Output('live-graph', 'figure'),
    Output('accuracy-prediction-time', 'children'),
    Input('interval-component', 'n_intervals'),
    Input('start-button', 'n_clicks'),
    Input('stop-button', 'n_clicks'),
    State('symbol-input', 'value'),
    State('interval-dropdown', 'value'),
    State('range-dropdown', 'value'),
    State('live-graph', 'figure'),
    prevent_initial_call=True
)

def update_graph(n_intervals, start_clicks, stop_clicks, symbol, interval, range, prev_graph_state):
    global running
    global df
    global past

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'start-button':
        past = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close'])
        running = True
        
    elif triggered_id == 'stop-button':
        past = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close'])
        running = False

    if running:
        last = past
        df2, future, accuracy, prediction_time = next_candle(symbol, interval, range)
        df = df2.iloc[-41:].reset_index(drop=True)
        
        past = pd.concat([past, future], ignore_index=True)
        past['Date'] = past['Date'].dt.floor('1min')
        
        Accuracy = None
        Accuracy_df = pd.DataFrame(columns=['Date', 'Accuracy'])
        if len(past)>2:
            merged_df = pd.merge(df, past, on='Date', suffixes=('_actual', '_predicted'))
            accuracy_list = []
            for index, row in merged_df.iterrows():
                actual = row[['Open_actual', 'High_actual', 'Low_actual', 'Close_actual']].values
                print(f'Actul {actual}')
                predicted = row[['Open_predicted', 'High_predicted', 'Low_predicted', 'Close_predicted']].values
                print(f'Actul {predicted}')
                accuracy = calculate_accuracy(actual, predicted)
                accuracy_list.append(accuracy)

            # Add 'Accuracy' and 'MAPE' columns to the merged dataframe
            merged_df['Accuracy'] = accuracy_list
            print(merged_df[['Date','Accuracy']])
            print(merged_df['Accuracy'].mean())
            
            Accuracy = merged_df['Accuracy'].mean()
            
            Accuracy_df = merged_df[['Date','Accuracy']]
        
        if df is None:
            return prev_graph_state, "", ""

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.8, 0.2])
        fig.add_trace(go.Candlestick(x=df['Date'],
                                    open=df['Open'],
                                    high=df['High'],
                                    low=df['Low'],
                                    close=df['Close'],
                                    name="Actual Candlesticks"),row=1, col=1)
        
        volume_trace = go.Bar(
            x=df['Date'],
            y=df['Volume'],
            marker=dict(color='blue'),
            showlegend=False
        )
        fig.add_trace(volume_trace, row=2, col=1)
        
        fig.add_trace(go.Candlestick(x=future['Date'],
                                    open=future['Open'],
                                    high=future['High'],
                                    low=future['Low'],
                                    close=future['Close'],
                                    increasing_line_color='gray',
                                    decreasing_line_color='gray',
                                    name='Predicted Candlestick'
                                    ),row=1, col=1)
                        
        fig.add_trace(go.Candlestick(x=last['Date'],
                                    open=last['Open'],
                                    high=last['High'],
                                    low=last['Low'],
                                    close=last['Close'],
                                    increasing_line_color='gray',
                                    decreasing_line_color='gray',
                                    name='Previous Prediction'
                                    ),row=1, col=1)
        
        # fig.update_xaxes(title_text='Time', row=1, col=1)
        fig.update_yaxes(title_text='Price', row=1, col=1)
        fig.update_layout(title=f'Stock: {symbol}', xaxis_rangeslider_visible=False)
        fig.update_xaxes(title_text='Time', row=2, col=1)
        fig.update_yaxes(title_text='Volume', row=2, col=1)
                
        fig.update_layout(
            title=f"STOCK : {symbol} | Interval : {str(interval)}m",
            title_x=0.5,
            xaxis_title='Time',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=5, label="5m", step="minute", stepmode="backward"),
                        dict(count=10, label="10m", step="minute", stepmode="backward"),
                        dict(count=30, label="30m", step="minute", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                type="date"
            )
        )
        
        fig.update_yaxes(title_text='Price', side='left', row=1, col=1)
        fig.update_yaxes(side='right', row=1, col=1)


        accuracy_prediction_time = html.Div([
            html.Label(f"Accuracy in Percentage: {Accuracy}%", style={'font-style': 'italic', 'color': 'green' if Accuracy else 'red'}),
            html.Br(),
            html.Div([
                dash_table.DataTable(
                    columns=[{"name": col, "id": col} for col in Accuracy_df.columns],
                    data=Accuracy_df.to_dict('records'),
                    )
                ], style={'margin-top': '20px', 'text-align': 'center'})
        ], style={'text-align': 'center'})

        return fig, accuracy_prediction_time

    return prev_graph_state, "", ""

if __name__ == '__main__':
    app.run_server(debug=False, port=8002)