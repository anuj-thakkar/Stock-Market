from flask import Flask, render_template
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta


app = Flask(__name__)
app.config.from_pyfile('config.py')


# Load the processed data
data_file_path = 'flask_app/data/clean/AAPL_feature_engineered.csv'
stock_data = pd.read_csv(data_file_path)

# Load the trained LSTM model
model_path = 'trained_lstm_model.h5'
loaded_model = load_model(model_path)


@app.route('/')
def index():
    print('Main page loaded')
    return render_template('index.html')


@app.route('/prediction')
def prediction():
    from sklearn.preprocessing import MinMaxScaler
    import joblib, os

    # Load the scaler object used for scaling
    scaler_directory = '/Users/anujthakkar/Documents/Purdue/Projects/Stock Market/flask_app/data/scalers'  # Provide the appropriate directory path
    scaler_filename = 'features_scaler.save'
    scaler_path = os.path.join(scaler_directory, scaler_filename)
    mm_scaler = joblib.load(scaler_path)

    # Define the numerical features to unscale
    numerical_features = ['Volume', 'Open', 'High', 'Low', 'Daily_Return',
                          '5_day_mean_close_price', '5_day_mean_volume', 'Daily_Range',
                          'Volatility', 'EMA_Close_5', 'EMA_Close_20']

    # Get today's date in the format used in your DataFrame (e.g., 'YYYY-MM-DD')
    today_date = pd.to_datetime('today').strftime('%Y-%m-%d')

    # Filter the stock_data DataFrame to get the data for today's date
    today_data = stock_data[stock_data['Date'] == today_date].iloc[0]
    print("today's date:" , today_data['Date'])

    # Extract the features needed for prediction
    features = today_data.drop(['Close', 'Date', 'Quarter']).values

    # Reshape features for LSTM input
    X_lstm = features.reshape(1, 1, len(features))
    X_lstm = X_lstm.astype('float32')
    print("X_lstm: ", X_lstm) # returns features just for that day

    # Predict the price for today
    predicted_scaled = loaded_model.predict(X_lstm)[0][0]

    # Inverse transform to get the actual predicted price
    scaler = StandardScaler()
    scaler.fit(stock_data['Close'].values.reshape(-1, 1))
    predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]


    print("predicted price for the date: ", predicted_price, "type: ", type(predicted_price))

    # now, we can do the same for the other features
    # Define the numerical features to unscale
    numerical_features = ['Volume', 'Open', 'High', 'Low', 'Daily_Return',
                          '5_day_mean_close_price', '5_day_mean_volume', 'Daily_Range',
                          'Volatility', 'EMA_Close_5', 'EMA_Close_20']
    
    # Get the latest data (most previous close data) from the processed data
    latest_data = stock_data.iloc[-1].copy()

    # Convert the latest data to a DataFrame
    latest_data_df = pd.DataFrame([latest_data])

    # Unscale the relevant features
    unscaled_data = unscale_data(latest_data_df.copy(), mm_scaler, numerical_features)
    unscaled_latest_data = unscaled_data.iloc[0]

    print("UNSCALED LATEST DATA:")
    print(unscaled_latest_data)

    # show the plot
    fig = px.line(stock_data, x='Date', y='Close', title='Apple Stock Price')
    fig.update_xaxes(rangeslider_visible=True)

    return render_template('prediction.html',
                            date=today_data['Date'],
                            close=round(today_data['Close'], 2),
                            # round Volume to whole number
                            volume = round(unscaled_latest_data['Volume']), 
                            open=round(unscaled_latest_data['Open'], 2),
                            high=round(unscaled_latest_data['High'], 2),
                            low=round(unscaled_latest_data['Low'], 2),
                            predicted_price=round(predicted_price, 2),
                            fig=fig.to_html(full_html=False, default_height=500, default_width=700)                       
                          )

def unscale_data(data_df, scaler, numerical_features):
    """
    Unscale the numerical features of the data_df using the scaler object
    """
    data_df[numerical_features] = scaler.inverse_transform(data_df[numerical_features])

    return data_df


                         
if __name__ == '__main__':
    app.run(debug=True)
