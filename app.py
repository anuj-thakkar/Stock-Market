from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)
app.config.from_pyfile('config.py')


# Load the processed data
data_file_path = 'flask_app/data/clean/AAPL_feature_engineered.csv'
stock_data = pd.read_csv(data_file_path)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prediction')
def prediction():
    # Get the latest data (today's data) from the processed data
    latest_data = stock_data.iloc[-1]

    # Extract required fields for display on the prediction page
    date = latest_data['Date']
    close = latest_data['Close']
    volume = latest_data['Volume']
    open_price = latest_data['Open']
    high = latest_data['High']
    low = latest_data['Low']

    # In a real app, you would use your LSTM model to make predictions here
    # For demonstration purposes, we'll just use a placeholder prediction
    predicted_close = close + 10.0

    return render_template('prediction.html',
                           date=date,
                           close=close,
                           volume=volume,
                           open_price=open_price,
                           high=high,
                           low=low,
                           predicted_close=predicted_close)


if __name__ == '__main__':
    app.run(debug=True)
