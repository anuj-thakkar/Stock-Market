from flask import Flask, render_template
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np

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
    return render_template('index.html')


@app.route('/prediction')
def prediction():
    # Get the latest data (today's data) from the processed data
    latest_data = stock_data.iloc[-1]
    print(latest_data)

    # Extract required fields for display on the prediction page
    date = latest_data['Date']
    close = latest_data['Close']
    volume = latest_data['Volume']
    open_price = latest_data['Open']
    high = latest_data['High']
    low = latest_data['Low']

    # Get the latest data (today's data) from the processed data
    latest_data = stock_data.iloc[-1]

    # Extract the features needed for prediction
    features = latest_data.drop(['Close', 'Date', 'Quarter']).values
    print(features)
    
    # Reshape features for LSTM input
    X_lstm = features.reshape(1, 1, len(features))

    # avoid this error: Failed to convert a NumPy array to a Tensor (Unsupported object type numpy.float64).

    X_lstm = X_lstm.astype('float32')

    # Predict the price for tomorrow
    predicted_scaled = loaded_model.predict(X_lstm)
    
    # Inverse transform to get the actual predicted price
    scaler = StandardScaler()
    scaler.fit(stock_data['Close'].values.reshape(-1, 1))
    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]
    print("predicted price: ", predicted_price)
    print(type(predicted_price))
    print("shape of predicted price: ", predicted_price.shape)

    return render_template('prediction.html',
                            date=date,
                            close=close,
                            volume=volume,
                            open=open_price,
                            high=high,
                            low=low,
                            predicted_price=predicted_price,
                          )
                         
if __name__ == '__main__':
    app.run(debug=True)
