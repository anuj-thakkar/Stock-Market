from flask import Flask, render_template
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.express as px


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

    # Get the latest data (today's data) from the processed data
    latest_data = stock_data.iloc[-1]
    print(latest_data)

    # Extract the features needed for prediction
    features = latest_data.drop(['Close', 'Date', 'Quarter']).values

    # Reshape features for LSTM input
    X_lstm = features.reshape(1, 1, len(features))

    # avoid this error: Failed to convert a NumPy array to a Tensor (Unsupported object type numpy.float64).

    X_lstm = X_lstm.astype('float32')

    # Predict the price for tomorrow
    predicted_scaled = loaded_model.predict(X_lstm)
    print("predicted scaled: ", predicted_scaled)
    
    # Inverse transform to get the actual predicted price
    scaler = StandardScaler()
    scaler.fit(stock_data['Close'].values.reshape(-1, 1))
    inverse = scaler.inverse_transform(predicted_scaled)
    predicted_price = inverse[0][0]

    print("predicted price: ", predicted_price)
    print(type(predicted_price))
    print("shape of predicted price: ", predicted_price.shape)


    # now, we can do the same for the other features
    # Define the numerical features to unscale
    numerical_features = ['Volume', 'Open', 'High', 'Low', 'Daily_Return',
                          '5_day_mean_close_price', '5_day_mean_volume', 'Daily_Range',
                          'Volatility', 'EMA_Close_5', 'EMA_Close_20']
    
    # Get the latest data (today's data) from the processed data
    latest_data = stock_data.iloc[-1].copy()

    # Convert the latest data to a DataFrame
    latest_data_df = pd.DataFrame([latest_data])

    # Unscale the relevant features
    unscaled_data = unscale_data(latest_data_df.copy(), mm_scaler, numerical_features)
    unscaled_latest_data = unscaled_data.iloc[0]

    print("Unscaled Latest Data:")
    print(unscaled_latest_data)

    # show the plot
    fig = px.line(stock_data, x='Date', y='Close', title='Apple Stock Price')
    fig.update_xaxes(rangeslider_visible=True)

    return render_template('prediction.html',
                            date=latest_data['Date'],
                            close=round(latest_data['Close'], 2),
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
    # Unscale the numerical features
    data_df[numerical_features] = scaler.inverse_transform(data_df[numerical_features])

    return data_df
                         
if __name__ == '__main__':
    app.run(debug=True)