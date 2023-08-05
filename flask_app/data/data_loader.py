import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import datetime

def fetch_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data.reset_index()

def update_data(df, recent_data_df):
    # Get the most recent date in the existing dataset
    last_date_in_df = pd.to_datetime(df['Date']).max()

    # Filter the recent_data_df to include only data from the last date in the existing dataset
    recent_data_df = recent_data_df[recent_data_df['Date'] > last_date_in_df]

    # Concatenate the existing dataset with the recent data
    updated_df = pd.concat([df, recent_data_df], ignore_index=True)

    return updated_df


def load_data(file_path):
    return pd.read_csv(file_path)

def generate_features(df):

    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change(periods=1)

    # 5-day rolling averages for close price and volume
    df['5_day_mean_close_price'] = df['Close'].rolling(5).mean()
    df['5_day_mean_volume'] = df['Volume'].rolling(5).mean()

    # Calculate daily range and volatility
    df['Daily_Range'] = df['High'] - df['Low']
    df['Volatility'] = df['Daily_Return'].rolling(5).std()

    # Create a new column called Quarter
    df['Quarter'] = pd.PeriodIndex(df['Date'], freq='Q')

    # Fill missing values
    df['5_day_mean_close_price'] = df['5_day_mean_close_price'].fillna(0)
    df['5_day_mean_volume'] = df['5_day_mean_volume'].fillna(0)
    df['Volatility'] = df['Volatility'].fillna(0)
    df['Daily_Return'] = df['Daily_Return'].fillna(0)

    # Calculate 5-day and 20-day exponential moving averages for closing price
    df['EMA_Close_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_Close_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    return df

def scale_data(df, features_to_scale):
    """Scale numerical features using MinMaxScaler."""
    scaler = MinMaxScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    return df

def save_data(df, output_file):
    """Save the processed DataFrame to a new CSV file."""
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    # Define file paths
    input_file = 'data/AAPL.csv'
    output_file = 'data/clean/AAPL_feature_engineered.csv'

    # Define numerical features to scale
    numerical_features = ['Volume', 'Open', 'High', 'Low', 'Daily_Return',
                          '5_day_mean_close_price', '5_day_mean_volume', 'Daily_Range',
                          'Volatility', 'EMA_Close_5', 'EMA_Close_20']

    # Load data
    apple_df = load_data(input_file)

    # Fetch recent data
    # Calculate start_date and end_date based on existing data and current date
    start_date = (pd.to_datetime(apple_df['Date']).max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    recent_data_df = fetch_stock_data('AAPL', start_date, end_date)

    # Update the existing dataset with the recent data
    apple_df = update_data(apple_df, recent_data_df)

    # Feature engineering
    apple_df = generate_features(apple_df)

    # Scale numerical features
    apple_df = scale_data(apple_df, numerical_features)

    # Save the processed data to a new CSV file
    save_data(apple_df, output_file)
