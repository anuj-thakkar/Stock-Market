import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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

    # Feature engineering
    apple_df = generate_features(apple_df)

    # Scale numerical features
    apple_df = scale_data(apple_df, numerical_features)

    # Save the processed data to a new CSV file
    save_data(apple_df, output_file)
