import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df
    

def preprocess_data(df):
    # Define the features and target variables
    target = ['Close']
    features = df.drop(['Close', 'Date', 'Quarter'], axis=1).columns.tolist()

    # Create arrays for the features and the response variable
    X = df[features].values
    y = df[target].values

    # Apply scaling to the target variable 'y' (Close prices)
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y)

    return X, y_scaled

def prepare_lstm_input(X):
    time_steps = 1
    batch_size = X.shape[0]  # Get the number of samples in the batch
    X_lstm = X.reshape(batch_size, time_steps, X.shape[1])
    X_lstm = X_lstm.astype('float32')

    return X_lstm

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape, activation='tanh', return_sequences=True))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_lstm_model(model, X_train, y_train, epochs, batch_size):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=False)
    return history

def print_model_summary(model):
    print(model.summary())

if __name__ == "__main__":
    data_file_path = '/Users/anujthakkar/Documents/Purdue/Projects/Stock Market/flask_app/data/clean/AAPL_feature_engineered.csv'
    epochs = 100
    batch_size = 5

    apple_df = load_data(data_file_path)

    X, y_scaled = preprocess_data(apple_df)
    X_lstm = prepare_lstm_input(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_scaled, test_size=0.2, shuffle=False)

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    history = train_lstm_model(model, X_train, y_train, epochs=epochs, batch_size=batch_size)
    print_model_summary(model)

    # print model accuracy
    train_score = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (train_score, np.sqrt(train_score)))

    # Save the trained model
    model.save('trained_lstm_model.h5')
    print('Model saved to trained_lstm_model.h5')