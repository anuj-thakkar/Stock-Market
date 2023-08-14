# Stock Price Prediction App

This project is a simple web application that predicts the next day's closing stock price using a trained LSTM model. The app fetches stock price data, performs feature engineering, scales the features, and then uses the trained model to predict the stock price.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [Endpoints](#endpoints)

## Getting Started

### Prerequisites

- Python 3.x
- Flask
- Pandas
- NumPy
- Scikit-Learn
- Keras

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/anujthakkar2001/Stock-Market
   cd stock-prediction-app
   
## Usage
Run the Flask app:
```sh
  python app.py
```
Open your web browser and go to http://localhost:5000.
The home page provides a description of the project, and the prediction page displays the latest stock price 
data along with the predicted closing price for the next day.

## Data
The app fetches stock price data using the Yahoo Finance API. 
The data_loader.py script handles data fetching, updating, preprocessing, and feature engineering. 
The processed data is saved in the data/clean directory.

## Model
The LSTM model used for prediction is trained using historical stock price data.
The training process is performed in the train_model.py script, and the trained model is saved as trained_lstm_model.h5.

## Endpoints
`/`: Displays the home page with project description.

`/prediction`: Displays the latest stock price data and the predicted closing price for the next day.
