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
- [Contributing](#contributing)
- [License](#license)

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
   git clone https://github.com/your-username/stock-prediction-app.git
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
