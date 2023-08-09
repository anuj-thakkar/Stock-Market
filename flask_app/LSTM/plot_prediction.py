import plotly.express as px

def create_prediction_plot(data_df, predicted_price):
    fig = px.line(data_df, x='Date', y=['Close'],
                  labels={'Close': 'Actual Close Price'},
                  title='Actual vs. Predicted Stock Prices')
    fig.add_scatter(x=[data_df['Date'].iloc[-1]], y=[predicted_price],
                    mode='markers', name='Predicted Price', marker=dict(size=10, color='red'))
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Price')
    return fig