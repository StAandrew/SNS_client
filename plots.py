import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data_acquisition import get_historical_data, process_stock_data


def plot_prediction(ticker, predictions):
    dataset = get_historical_data(ticker)
    dataset = process_stock_data(dataset)
    dataset.set_index("Date", inplace=True)
    
    # Convert the index to datetime objects
    dataset.index = pd.to_datetime(dataset.index)
    predictions.index = pd.to_datetime(predictions.index)

    fig, ax = plt.subplots()
    
    ax.plot(dataset.tail(60).index, dataset.tail(60)['Close'], color='red', label=f'Historical {ticker} Price')
    ax.plot(predictions.index, predictions['Close'], color='blue', label=f'Predicted {ticker} Price')
    ax.set_title(f'{ticker} Price Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{ticker} Price')
    
    # Set the x-axis to display one label per month
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Rotate the x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax.legend()

def plot_daily_returns(ticker, predictions):
    
    # Convert the index to datetime objects
    predictions.index = pd.to_datetime(predictions.index)

    fig, ax = plt.subplots()
    
    ax.plot(predictions.index, predictions['Close'], color='blue')
    ax.set_title(f'{ticker} Daily Return Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{ticker} Daily Return')
    
    # Set the x-axis to display one label per month
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Rotate the x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax.legend()

def plot_daily_portfolio_returns(ticker_list, combined_returns):    # feed combined returns df

    combined_returns.index = pd.to_datetime(combined_returns.index)
    total_returns = combined_returns.sum(axis=1)

    fig, ax = plt.subplots()

    for ticker in ticker_list:
        ax.plot(combined_returns.index, combined_returns[ticker], label=f'{ticker}')
    ax.plot(total_returns.index, total_returns.iloc[:,0], linewidth=2, label=f'Portfolio')
    ax.set_title(f'Portfolio Daily Return Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel(f'Daily Returns')
    
    # Set the x-axis to display one label per month
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Rotate the x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax.legend()

def plot_cumulative_portfolio_returns(ticker_list, combined_returns):

    combined_returns.index = pd.to_datetime(combined_returns.index)
    cumulative_returns = combined_returns.cumsum()
    total_returns = cumulative_returns.sum(axis=1)

    fig, ax = plt.subplots()

    for ticker in ticker_list:
        ax.plot(cumulative_returns.index, cumulative_returns[ticker], label=f'{ticker}')
    ax.plot(total_returns.index, total_returns.iloc[:,0], linewidth=2, label=f'Portfolio')
    ax.set_title(f'Portfolio Cumulative Return Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel(f'Returns')
    
    # Set the x-axis to display one label per month
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Rotate the x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax.legend()

def plot_opt_portfolio(ticker_list, weights, type):

    fig, ax = plt.subplots()
    ax.pie(weights, labels=ticker_list, autopct='%1.1f%%')
    if type == 'var':
        ax.set_title(f'Minimum Variance Portfolio')
    elif type == 'sharpe':
        ax.set_title(f'Maximum Sharpe Portfolio')