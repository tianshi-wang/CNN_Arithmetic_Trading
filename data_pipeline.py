import os
from typing import List

import yfinance as yf
import pandas as pd
import numpy as np
from pyts.image import GramianAngularField
from pathlib import Path

# all tickers to pull data
TICKERS = ('AAPL', 'QQQ', 'AMZN', 'MSFT', 'INTC', 'MARA')
data_folder = Path(__file__).parent/"training_data"

pass

def get_time_series_df(period: str = '1y', tickers: List[str] = TICKERS):
    """
    period: the length of data to retrieve from current day
    Example:
    data.[('AAPL', 'Close')].values to get numpy array for all adjusted close prices of AAPL
    data.[('AAPL', 'Volume')].values to get numpy array for volumes of AAPL

    ToDo:
    1. convert a single time series to a list of smaller time series with a fixed window
    2. construct y-vector (predicted returns)
    """
    data = yf.download(tickers=tickers,
                       period=period,
                       group_by='tickers',
                       back_adjust=True,
                       auto_adjust=True)
    if len(tickers) == 1:
        data = data.drop(columns=['Open', 'High', 'Low'])
        data['Close_pct'] = data['Close'].pct_change()
        data = data.iloc[1:]
    else:
        data = data.drop(columns=[(ticker, column) for ticker in tickers for column in ('Open', 'High', 'Low')])
        for ticker in tickers:
            data[ticker, 'Close_pct'] = data[ticker]['Close'].pct_change()
        data = data.iloc[1:]
    return data


def set_gaf_data(df, image_size: int = 20, sp500=None):
    """
    :param df: DataFrame data
    :return: None
    """
    dates = df.index.values
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'Date'})
    index = image_size * 7
    date_to_labels = {}    # key: datetime, value: percentage change on the day i.e. close vs previous day close
    date_to_gafs = {}

    while True:
        if index >= len(dates) - 1:
            break
        # Select appropriate timeframe
        data_slice = df.loc[(df['Date'] > dates[index] - np.timedelta64(image_size * 7, 'D'))
                             & (df['Date'] < dates[index])]
        sp500_data_slice = sp500.loc[(sp500['Date'] > dates[index] - np.timedelta64(image_size * 7, 'D'))
                                     & (sp500['Date'] < dates[index])]
        # Group data_slice by time frequency
        stack = []

        group_dt_1d = data_slice.groupby(pd.Grouper(key='Date', freq='1d')).mean().reset_index().dropna()
        group_dt_7d = data_slice.groupby(pd.Grouper(key='Date', freq='7d')).mean().reset_index().dropna()
        close_pct_1d = group_dt_1d['Close_pct'].tail(image_size).values
        close_pct_7d = group_dt_7d['Close_pct'].tail(image_size).values
        volume_1d = group_dt_1d['Volume'].tail(image_size).values / max(group_dt_1d['Volume'].tail(image_size).values) / 10
        volume_7d = group_dt_7d['Volume'].tail(image_size).values / max(group_dt_7d['Volume'].tail(image_size).values) / 10

        sp500_group_dt_1d = sp500_data_slice.groupby(pd.Grouper(key='Date', freq='1d')).mean().reset_index().dropna()
        sp500_group_dt_7d = sp500_data_slice.groupby(pd.Grouper(key='Date', freq='7d')).mean().reset_index().dropna()
        sp500_close_pct_1d = sp500_group_dt_1d['Close_pct'].tail(image_size).values
        sp500_close_pct_7d = sp500_group_dt_7d['Close_pct'].tail(image_size).values

        # Layer 1 (1d close pct self correlation)
        stack.append(close_pct_1d.reshape(1, -1).T @ close_pct_1d.reshape(1, -1) * 100)

        # Layer 2 (7d close pct self correlation)
        stack.append(close_pct_7d.reshape(1, -1).T @ close_pct_7d.reshape(1, -1) * 100)

        # Layer 3 (1d close pct, 1day close volume, cross correlation)
        stack.append(close_pct_1d.reshape(1, -1).T @ volume_1d.reshape(1, -1) * 10)

        # layer 4 (7d close pct, 7day close volume, cross correlation)
        stack.append(close_pct_7d.reshape(1, -1).T @ volume_7d.reshape(1, -1) * 10)

        # Layer 5 (1d close pct Recurrence Layer)
        stack.append(np.array([abs(close_pct_1d - close_pct_1d[i]) for i in range(image_size)]) * 10)

        # Layer 6 (1d close pct, 1d sp500 close pct, cross correlation)
        stack.append(close_pct_1d.reshape(1, -1).T @ sp500_close_pct_1d.reshape(1, -1) * 100)

        # Layer 7 (7d close pct, 7d sp500 close pct, cross correlation)
        stack.append(close_pct_7d.reshape(1, -1).T @ sp500_close_pct_7d.reshape(1, -1) * 100)

        # Decide what trading position we should take on that day
        date_to_labels[dates[index]] = df[df['Date'] == dates[index]]['Close_pct'].iloc[-1]
        date_to_gafs[dates[index]] = np.array(stack)
        index += 1
    return date_to_gafs, date_to_labels


def create_gaf(stack):
    """
    :param ts:
    :return:
    """
    row, col = stack.shape
    gadf = GramianAngularField(method='difference', image_size=col)
    image = None
    for i in range(0, row):
        layer = gadf.fit_transform(pd.DataFrame(stack[i, :]).T)
        if image is None:
            image = layer
        else:
            image = np.append(image, layer, axis=0)
    return image


if __name__ == '__main__':
    period = '5y'
    data = get_time_series_df(period=period)
    sp500 = get_time_series_df(period=period, tickers=['SPY'])
    sp500.reset_index(inplace=True)
    sp500 = sp500.rename(columns={'index': 'Date'})
    for ticker in TICKERS:
        date_to_gafs, date_to_labels = set_gaf_data(data[ticker], sp500=sp500)
        if not os.path.isdir(data_folder/ticker):
            os.mkdir(data_folder/ticker)

        dates = sorted([dt for dt in date_to_labels])
        labels = [date_to_labels[dt] for dt in dates]
        pd.DataFrame({'date': dates, 'label': labels}).to_csv(path_or_buf=data_folder/ticker/'labels.csv', index=False)

        for dt in date_to_gafs:
            np.save(data_folder/ticker/(str(dt)[:10]+".npy"), date_to_gafs[dt])

