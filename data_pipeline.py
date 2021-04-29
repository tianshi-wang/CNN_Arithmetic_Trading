import os

import yfinance as yf
import pandas as pd
import numpy as np
from pyts.image import GramianAngularField
from pathlib import Path

# all tickers to pull data
tickers=['AAPL', 'QQQ']
data_folder = Path(__file__).parent/"training_data"


def get_time_series_df(period: str = '1y'):
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

    data = data.drop(columns=[(ticker, column) for ticker in tickers for column in ('Open', 'High', 'Low')])
    for ticker in tickers:
        data[ticker, 'Close_pct'] = data[ticker]['Close'].pct_change()
    return data


def set_gaf_data(df, image_size: int = 20):
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
        # Group data_slice by time frequency
        stack = None
        for freq in ['1d', '7d']:
            group_dt = data_slice.groupby(pd.Grouper(key='Date', freq=freq)).mean().reset_index()
            group_dt = group_dt.dropna()
            layer = group_dt['Close'].tail(image_size).values
            layer = layer[None, ...]
            if stack is None:
                stack = layer
            else:
                stack = np.append(stack, layer, axis=0)
        # Decide what trading position we should take on that day
        date_to_labels[dates[index]] = df[df['Date'] == dates[index]]['Close_pct'].iloc[-1]
        date_to_gafs[dates[index]] = stack
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
    data = get_time_series_df()
    for ticker in tickers:
        date_to_gafs, date_to_labels = set_gaf_data(data[ticker])
        if not os.path.isdir(data_folder/ticker):
            os.mkdir(data_folder/ticker)

        dates = sorted([dt for dt in date_to_labels])
        labels = [date_to_labels[dt] for dt in dates]
        pd.DataFrame({'date': dates, 'label': labels}).to_csv(path_or_buf=data_folder/ticker/'labels.csv', index=False)

        for dt in date_to_gafs:
            image = create_gaf(date_to_gafs[dt])
            np.save(data_folder/ticker/(str(dt)[:10]+".npy"), image)
