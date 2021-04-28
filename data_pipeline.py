import yfinance as yf
import pandas as pd
import numpy as np
from pyts.image import GramianAngularField

# all tickers to pull data
tickers=['AAPL', 'QQQ']


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
    # df['Date'] = dates
    # dates = dates.drop_duplicates()
    # list_dates = dates.apply(str).tolist()
    index = image_size * 7
    labels = {}    # key: datetime, value: percentage change on the day i.e. close vs previous day close

    while True:
        if index >= len(dates) - 1:
            break
        # Select appropriate timeframe
        data_slice = df.loc[(df['Date'] > dates[index] - np.timedelta64(image_size * 7, 'D'))
                             & (df['Date'] < dates[index])]
        gafs = []
        # Group data_slice by time frequency
        for freq in ['1d', '7d']:
            group_dt = data_slice.groupby(pd.Grouper(key='Date', freq=freq)).mean().reset_index()
            group_dt = group_dt.dropna()
            gafs.append(group_dt['Close'].tail(image_size))
        # Decide what trading position we should take on that day
        labels[dates[index]] = df[df['Date'] == dates[index]]['Close_pct'].iloc[-1]
        # current_value = data_slice['Close'].iloc[-1]
        # decision = trading_action(future_close=future_value, current_close=current_value)
        # decision_map[decision].append([list_dates[index - 1], gafs])
        index += 1
    return gafs, labels

def create_gaf(ts):
    """
    :param ts:
    :return:
    """
    data = dict()
    gadf = GramianAngularField(method='difference', image_size=ts.shape[0])
    data['gadf'] = gadf.fit_transform(pd.DataFrame(ts).T)[0] # ts.T)
    return data


# def data_to_image_preprocess():
#     """
#     :return: None
#     """
#     ive_data = 'IVE_tickbidask.txt'
#     col_name = ['Date', 'Close', 'Close_pct', 'Volume']
#     # df = pd.read_csv(os.path.join(PATH, ive_data), names=col_name, header=None)
#     # # Drop unnecessary data
#     # df = df.drop(['High', 'Low', 'Volume'], axis=1)
#     # df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], infer_datetime_format=True)
#     # df = df.groupby(pd.Grouper(key='DateTime', freq='1h')).mean().reset_index()
#     # df['Open'] = df['Open'].replace(to_replace=0, method='ffill')
#     # Remove non trading days and times
#     # clean_df = clean_non_trading_times(df)
#     # Send to slicing
#     set_gaf_data(clean_df)


if __name__ == '__main__':
    data = get_time_series_df()
    for ticker in tickers:
        gafs, labels = set_gaf_data(data[ticker])

    pass


