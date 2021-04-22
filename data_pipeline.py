import yfinance as yf


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
    return data



if __name__ == '__main__':
    get_time_series_df()

