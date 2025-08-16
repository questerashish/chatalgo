import streamlit as st
import pandas as pd
import datetime

"""
Streamlit application to backtest a simple moving‑average crossover strategy on
Indian equities.  By default the app pulls daily price data for Reliance
Industries from Yahoo! Finance (ticker ``RELIANCE.NS``) over the last two
months, computes 5‑ and 10‑day simple moving averages and simulates a
long‑only strategy that buys when the short average crosses above the long
average and exits when it crosses below.

This script requires the ``yfinance`` package to fetch data.  If it is not
installed on your system you can install it with ``pip install yfinance``.
"""

def load_data(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """Fetch historical price data for the given symbol using yfinance.

    Parameters
    ----------
    symbol : str
        The ticker symbol to download (e.g. ``"RELIANCE.NS"``).
    start : datetime.date
        The start date for the download.
    end : datetime.date
        The end date for the download.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by date with columns for open, high, low,
        close, adjusted close and volume.
    """
    import yfinance as yf
    data = yf.download(symbol, start=start, end=end)
    data = data.rename(columns=str.lower)
    return data

def compute_strategy(df: pd.DataFrame, short_window: int = 5, long_window: int = 10) -> pd.DataFrame:
    """Compute a simple moving average crossover strategy.

    The strategy goes long when the short moving average crosses above the long
    moving average and exits when it crosses below.  Returns a copy of the
    original DataFrame with additional columns for the moving averages,
    trading signals and strategy returns.

    Parameters
    ----------
    df : pd.DataFrame
        Historical price data with a ``close`` column.
    short_window : int, optional
        Window length for the short moving average, by default 5.
    long_window : int, optional
        Window length for the long moving average, by default 10.

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns ``sma_short``, ``sma_long``, ``signal``,
        ``position`` and ``strategy_return``.
    """
    df = df.copy()
    df['sma_short'] = df['close'].rolling(window=short_window).mean()
    df['sma_long'] = df['close'].rolling(window=long_window).mean()

    # initialise signals
    df['signal'] = 0
    for i in range(1, len(df)):
        if pd.notnull(df.at[df.index[i], 'sma_short']) and pd.notnull(df.at[df.index[i], 'sma_long']):
            prev_short = df.at[df.index[i - 1], 'sma_short']
            prev_long = df.at[df.index[i - 1], 'sma_long']
            curr_short = df.at[df.index[i], 'sma_short']
            curr_long = df.at[df.index[i], 'sma_long']
            if curr_short > curr_long and prev_short <= prev_long:
                df.at[df.index[i], 'signal'] = 1  # buy
            elif curr_short < curr_long and prev_short >= prev_long:
                df.at[df.index[i], 'signal'] = -1  # sell/exit

    # derive positions by carrying forward the last signal
    position = 0
    positions = []
    for sig in df['signal']:
        if sig == 1:
            position = 1
        elif sig == -1:
            position = 0
        positions.append(position)
    df['position'] = positions

    # daily percentage changes
    df['pct_change'] = df['close'].pct_change().fillna(0)
    # strategy returns assume trades executed next day at close
    df['strategy_return'] = df['position'].shift(1).fillna(0) * df['pct_change']

    return df

def app() -> None:
    st.title("Backtest Moving‑Average Strategy on NSE Stocks")
    st.markdown(
        """
        This tool fetches daily price data from Yahoo! Finance for the selected stock,
        computes short and long simple moving averages and evaluates a basic
        crossover strategy.  Adjust the parameters below to experiment with
        different settings.
        """
    )

    # sidebar inputs
    symbol = st.sidebar.text_input("Ticker symbol (e.g. RELIANCE.NS)", "RELIANCE.NS")
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=60)
    start_date = st.sidebar.date_input("Start date", default_start)
    end_date = st.sidebar.date_input("End date", today)
    short_window = st.sidebar.number_input("Short MA window", min_value=1, max_value=50, value=5)
    long_window = st.sidebar.number_input("Long MA window", min_value=2, max_value=200, value=10)

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return
    if short_window >= long_window:
        st.error("Short window must be less than long window.")
        return

    # fetch and display data
    with st.spinner("Downloading data..."):
        data = load_data(symbol, start_date, end_date)
    if data.empty:
        st.error("No data returned for the selected period.")
        return
    st.subheader("Price Chart")
    st.line_chart(data['close'], height=300)

    # run strategy
    result = compute_strategy(data, short_window, long_window)
    st.subheader("Backtest Results")
    buy_and_hold = (result['pct_change'] + 1).prod() - 1
    strategy_ret = (result['strategy_return'] + 1).prod() - 1
    st.write(f"Buy and Hold Return: **{buy_and_hold * 100:.2f}%**")
    st.write(f"Strategy Return: **{strategy_ret * 100:.2f}%**")
    st.dataframe(result[['close', 'sma_short', 'sma_long', 'signal', 'position', 'strategy_return']].tail())

    st.markdown(
        """
        **Notes**

        * The strategy presented here is for educational purposes only and
          should not be used as financial advice.
        * Yahoo! Finance data may be subject to outages or limitations,
          especially for Indian equities.  If no data appears, try again later
          or consider installing and using the `nsepy` package for more
          reliable NSE data.
        """
    )

if __name__ == "__main__":
    app()
