# services/market_data.py
import yfinance as yf
import pandas as pd


def fetch_price_history(symbol: str, period: str = "5y") -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, progress=False)

        if df is None or df.empty:
            return pd.DataFrame()

        # Jos sarakkeet ovat MultiIndex (('Close','BZ=F')), litistetään -> 'Close'
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df = df.reset_index()

        if "Date" not in df.columns or "Close" not in df.columns:
            return pd.DataFrame()

        return df

    except Exception:
        return pd.DataFrame()










