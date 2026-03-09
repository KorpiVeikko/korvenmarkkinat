import requests
import pandas as pd


@pd.api.extensions.register_dataframe_accessor("finance")
class FinanceAccessor:
    pass


def fetch_government_debt() -> pd.DataFrame:
    """
    Hakee Suomen valtion EMTN-lainat Tutkihallintoa-rajapinnasta.
    Palauttaa raakadatana pandas DataFrame.
    """
    url = "https://api.tutkihallintoa.fi/central-government-debt/v1/emtn-bond-issues?lang=FI"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    df = pd.DataFrame(r.json())
    return df

