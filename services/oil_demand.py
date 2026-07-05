from __future__ import annotations

import pandas as pd
import requests


OWID_OIL_CONSUMPTION_URL = (
    "https://ourworldindata.org/grapher/oil-consumption-by-country.csv"
)


def fetch_world_oil_demand_debug() -> tuple[pd.DataFrame, str | None]:
    """
    Hakee maailman öljynkulutuksen Our World in Data -datasta.

    Palauttaa DataFramen sarakkeilla:
    - Year
    - Value
    - Unit
    """

    try:
        response = requests.get(
            OWID_OIL_CONSUMPTION_URL,
            timeout=30,
        )
        response.raise_for_status()

        from io import StringIO

        df = pd.read_csv(StringIO(response.text))

        if df.empty:
            return pd.DataFrame(), "Our World in Data -öljynkulutusdata oli tyhjä."

        # Tyypilliset sarakkeet: Entity, Code, Year, Oil consumption - TWh
        value_cols = [
            c for c in df.columns
            if c not in ["Entity", "Code", "Year"]
        ]

        if not value_cols:
            return pd.DataFrame(), f"Öljynkulutuksen arvosaraketta ei löytynyt. Sarakkeet: {list(df.columns)}"

        value_col = value_cols[0]

        world = df[df["Entity"].astype(str).str.lower() == "world"].copy()

        if world.empty:
            return pd.DataFrame(), "World-riviä ei löytynyt OWID-öljynkulutusdatasta."

        world["Year"] = pd.to_numeric(world["Year"], errors="coerce")
        world["Value"] = pd.to_numeric(world[value_col], errors="coerce")

        world = (
            world.dropna(subset=["Year", "Value"])
            .sort_values("Year")
            .reset_index(drop=True)
        )

        world["Year"] = world["Year"].astype(int)
        world["Unit"] = value_col

        return world[["Year", "Value", "Unit"]], None

    except Exception as e:
        return pd.DataFrame(), f"Öljynkulutusdatan haku epäonnistui: {e}"