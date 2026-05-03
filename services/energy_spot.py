from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# 🔧 Ladataan .env varmasti projektin juuresta
BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone


ENTSOE_ENDPOINT = "https://web-api.tp.entsoe.eu/api"
DOMAIN_FI = "10YFI-1--------U"


# ----------------------------------------
# 🧪 DEBUG
# ----------------------------------------
def debug_entsoe_env() -> dict:
    key = os.getenv("ENTSOE_API_KEY")

    return {
        "energy_spot_file": str(Path(__file__).resolve()),
        "base_dir": str(BASE_DIR),
        "env_path": str(ENV_PATH),
        "env_exists": ENV_PATH.exists(),
        "env_key_found": key is not None and len(key.strip()) > 0,
        "env_key_length": len(key.strip()) if key else 0,
        "working_directory": str(Path.cwd()),
    }


# ----------------------------------------
# 🕒 Aikaväli
# ----------------------------------------
def _build_time_range(hours_back: int = 24 * 7, hours_forward: int = 24) -> tuple[str, str]:
    now = datetime.now(timezone.utc)

    start = now - timedelta(hours=hours_back)
    end = now + timedelta(hours=hours_forward)

    return (
        start.strftime("%Y%m%d%H%M"),
        end.strftime("%Y%m%d%H%M"),
    )


# ----------------------------------------
# ⚡ DATA HAKU
# ----------------------------------------
def fetch_fi_day_ahead_spot(
    hours_back: int = 24 * 7,
    hours_forward: int = 24,
) -> pd.DataFrame:

    api_key = os.getenv("ENTSOE_API_KEY")

    if not api_key:
        raise ValueError("ENTSOE_API_KEY puuttuu ympäristömuuttujista")

    start, end = _build_time_range(hours_back, hours_forward)

    params = {
        "securityToken": api_key,
        "documentType": "A44",
        "in_Domain": DOMAIN_FI,
        "out_Domain": DOMAIN_FI,
        "periodStart": start,
        "periodEnd": end,
    }

    response = requests.get(ENTSOE_ENDPOINT, params=params, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(f"ENTSOE API error: {response.status_code} {response.text}")

    root = ET.fromstring(response.content)

    ns = {"ns": root.tag.split("}")[0].strip("{")}

    records = []

    for ts in root.findall(".//ns:TimeSeries", ns):
        period = ts.find(".//ns:Period", ns)
        if period is None:
            continue

        start_time = period.findtext("ns:timeInterval/ns:start", namespaces=ns)
        if start_time is None:
            continue

        start_dt = pd.to_datetime(start_time)

        for point in period.findall("ns:Point", ns):
            position = int(point.findtext("ns:position", namespaces=ns))
            price = point.findtext("ns:price.amount", namespaces=ns)

            if price is None:
                continue

            price = float(price)
            ts_dt = start_dt + pd.Timedelta(hours=position - 1)

            records.append({
                "Time": ts_dt,
                "Price_EUR_MWh": price,
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    df["Time"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")
    df["Price_EUR_MWh"] = pd.to_numeric(df["Price_EUR_MWh"], errors="coerce")
    df = df.dropna(subset=["Time", "Price_EUR_MWh"])

    # Jos ENTSO-E palauttaa samaan tuntiin useamman rivin, yhdistetään ne.
    df = (
        df.groupby("Time", as_index=False)["Price_EUR_MWh"]
        .mean()
        .sort_values("Time")
    )

    # €/MWh → snt/kWh
    df["Price_snt_kWh"] = df["Price_EUR_MWh"] / 10.0

    # Suomen aika
    df["Time"] = df["Time"].dt.tz_convert("Europe/Helsinki")

    return df.reset_index(drop=True)

# ----------------------------------------
# 📊 YHTEENVETO
# ----------------------------------------
def build_spot_summary(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}

    latest = df.iloc[-1]["Price_snt_kWh"]

    prev = df.iloc[-24]["Price_snt_kWh"] if len(df) > 24 else None

    delta = None
    if prev is not None and prev != 0:
        delta = (latest / prev - 1) * 100

    return {
        "latest": latest,
        "delta_24h": delta,
        "min": df["Price_snt_kWh"].min(),
        "max": df["Price_snt_kWh"].max(),
    }