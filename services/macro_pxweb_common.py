from __future__ import annotations

import itertools

import pandas as pd
import requests


def dedupe_columns(cols: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []

    for col in cols:
        key = str(col)
        if key not in seen:
            seen[key] = 1
            out.append(key)
        else:
            seen[key] += 1
            out.append(f"{key}__{seen[key]}")

    return out


def parse_jsonstat2(payload: dict) -> pd.DataFrame:
    if not isinstance(payload, dict) or "dimension" not in payload:
        return pd.DataFrame()

    dim = payload["dimension"]
    ids = payload.get("id") or dim.get("id")
    values = payload.get("value")

    if not ids or values is None:
        return pd.DataFrame()

    dim_levels: list[list[str]] = []

    for dim_id in ids:
        dim_meta = dim.get(dim_id, {})
        category = dim_meta.get("category") or {}
        index = category.get("index")
        labels = category.get("label") or {}

        if isinstance(index, dict) and index:
            keys = [k for k, _ in sorted(index.items(), key=lambda kv: kv[1])]
        elif isinstance(index, list) and index:
            keys = index
        else:
            keys = list(labels.keys())

        # Palautetaan ensisijaisesti KOODI, ei tekstiä.
        # Tämä helpottaa uudessa PxWebissä, jossa muuttujakoodit ovat tärkeämpiä.
        dim_levels.append([str(k) for k in keys])

    combos = list(itertools.product(*dim_levels))
    cols = dedupe_columns([str(x) for x in ids])
    df = pd.DataFrame(combos, columns=cols)

    # JSON-stat voi palauttaa arvot joko listana tai harvana dictinä.
    if isinstance(values, list):
        if len(values) != len(combos):
            return pd.DataFrame()
        df["Arvo"] = pd.to_numeric(values, errors="coerce")

    elif isinstance(values, dict):
        out_values = [None] * len(combos)
        for k, v in values.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if 0 <= idx < len(out_values):
                out_values[idx] = v
        df["Arvo"] = pd.to_numeric(out_values, errors="coerce")

    else:
        return pd.DataFrame()

    df.columns = dedupe_columns(list(df.columns))
    return df


def post_px(url: str, query: dict, timeout: int = 45) -> pd.DataFrame:
    response = requests.post(
        url,
        json=query,
        timeout=timeout,
        headers={"User-Agent": "TaloudenSeuranta/1.0"},
    )

    if response.status_code != 200:
        raise RuntimeError(
            "PXWeb POST epäonnistui.\n\n"
            f"URL: {url}\n"
            f"Status: {response.status_code}\n\n"
            f"Query:\n{query}\n\n"
            f"Response:\n{response.text[:1000]}"
        )

    return parse_jsonstat2(response.json())


def get_px_meta(url: str, timeout: int = 45) -> dict:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "TaloudenSeuranta/1.0"},
    )

    if response.status_code != 200:
        raise RuntimeError(
            "PXWeb metadata-haku epäonnistui.\n\n"
            f"URL: {url}\n"
            f"Status: {response.status_code}\n"
            f"Response:\n{response.text[:1000]}"
        )

    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.columns = dedupe_columns(list(out.columns))

    candidates = [
        c for c in out.columns
        if str(c).strip().lower() in {
            "kuukausi",
            "vuosineljännes",
            "vuosineljannes",
            "neljännes",
            "aika",
            "time",
            "vuosi",
            "quarter",
        }
    ]

    time_col = candidates[0] if candidates else out.columns[0]
    s = out[time_col].astype(str).str.strip()
    out["Aika"] = s

    month_match = s.str.extract(r"^(?P<y>\d{4})M(?P<m>\d{2})$")
    quarter_match = s.str.extract(r"^(?P<y>\d{4})Q(?P<q>[1-4])")
    year_match = s.str.extract(r"^(?P<y>\d{4})$")

    if month_match["y"].notna().any():
        out["Vuosi_num"] = pd.to_numeric(month_match["y"], errors="coerce")
        out["Kuukausi_num"] = pd.to_numeric(month_match["m"], errors="coerce")
        out["Aika_dt"] = pd.to_datetime(
            out["Vuosi_num"].astype("Int64").astype(str)
            + "-"
            + out["Kuukausi_num"].astype("Int64").astype(str).str.zfill(2)
            + "-01",
            errors="coerce",
        )

    elif quarter_match["y"].notna().any():
        out["Vuosi_num"] = pd.to_numeric(quarter_match["y"], errors="coerce")
        qn = pd.to_numeric(quarter_match["q"], errors="coerce")
        start_month = (qn - 1) * 3 + 1
        out["Aika_dt"] = pd.to_datetime(
            out["Vuosi_num"].astype("Int64").astype(str)
            + "-"
            + start_month.astype("Int64").astype(str).str.zfill(2)
            + "-01",
            errors="coerce",
        )

    elif year_match["y"].notna().any():
        out["Vuosi_num"] = pd.to_numeric(year_match["y"], errors="coerce")
        out["Aika_dt"] = pd.to_datetime(
            out["Vuosi_num"].astype("Int64").astype(str) + "-01-01",
            errors="coerce",
        )

    else:
        out["Aika_dt"] = pd.to_datetime(s, errors="coerce")

    return out


def add_quarter_date(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    time_col = None
    for col in out.columns:
        c = str(col).strip().lower()
        if (
            "vuosineljännes" in c
            or "vuosineljannes" in c
            or "quarter" in c
            or "timeperiod_q" in c
            or c == "aika"
            or c == "time"
        ):
            time_col = col
            break

    if time_col is None:
        return pd.DataFrame()

    q = out[time_col].astype(str).str.extract(r"(\d{4}Q[1-4])", expand=False)

    out["Date"] = pd.PeriodIndex(q.dropna(), freq="Q").to_timestamp(how="start")

    # Kohdistetaan takaisin alkuperäiseen indeksiin
    out["Date"] = pd.to_datetime(
        out[time_col]
        .astype(str)
        .str.extract(r"(\d{4}Q[1-4])", expand=False)
        .map(lambda x: pd.Period(x, freq="Q").to_timestamp(how="start") if pd.notna(x) else pd.NaT),
        errors="coerce",
    )

    out = out.dropna(subset=["Date"]).copy()

    return out



def find_time_code(meta: dict) -> str | None:
    variables = meta.get("variables") or []

    for var in variables:
        if var.get("time") is True:
            return var.get("code")
        if str(var.get("type", "")).lower() in {"t", "time"}:
            return var.get("code")

    for var in variables:
        code = str(var.get("code", "")).strip().lower()
        if code in {
            "kuukausi",
            "vuosineljännes",
            "vuosineljannes",
            "neljännes",
            "vuosi",
            "aika",
            "quarter",
            "time",
        }:
            return var.get("code")

    return variables[-1].get("code") if variables else None


def pick_value(
    meta: dict,
    var_code: str | None,
    want_contains_any: list[str],
    fallback_first: bool = True,
) -> str | None:
    if not var_code:
        return None

    variables = meta.get("variables") or []
    var = next(
        (
            v
            for v in variables
            if str(v.get("code", "")).strip().lower() == str(var_code).strip().lower()
        ),
        None,
    )
    if var is None:
        return None

    values = var.get("values") or []
    texts = var.get("valueTexts") or []

    if not values:
        return None

    if not texts or len(values) != len(texts):
        return values[0] if fallback_first else None

    wants = [w.strip().lower() for w in want_contains_any if w and w.strip()]
    for i, txt in enumerate(texts):
        if any(w in str(txt).lower() for w in wants):
            return values[i]

    return values[0] if fallback_first else None


def pick_value_no_fallback(meta: dict, var_code: str | None, want_contains_any: list[str]) -> str | None:
    return pick_value(meta, var_code, want_contains_any, fallback_first=False)



def merge_on_date(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = pd.DataFrame()

    for frame in frames:
        if frame is None or frame.empty:
            continue

        if out.empty:
            out = frame.copy()
        else:
            out = pd.merge(out, frame, on="Date", how="outer")

    return out.sort_values("Date").reset_index(drop=True) if not out.empty else pd.DataFrame()