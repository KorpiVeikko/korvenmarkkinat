from __future__ import annotations

import pandas as pd


def _pct(now: float | None, then: float | None) -> float | None:
    if now is None or then is None or then == 0 or pd.isna(now) or pd.isna(then):
        return None
    return (now / then - 1.0) * 100.0


def _status_from_pct(pct: float | None, positive_good: bool = True) -> tuple[str, str]:
    if pct is None or pd.isna(pct):
        return "⚪", "Ei dataa"

    v = pct if positive_good else -pct

    if v >= 8:
        return "🟢", "Vahva"
    if v >= 2:
        return "🟢", "Kasvava"
    if v > -2:
        return "🟡", "Vakaa"
    if v > -8:
        return "🟠", "Heikkenevä"
    return "🔴", "Heikko"


def _latest_and_yoy(df: pd.DataFrame, date_col: str, value_col: str, periods: int) -> tuple[float | None, float | None]:
    if df is None or df.empty:
        return None, None

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if d.empty:
        return None, None

    latest = float(d.iloc[-1][value_col])

    if len(d) <= periods:
        return latest, None

    prev = float(d.iloc[-(periods + 1)][value_col])
    return latest, _pct(latest, prev)


def _latest_yoy_col(df: pd.DataFrame, value_col: str = "Arvo", yoy_col: str = "YoY_pct") -> tuple[float | None, float | None]:
    if df is None or df.empty:
        return None, None

    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    if yoy_col in d.columns:
        d[yoy_col] = pd.to_numeric(d[yoy_col], errors="coerce")
    else:
        d[yoy_col] = pd.NA

    d = d.dropna(subset=[value_col])
    if d.empty:
        return None, None

    latest = float(d.iloc[-1][value_col])
    yoy = d.iloc[-1][yoy_col]
    return latest, float(yoy) if pd.notna(yoy) else None


def _construction_stage_yoy(construction_df: pd.DataFrame, stage: str) -> tuple[float | None, float | None]:
    if construction_df is None or construction_df.empty:
        return None, None

    d = construction_df[
        (construction_df["Alue"] == "Koko maa")
        & (construction_df["Vaihe"] == stage)
    ].copy()

    if d.empty or "Arvo_sum12" not in d.columns:
        return None, None

    return _latest_and_yoy(d, "Aika_dt", "Arvo_sum12", periods=12)


def _build_strengths_risks_watchlist(indicators: list[dict]) -> tuple[list[str], list[str], list[str]]:
    strengths: list[str] = []
    risks: list[str] = []
    watchlist: list[str] = []

    for item in indicators:
        name = item.get("Osa-alue", "")
        pct = item.get("Muutos")

        if pct is None or pd.isna(pct):
            watchlist.append(f"{name}: dataa kannattaa seurata, kun uusi havainto päivittyy.")
            continue

        if pct >= 5:
            strengths.append(f"{name}: kehitys on selvästi positiivinen ({pct:+.1f} %).")
        elif pct <= -5:
            risks.append(f"{name}: kehitys on selvästi negatiivinen ({pct:+.1f} %).")
        else:
            watchlist.append(f"{name}: tilanne on melko vakaa ({pct:+.1f} %).")

    if not strengths:
        strengths.append("Selviä vahvuuksia ei erottunut nykyisestä datasta.")

    if not risks:
        risks.append("Selviä riskisignaaleja ei erottunut nykyisestä datasta.")

    watchlist.append("Seuraa erityisesti, liikkuvatko kauppamäärät, hinnat ja rakennusluvat samaan suuntaan.")

    return strengths, risks, watchlist


def build_realestate_analysis_bundle(
    df_counts: pd.DataFrame,
    df_prices: pd.DataFrame,
    tontti_df: pd.DataFrame,
    construction_df: pd.DataFrame,
) -> dict:
    asunto_lkm_latest, asunto_lkm_yoy = _latest_yoy_col(df_counts)
    asunto_price_latest, asunto_price_yoy = _latest_yoy_col(df_prices)

    tontti_hinta_df = tontti_df[
        (tontti_df["Alue"] == "Koko maa")
        & (tontti_df["Tiedot"] == "Hintaindeksi")
    ].copy()

    tontti_real_df = tontti_df[
        (tontti_df["Alue"] == "Koko maa")
        & (tontti_df["Tiedot"] == "Reaalihintaindeksi")
    ].copy()

    tontti_lkm_df = tontti_df[tontti_df["Tiedot"] == "Kauppojen lukumäärä"].copy()
    if not tontti_lkm_df.empty:
        tontti_lkm_df = (
            tontti_lkm_df.groupby("Jakso_dt", as_index=False)["Arvo"]
            .sum()
            .sort_values("Jakso_dt")
        )

    tontti_hinta_latest, tontti_hinta_yoy = _latest_and_yoy(tontti_hinta_df, "Jakso_dt", "Arvo", periods=4)
    tontti_real_latest, tontti_real_yoy = _latest_and_yoy(tontti_real_df, "Jakso_dt", "Arvo", periods=4)
    tontti_lkm_latest, tontti_lkm_yoy = _latest_and_yoy(tontti_lkm_df, "Jakso_dt", "Arvo", periods=4)

    permits_latest, permits_yoy = _construction_stage_yoy(construction_df, "Rakennusluvat")
    completed_latest, completed_yoy = _construction_stage_yoy(construction_df, "Valmistuneet")

    indicators = [
        {
            "Osa-alue": "Asuntokaupat",
            "Muutos": asunto_lkm_yoy,
            "Ikoni": _status_from_pct(asunto_lkm_yoy)[0],
            "Tila": _status_from_pct(asunto_lkm_yoy)[1],
            "Selite": "Uusien asuntojen kauppamäärän vuosimuutos.",
        },
        {
            "Osa-alue": "Uusien asuntojen hinnat",
            "Muutos": asunto_price_yoy,
            "Ikoni": _status_from_pct(asunto_price_yoy)[0],
            "Tila": _status_from_pct(asunto_price_yoy)[1],
            "Selite": "Uusien asuntojen neliöhinnan vuosimuutos.",
        },
        {
            "Osa-alue": "Tonttien hintaindeksi",
            "Muutos": tontti_hinta_yoy,
            "Ikoni": _status_from_pct(tontti_hinta_yoy)[0],
            "Tila": _status_from_pct(tontti_hinta_yoy)[1],
            "Selite": "Omakotitalotonttien hintaindeksin vuosimuutos.",
        },
        {
            "Osa-alue": "Tonttikauppa",
            "Muutos": tontti_lkm_yoy,
            "Ikoni": _status_from_pct(tontti_lkm_yoy)[0],
            "Tila": _status_from_pct(tontti_lkm_yoy)[1],
            "Selite": "Omakotitalotonttien kauppamäärän vuosimuutos.",
        },
        {
            "Osa-alue": "Rakennusluvat",
            "Muutos": permits_yoy,
            "Ikoni": _status_from_pct(permits_yoy)[0],
            "Tila": _status_from_pct(permits_yoy)[1],
            "Selite": "Rakennuslupien 12 kk kertymän vuosimuutos.",
        },
        {
            "Osa-alue": "Valmistuneet asunnot",
            "Muutos": completed_yoy,
            "Ikoni": _status_from_pct(completed_yoy)[0],
            "Tila": _status_from_pct(completed_yoy)[1],
            "Selite": "Valmistuneiden asuntojen 12 kk kertymän vuosimuutos.",
        },
    ]

    usable_scores = []
    for item in indicators:
        pct = item["Muutos"]
        if pct is None or pd.isna(pct):
            continue
        if pct >= 8:
            usable_scores.append(2)
        elif pct >= 2:
            usable_scores.append(1)
        elif pct > -2:
            usable_scores.append(0)
        elif pct > -8:
            usable_scores.append(-1)
        else:
            usable_scores.append(-2)

    avg_score = sum(usable_scores) / len(usable_scores) if usable_scores else None

    if avg_score is None:
        cycle_label = "Ei riittävästi dataa"
        cycle_icon = "⚪"
    elif avg_score >= 1.0:
        cycle_label = "Vahva / kasvava vaihe"
        cycle_icon = "🟢"
    elif avg_score >= 0.2:
        cycle_label = "Lievä kasvu"
        cycle_icon = "🟢"
    elif avg_score > -0.4:
        cycle_label = "Vakaa / tasaantuva vaihe"
        cycle_icon = "🟡"
    elif avg_score > -1.0:
        cycle_label = "Hidastuva vaihe"
        cycle_icon = "🟠"
    else:
        cycle_label = "Heikko vaihe"
        cycle_icon = "🔴"

    summary_parts = []

    if asunto_lkm_yoy is not None:
        if asunto_lkm_yoy > 5:
            summary_parts.append("Asuntokauppa on piristynyt vuoden takaiseen verrattuna.")
        elif asunto_lkm_yoy < -5:
            summary_parts.append("Asuntokauppa on edelleen heikkoa, mikä kertoo varovaisesta kysynnästä.")
        else:
            summary_parts.append("Asuntokaupan määrä on melko vakaa.")

    if tontti_lkm_yoy is not None:
        if tontti_lkm_yoy > 5:
            summary_parts.append("Tonttikauppa näyttää piristyvän, mikä voi ennakoida rakentamisen kiinnostuksen vahvistumista.")
        elif tontti_lkm_yoy < -5:
            summary_parts.append("Tonttikauppa on vaimeaa, mikä kertoo rakentamisen ja omakotitalokysynnän varovaisuudesta.")

    if permits_yoy is not None:
        if permits_yoy > 5:
            summary_parts.append("Rakennuslupien kasvu viittaa tulevan tarjonnan mahdolliseen piristymiseen.")
        elif permits_yoy < -5:
            summary_parts.append("Rakennusluvat ovat laskussa, mikä kertoo rakentamisen putken heikkenemisestä.")
        else:
            summary_parts.append("Rakennuslupien kehitys on melko tasainen.")

    if tontti_real_yoy is not None:
        if tontti_real_yoy < -2:
            summary_parts.append("Tonttien reaalihinnat ovat paineessa, eli inflaatio huomioiden hintakehitys on heikkoa.")
        elif tontti_real_yoy > 2:
            summary_parts.append("Tonttien reaalihinnat ovat vahvistuneet.")

    if not summary_parts:
        summary_parts.append("Kiinteistömarkkinasta ei saatu riittävästi dataa analyysin muodostamiseen.")

    strengths, risks, watchlist = _build_strengths_risks_watchlist(indicators)

    return {
        "cycle_icon": cycle_icon,
        "cycle_label": cycle_label,
        "cycle_score": avg_score,
        "summary": " ".join(summary_parts),
        "indicators": indicators,
        "strengths": strengths,
        "risks": risks,
        "watchlist": watchlist,
        "metrics": {
            "asunto_lkm_latest": asunto_lkm_latest,
            "asunto_lkm_yoy": asunto_lkm_yoy,
            "asunto_price_latest": asunto_price_latest,
            "asunto_price_yoy": asunto_price_yoy,
            "tontti_hinta_latest": tontti_hinta_latest,
            "tontti_hinta_yoy": tontti_hinta_yoy,
            "tontti_real_latest": tontti_real_latest,
            "tontti_real_yoy": tontti_real_yoy,
            "tontti_lkm_latest": tontti_lkm_latest,
            "tontti_lkm_yoy": tontti_lkm_yoy,
            "permits_latest": permits_latest,
            "permits_yoy": permits_yoy,
            "completed_latest": completed_latest,
            "completed_yoy": completed_yoy,
        },
    }