from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from services import fuel_forecast_xgb as base
from services.diesel_forecast_calibration import build_diesel_calibration_frame

V7_CALIBRATION_WINDOW = 26
V7_MIN_HISTORY = 12
V7_VOLATILITY_WINDOW = 8
V7_VOLATILITY_REFERENCE_WINDOW = 52
V7_VOLATILITY_MIN_HISTORY = 26
V7_TREND_WINDOW = 8
V7_STRONG_RISE_THRESHOLD = 0.04
V7_STRONG_FALL_THRESHOLD = -0.04


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _direction_from_change_cents(change_cents: float | None) -> str:
    if change_cents is None or not np.isfinite(change_cents):
        return "Ei arviota"
    if change_cents >= 2.0:
        return "Nousupaine"
    if change_cents <= -2.0:
        return "Laskupaine"
    return "Melko vakaa"


def _confidence_from_context(*, interval_width_cents: float | None, volatility_regime: str) -> str:
    if interval_width_cents is None:
        return "Varovainen"
    if volatility_regime == "Korkea volatiliteetti" or interval_width_cents >= 24.0:
        return "Korkea epävarmuus"
    if interval_width_cents >= 16.0:
        return "Kohtalainen epävarmuus"
    return "Varovainen"


def _safe_regression_metrics(frame: pd.DataFrame, *, prediction_column: str) -> dict[str, Any]:
    temp = frame.copy()
    temp["PredictedPrice"] = pd.to_numeric(temp[prediction_column], errors="coerce")
    temp["PredictedChange"] = temp["PredictedPrice"] / temp["FuelPrice"] - 1.0
    temp = temp.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["FuelPrice", "ActualPrice", "ActualChange", "PredictedPrice", "PredictedChange"]
    )
    if temp.empty:
        return {}
    return base._regression_metrics(temp)


def add_causal_diesel_regimes(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["FuelPrice"] = pd.to_numeric(out["FuelPrice"], errors="coerce")
    out["FuelReturn_1w_Known"] = out["FuelPrice"].pct_change(1, fill_method=None)
    out["FuelMomentum_8w_Known"] = out["FuelPrice"].pct_change(V7_TREND_WINDOW, fill_method=None)
    out["FuelVolatility_8w_Known"] = out["FuelReturn_1w_Known"].rolling(
        V7_VOLATILITY_WINDOW, min_periods=V7_VOLATILITY_WINDOW
    ).std()
    prior_volatility = out["FuelVolatility_8w_Known"].shift(1)
    out["VolatilityLowThreshold"] = prior_volatility.rolling(
        V7_VOLATILITY_REFERENCE_WINDOW, min_periods=V7_VOLATILITY_MIN_HISTORY
    ).quantile(0.33)
    out["VolatilityHighThreshold"] = prior_volatility.rolling(
        V7_VOLATILITY_REFERENCE_WINDOW, min_periods=V7_VOLATILITY_MIN_HISTORY
    ).quantile(0.67)
    out["TrendRegime"] = np.select(
        [
            out["FuelMomentum_8w_Known"] >= V7_STRONG_RISE_THRESHOLD,
            out["FuelMomentum_8w_Known"] <= V7_STRONG_FALL_THRESHOLD,
        ],
        ["Voimakas nousu", "Voimakas lasku"],
        default="Sivuttainen / maltillinen",
    )
    out["VolatilityRegime"] = np.select(
        [
            out["VolatilityLowThreshold"].notna()
            & (out["FuelVolatility_8w_Known"] <= out["VolatilityLowThreshold"]),
            out["VolatilityHighThreshold"].notna()
            & (out["FuelVolatility_8w_Known"] >= out["VolatilityHighThreshold"]),
        ],
        ["Matala volatiliteetti", "Korkea volatiliteetti"],
        default="Keskimääräinen volatiliteetti",
    )
    return out


def apply_diesel_v7_gate(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    required = {"PredictedPrice", "PredictedPrice_BiasCorrectedBlend", "TrendRegime", "VolatilityRegime"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"Diesel V7 -gatingista puuttuvat sarakkeet: {sorted(missing)}")
    strong_fall = out["TrendRegime"] == "Voimakas lasku"
    strong_rise = out["TrendRegime"] == "Voimakas nousu"
    high_volatility = out["VolatilityRegime"] == "Korkea volatiliteetti"
    use_calibrated = ~strong_fall & (strong_rise | high_volatility)
    out["V7UseCalibrated"] = use_calibrated
    out["V7ModelChoice"] = np.where(use_calibrated, "Kalibroitu ennuste", "Perusennuste")
    out["PredictedPrice_V7"] = np.where(
        use_calibrated,
        out["PredictedPrice_BiasCorrectedBlend"],
        out["PredictedPrice"],
    )
    out["PredictedChange_V7"] = out["PredictedPrice_V7"] / out["FuelPrice"] - 1.0
    return out


def build_diesel_v7_backtest(
    diesel_backtest_df: pd.DataFrame,
    *,
    calibration_window: int = V7_CALIBRATION_WINDOW,
    min_history: int = V7_MIN_HISTORY,
) -> tuple[pd.DataFrame, str | None]:
    calibration_df, message = build_diesel_calibration_frame(
        diesel_backtest_df,
        calibration_window=calibration_window,
        min_history=min_history,
    )
    if calibration_df.empty:
        return pd.DataFrame(), message or "Diesel V7 -kalibrointidata jäi tyhjäksi."
    return apply_diesel_v7_gate(add_causal_diesel_regimes(calibration_df)), None


def _latest_calibration_state(
    diesel_backtest_df: pd.DataFrame,
    *,
    calibration_window: int = V7_CALIBRATION_WINDOW,
    min_history: int = V7_MIN_HISTORY,
) -> tuple[dict[str, float], str | None]:
    required = {"ActualPrice", "PredictedPrice", "FuelPrice"}
    if (
        diesel_backtest_df is None
        or diesel_backtest_df.empty
        or not required.issubset(diesel_backtest_df.columns)
    ):
        return {"bias_correction_eur": 0.0, "xgb_weight": 0.50}, "Dieselin kalibrointihistoria ei ollut käytettävissä."

    frame = diesel_backtest_df.copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required))
    if len(frame) < min_history:
        return {"bias_correction_eur": 0.0, "xgb_weight": 0.50}, f"Dieselin kalibrointihistoria on vielä lyhyt: {len(frame)} havaintoa."

    recent = frame.tail(calibration_window)
    residual = recent["ActualPrice"] - recent["PredictedPrice"]
    xgb_mae = float(residual.abs().mean())
    no_change_mae = float((recent["ActualPrice"] - recent["FuelPrice"]).abs().mean())
    denominator = xgb_mae + no_change_mae
    xgb_weight = 0.50 if denominator <= 0 or not np.isfinite(denominator) else no_change_mae / denominator
    return {
        "bias_correction_eur": float(residual.median()),
        "xgb_weight": float(np.clip(xgb_weight, 0.15, 0.85)),
    }, None


def _clean_current_diesel_history(weekly_fuel_df: pd.DataFrame) -> pd.DataFrame:
    required = {"Date", "Fuel", "Price_EUR_L"}
    if weekly_fuel_df is None or weekly_fuel_df.empty or not required.issubset(weekly_fuel_df.columns):
        return pd.DataFrame()
    out = weekly_fuel_df.loc[
        weekly_fuel_df["Fuel"].astype(str).str.strip() == "Diesel",
        ["Date", "Price_EUR_L"],
    ].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["FuelPrice"] = pd.to_numeric(out["Price_EUR_L"], errors="coerce")
    return (
        out.dropna(subset=["Date", "FuelPrice"])
        .loc[lambda frame: (frame["FuelPrice"] > 0) & (frame["FuelPrice"] < 10)]
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")[["Date", "FuelPrice"]]
        .reset_index(drop=True)
    )


def get_current_diesel_regime(weekly_fuel_df: pd.DataFrame) -> tuple[dict[str, Any], str | None]:
    history = _clean_current_diesel_history(weekly_fuel_df)
    if len(history) < V7_VOLATILITY_REFERENCE_WINDOW + V7_VOLATILITY_WINDOW + 2:
        return {}, f"Dieselin regime-historiaa on liian vähän: {len(history)} havaintoa."

    history["FuelReturn_1w"] = history["FuelPrice"].pct_change(1, fill_method=None)
    history["FuelMomentum_8w"] = history["FuelPrice"].pct_change(V7_TREND_WINDOW, fill_method=None)
    history["FuelVolatility_8w"] = history["FuelReturn_1w"].rolling(
        V7_VOLATILITY_WINDOW, min_periods=V7_VOLATILITY_WINDOW
    ).std()

    latest = history.iloc[-1]
    current_momentum = _safe_float(latest["FuelMomentum_8w"])
    current_volatility = _safe_float(latest["FuelVolatility_8w"])
    prior_volatility = history["FuelVolatility_8w"].iloc[:-1].dropna().tail(V7_VOLATILITY_REFERENCE_WINDOW)

    if current_momentum is None or current_volatility is None or len(prior_volatility) < V7_VOLATILITY_MIN_HISTORY:
        return {}, "Dieselin nykyistä regimeä ei voitu luokitella luotettavasti."

    low_threshold = float(prior_volatility.quantile(0.33))
    high_threshold = float(prior_volatility.quantile(0.67))

    if current_momentum >= V7_STRONG_RISE_THRESHOLD:
        trend_regime = "Voimakas nousu"
    elif current_momentum <= V7_STRONG_FALL_THRESHOLD:
        trend_regime = "Voimakas lasku"
    else:
        trend_regime = "Sivuttainen / maltillinen"

    if current_volatility <= low_threshold:
        volatility_regime = "Matala volatiliteetti"
    elif current_volatility >= high_threshold:
        volatility_regime = "Korkea volatiliteetti"
    else:
        volatility_regime = "Keskimääräinen volatiliteetti"

    return {
        "date": pd.to_datetime(latest["Date"]),
        "trend_regime": trend_regime,
        "volatility_regime": volatility_regime,
        "momentum_8w": current_momentum,
        "volatility_8w": current_volatility,
    }, None


def calculate_diesel_v7_forecast(
    *,
    v6_summary: dict[str, Any],
    diesel_backtest_df: pd.DataFrame,
    weekly_fuel_df: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, str | None]:
    if not v6_summary:
        return {}, pd.DataFrame(), "Diesel V6 -pohjaennuste puuttuu."

    latest_price = _safe_float(v6_summary.get("latest_fuel_price"))
    base_predicted_price = _safe_float(v6_summary.get("predicted_price"))
    if latest_price is None or base_predicted_price is None or latest_price <= 0:
        return {}, pd.DataFrame(), "Diesel V6 -pohjaennusteen hintatiedot ovat puutteelliset."

    v7_backtest, backtest_message = build_diesel_v7_backtest(diesel_backtest_df)
    calibration_state, calibration_message = _latest_calibration_state(diesel_backtest_df)
    regime, regime_message = get_current_diesel_regime(weekly_fuel_df)
    if not regime:
        return {}, v7_backtest, regime_message or "Dieselin nykyistä regimeä ei voitu muodostaa."

    bias_correction = float(calibration_state["bias_correction_eur"])
    xgb_weight = float(calibration_state["xgb_weight"])
    bias_corrected_price = base_predicted_price + bias_correction
    calibrated_blend_price = xgb_weight * bias_corrected_price + (1.0 - xgb_weight) * latest_price

    strong_fall = regime["trend_regime"] == "Voimakas lasku"
    strong_rise = regime["trend_regime"] == "Voimakas nousu"
    high_volatility = regime["volatility_regime"] == "Korkea volatiliteetti"
    use_calibrated = (not strong_fall) and (strong_rise or high_volatility)

    predicted_price = calibrated_blend_price if use_calibrated else base_predicted_price
    predicted_change = predicted_price / latest_price - 1.0
    predicted_change_cents = (predicted_price - latest_price) * 100.0

    base_interval_low = _safe_float(v6_summary.get("interval_low_price"))
    base_interval_high = _safe_float(v6_summary.get("interval_high_price"))
    interval_low_price = interval_high_price = interval_width_cents = None
    if base_interval_low is not None and base_interval_high is not None:
        lower_distance = max(0.0, base_predicted_price - base_interval_low)
        upper_distance = max(0.0, base_interval_high - base_predicted_price)
        interval_low_price = max(0.0, predicted_price - lower_distance)
        interval_high_price = predicted_price + upper_distance
        interval_width_cents = (interval_high_price - interval_low_price) * 100.0

    historical_metrics = {}
    if v7_backtest is not None and not v7_backtest.empty:
        historical_metrics = _safe_regression_metrics(v7_backtest, prediction_column="PredictedPrice_V7")

    summary = dict(v6_summary)
    summary.update({
        "model_version": "Diesel V7 – lukittu regime-malli",
        "predicted_price": predicted_price,
        "predicted_change_pct": predicted_change * 100.0,
        "predicted_change_cents": predicted_change_cents,
        "interval_low_price": interval_low_price,
        "interval_high_price": interval_high_price,
        "direction": _direction_from_change_cents(predicted_change_cents),
        "confidence": _confidence_from_context(
            interval_width_cents=interval_width_cents,
            volatility_regime=regime["volatility_regime"],
        ),
        "trend_regime": regime["trend_regime"],
        "volatility_regime": regime["volatility_regime"],
        "regime_date": regime["date"],
        "v7_use_calibrated": use_calibrated,
        "v7_model_choice": "Kalibroitu ennuste" if use_calibrated else "Perusennuste",
        "v7_bias_correction_eur": bias_correction,
        "v7_xgb_weight": xgb_weight,
        "v7_base_predicted_price": base_predicted_price,
        "v7_calibrated_predicted_price": calibrated_blend_price,
        "walk_forward_mae_cents": historical_metrics.get("mae_cents", v6_summary.get("walk_forward_mae_cents")),
        "walk_forward_rmse_cents": historical_metrics.get("rmse_cents", v6_summary.get("walk_forward_rmse_cents")),
        "balanced_accuracy": historical_metrics.get("balanced_accuracy"),
        "macro_f1": historical_metrics.get("macro_f1"),
        "r_squared": historical_metrics.get("r_squared"),
        "test_observations": int(historical_metrics.get("observations", len(v7_backtest) if v7_backtest is not None else 0)),
    })

    messages = [m for m in (backtest_message, calibration_message, regime_message) if m]
    return summary, v7_backtest, " | ".join(messages) if messages else None