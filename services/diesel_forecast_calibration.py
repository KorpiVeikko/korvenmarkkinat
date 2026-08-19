from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from services import fuel_forecast_xgb as base


DEFAULT_CALIBRATION_WINDOW = 26
DEFAULT_MIN_HISTORY = 12


def _safe_float(
    value: Any,
) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(number):
        return None

    return number


def _build_metric_row(
    label: str,
    frame: pd.DataFrame,
    *,
    prediction_column: str,
) -> dict[str, Any]:
    temp = frame.copy()

    temp["PredictedPrice"] = pd.to_numeric(
        temp[prediction_column],
        errors="coerce",
    )

    temp["PredictedChange"] = (
        temp["PredictedPrice"]
        / temp["FuelPrice"]
        - 1.0
    )

    temp = temp.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(
        subset=[
            "FuelPrice",
            "ActualPrice",
            "ActualChange",
            "PredictedPrice",
            "PredictedChange",
        ]
    )

    metrics = base._regression_metrics(
        temp
    )

    return {
        "Malli": label,
        "MAE_snt_l": metrics.get(
            "mae_cents"
        ),
        "RMSE_snt_l": metrics.get(
            "rmse_cents"
        ),
        "Suuntatarkkuus": metrics.get(
            "direction_accuracy"
        ),
        "Tasapainotettu_tarkkuus": metrics.get(
            "balanced_accuracy"
        ),
        "Macro_F1": metrics.get(
            "macro_f1"
        ),
        "R2": metrics.get(
            "r_squared"
        ),
        "Havaintoja": metrics.get(
            "observations"
        ),
    }


def _rolling_bias_correction(
    backtest_df: pd.DataFrame,
    *,
    window: int,
    min_history: int,
) -> pd.Series:
    """
    Laskee ennustehetkellä tunnetuista aiemmista virheistä
    robustin mediaanikorjauksen.

    Residuaali:
        toteutunut - ennuste

    Positiivinen korjaus nostaa ennustetta.
    Negatiivinen korjaus laskee ennustetta.

    shift(1) estää nykyisen toteuman vuotamisen korjaukseen.
    """
    residual_eur = (
        pd.to_numeric(
            backtest_df["ActualPrice"],
            errors="coerce",
        )
        - pd.to_numeric(
            backtest_df["PredictedPrice"],
            errors="coerce",
        )
    )

    correction = (
        residual_eur.shift(1)
        .rolling(
            window=window,
            min_periods=min_history,
        )
        .median()
    )

    return correction.fillna(0.0)


def _rolling_inverse_error_weight(
    backtest_df: pd.DataFrame,
    *,
    window: int,
    min_history: int,
) -> pd.Series:
    """
    Laskee vain aiemmista toteumista dynaamisen painon
    XGBoost-ennusteelle suhteessa ei muutosta -vertailuun.

    Paino 1 = käytä XGBoostia.
    Paino 0 = käytä nykyistä hintaa.
    """
    actual = pd.to_numeric(
        backtest_df["ActualPrice"],
        errors="coerce",
    )

    xgb_prediction = pd.to_numeric(
        backtest_df["PredictedPrice"],
        errors="coerce",
    )

    no_change_prediction = pd.to_numeric(
        backtest_df["FuelPrice"],
        errors="coerce",
    )

    xgb_error = (
        actual
        - xgb_prediction
    ).abs()

    no_change_error = (
        actual
        - no_change_prediction
    ).abs()

    rolling_xgb_mae = (
        xgb_error.shift(1)
        .rolling(
            window=window,
            min_periods=min_history,
        )
        .mean()
    )

    rolling_no_change_mae = (
        no_change_error.shift(1)
        .rolling(
            window=window,
            min_periods=min_history,
        )
        .mean()
    )

    denominator = (
        rolling_xgb_mae
        + rolling_no_change_mae
    )

    weight = (
        rolling_no_change_mae
        / denominator
    )

    return (
        weight.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.50)
        .clip(
            lower=0.15,
            upper=0.85,
        )
    )


def build_diesel_calibration_frame(
    backtest_df: pd.DataFrame,
    *,
    calibration_window: int = DEFAULT_CALIBRATION_WINDOW,
    min_history: int = DEFAULT_MIN_HISTORY,
) -> tuple[pd.DataFrame, str | None]:
    required = {
        "Date",
        "ForecastDate",
        "Fuel",
        "FuelPrice",
        "ActualPrice",
        "ActualChange",
        "PredictedPrice",
        "PredictedChange",
    }

    if (
        backtest_df is None
        or backtest_df.empty
        or not required.issubset(
            backtest_df.columns
        )
    ):
        missing = (
            required.difference(
                backtest_df.columns
            )
            if backtest_df is not None
            else required
        )

        return (
            pd.DataFrame(),
            (
                "Kalibrointiin tarvittavia sarakkeita puuttuu: "
                f"{sorted(missing)}"
            ),
        )

    if calibration_window < 4:
        return (
            pd.DataFrame(),
            "Kalibrointi-ikkunan pitää olla vähintään 4 havaintoa.",
        )

    if min_history < 3:
        return (
            pd.DataFrame(),
            "Kalibroinnin minimi-historian pitää olla vähintään 3.",
        )

    frame = backtest_df.copy()

    frame["Date"] = pd.to_datetime(
        frame["Date"],
        errors="coerce",
    )

    frame["ForecastDate"] = pd.to_datetime(
        frame["ForecastDate"],
        errors="coerce",
    )

    numeric_columns = [
        "FuelPrice",
        "ActualPrice",
        "ActualChange",
        "PredictedPrice",
        "PredictedChange",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = (
        frame.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[
                "Date",
                "ForecastDate",
                *numeric_columns,
            ]
        )
        .sort_values(
            "ForecastDate"
        )
        .reset_index(drop=True)
    )

    if len(frame) < (
        min_history + 10
    ):
        return (
            pd.DataFrame(),
            (
                "Kalibrointiin jäi liian vähän havaintoja: "
                f"{len(frame)}."
            ),
        )

    frame["BiasCorrectionEUR"] = (
        _rolling_bias_correction(
            frame,
            window=calibration_window,
            min_history=min_history,
        )
    )

    frame["PredictedPrice_BiasCorrected"] = (
        frame["PredictedPrice"]
        + frame["BiasCorrectionEUR"]
    )

    frame["AdaptiveXGBWeight"] = (
        _rolling_inverse_error_weight(
            frame,
            window=calibration_window,
            min_history=min_history,
        )
    )

    frame["PredictedPrice_AdaptiveBlend"] = (
        frame["AdaptiveXGBWeight"]
        * frame["PredictedPrice"]
        + (
            1.0
            - frame["AdaptiveXGBWeight"]
        )
        * frame["FuelPrice"]
    )

    frame[
        "PredictedPrice_BiasCorrectedBlend"
    ] = (
        frame["AdaptiveXGBWeight"]
        * frame[
            "PredictedPrice_BiasCorrected"
        ]
        + (
            1.0
            - frame["AdaptiveXGBWeight"]
        )
        * frame["FuelPrice"]
    )

    return frame, None


def run_diesel_calibration_experiment(
    backtest_df: pd.DataFrame,
    *,
    calibration_window: int = DEFAULT_CALIBRATION_WINDOW,
    min_history: int = DEFAULT_MIN_HISTORY,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    str | None,
]:
    frame, message = (
        build_diesel_calibration_frame(
            backtest_df,
            calibration_window=calibration_window,
            min_history=min_history,
        )
    )

    if frame.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            message,
        )

    model_columns = {
        "Diesel V6": "PredictedPrice",
        "V6 + rolling bias -korjaus": (
            "PredictedPrice_BiasCorrected"
        ),
        "V6 + adaptiivinen ensemble": (
            "PredictedPrice_AdaptiveBlend"
        ),
        "V6 + bias + ensemble": (
            "PredictedPrice_BiasCorrectedBlend"
        ),
    }

    rows = [
        _build_metric_row(
            label,
            frame,
            prediction_column=column,
        )
        for label, column in model_columns.items()
    ]

    result_df = pd.DataFrame(
        rows
    )

    if result_df.empty:
        return (
            pd.DataFrame(),
            frame,
            "Kalibrointikokeilu ei tuottanut tuloksia.",
        )

    base_rows = result_df.loc[
        result_df["Malli"]
        == "Diesel V6"
    ]

    if base_rows.empty:
        return (
            result_df,
            frame,
            "Vertailusta puuttuu Diesel V6 -perusmalli.",
        )

    base_row = base_rows.iloc[0]

    result_df["MAE_parannus_vs_V6"] = (
        float(
            base_row["MAE_snt_l"]
        )
        - result_df["MAE_snt_l"]
    )

    result_df[
        "Balanced_parannus_pp_vs_V6"
    ] = (
        result_df[
            "Tasapainotettu_tarkkuus"
        ]
        - float(
            base_row[
                "Tasapainotettu_tarkkuus"
            ]
        )
    ) * 100.0

    result_df = (
        result_df.sort_values(
            [
                "MAE_snt_l",
                "Tasapainotettu_tarkkuus",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return (
        result_df,
        frame,
        None,
    )