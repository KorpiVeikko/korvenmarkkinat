
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier, XGBRegressor

DEFAULT_FORECAST_HORIZON_WEEKS = 4
DEFAULT_TRAIN_SHARE = 0.80
DEFAULT_MIN_TRAIN_OBSERVATIONS = 150
DEFAULT_REFIT_EVERY = 4
DIRECTION_NEUTRAL_LIMIT = 0.005
MAX_ABSOLUTE_FORECAST_CHANGE = 0.15
CANDIDATE_LAG_WEEKS = tuple(range(0, 9))
PRODUCTION_LAG_WEEKS = tuple(range(0, 5))
LAG_WEEKS = CANDIDATE_LAG_WEEKS

DIRECTION_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_TO_DIRECTION = {0: -1, 1: 0, 2: 1}

FUEL_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    fuel: {
        "feature_columns": (
            "Brent_Return_1w_Lag_0w", "Brent_Return_1w_Lag_1w",
            "Brent_Return_1w_Lag_2w", "Brent_Return_1w_Lag_3w",
            "Brent_Return_1w_Lag_4w", "Fuel_Return_1w_Lag_0w",
            "Fuel_Return_1w_Lag_1w", "Fuel_Return_1w_Lag_2w",
            "Fuel_Return_1w_Lag_3w", "Fuel_Return_1w_Lag_4w",
            "Brent_Momentum_4w", "Brent_Momentum_8w",
            "Fuel_Momentum_4w", "Fuel_Momentum_8w",
            "Brent_Volatility_4w", "Fuel_Volatility_4w",
            "Month_Sin", "Month_Cos",
        ),
        "params": {
            "n_estimators": 300,
            "learning_rate": 0.025,
            "max_depth": 2,
            "min_child_weight": 8.0,
            "subsample": 0.80,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.0,
            "reg_lambda": 5.0,
            "gamma": 0.0,
        },
    }
    for fuel in ("95E10", "Diesel")
}

FEATURE_COLUMNS = list(FUEL_MODEL_CONFIGS["95E10"]["feature_columns"])
FEATURE_LABELS = {
    **{f"Brent_Return_1w_Lag_{lag}w": (
        "Brentin 1 vk muutos" if lag == 0 else f"Brentin 1 vk muutos, viive {lag} vk"
    ) for lag in CANDIDATE_LAG_WEEKS},
    **{f"Fuel_Return_1w_Lag_{lag}w": (
        "Polttoaineen 1 vk muutos" if lag == 0 else f"Polttoaineen 1 vk muutos, viive {lag} vk"
    ) for lag in CANDIDATE_LAG_WEEKS},
    "Brent_Momentum_4w": "Brentin 4 vk momentum",
    "Brent_Momentum_8w": "Brentin 8 vk momentum",
    "Brent_Volatility_4w": "Brentin volatiliteetti, 4 vk",
    "Brent_Volatility_8w": "Brentin volatiliteetti, 8 vk",
    "Fuel_Momentum_4w": "Polttoaineen 4 vk momentum",
    "Fuel_Momentum_8w": "Polttoaineen 8 vk momentum",
    "Fuel_Volatility_4w": "Polttoaineen volatiliteetti, 4 vk",
    "Fuel_Volatility_8w": "Polttoaineen volatiliteetti, 8 vk",
    "Month_Sin": "Historiallinen kausivaihtelu 1",
    "Month_Cos": "Historiallinen kausivaihtelu 2",
}


def _get_fuel_model_config(fuel_name: str) -> dict[str, Any]:
    if fuel_name not in FUEL_MODEL_CONFIGS:
        raise ValueError(f"Tuntematon polttoaineen malliasetus: {fuel_name}")
    return FUEL_MODEL_CONFIGS[fuel_name]


def _clean_market_weekly(df: pd.DataFrame, value_col: str, output_col: str) -> pd.DataFrame:
    if df is None or df.empty or not {"Date", value_col}.issubset(df.columns):
        return pd.DataFrame()
    out = df[["Date", value_col]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = (out.dropna(subset=["Date", value_col])
           .loc[lambda x: x[value_col] > 0]
           .sort_values("Date")
           .drop_duplicates("Date", keep="last"))
    if out.empty:
        return pd.DataFrame()
    return (out.set_index("Date")[value_col].resample("W-MON").mean()
            .dropna().rename(output_col).reset_index())


def _clean_fuel_weekly(fuel_df: pd.DataFrame, fuel_name: str) -> pd.DataFrame:
    required = {"Date", "Fuel", "Price_EUR_L"}
    if fuel_df is None or fuel_df.empty or not required.issubset(fuel_df.columns):
        return pd.DataFrame()
    out = fuel_df.loc[
        fuel_df["Fuel"].astype(str).str.strip() == fuel_name,
        ["Date", "Price_EUR_L"],
    ].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Price_EUR_L"] = pd.to_numeric(out["Price_EUR_L"], errors="coerce")
    return (out.dropna(subset=["Date", "Price_EUR_L"])
            .loc[lambda x: (x["Price_EUR_L"] > 0) & (x["Price_EUR_L"] < 10)]
            .sort_values("Date").drop_duplicates("Date", keep="last")
            .rename(columns={"Price_EUR_L": "FuelPrice"}).reset_index(drop=True))


def _add_lag_features(frame: pd.DataFrame, source: str, lags: tuple[int, ...]) -> None:
    for lag in lags:
        frame[f"{source}_Lag_{lag}w"] = frame[source].shift(lag)


def _add_candidate_features(merged: pd.DataFrame) -> None:
    merged["Brent_Return_1w"] = merged["Brent_USD"].pct_change(fill_method=None)
    merged["Fuel_Return_1w"] = merged["FuelPrice"].pct_change(fill_method=None)
    _add_lag_features(merged, "Brent_Return_1w", CANDIDATE_LAG_WEEKS)
    _add_lag_features(merged, "Fuel_Return_1w", CANDIDATE_LAG_WEEKS)
    for window in (4, 8):
        merged[f"Brent_Momentum_{window}w"] = merged["Brent_USD"].pct_change(window, fill_method=None)
        merged[f"Fuel_Momentum_{window}w"] = merged["FuelPrice"].pct_change(window, fill_method=None)
        merged[f"Brent_Volatility_{window}w"] = merged["Brent_Return_1w"].rolling(window, min_periods=window).std()
        merged[f"Fuel_Volatility_{window}w"] = merged["Fuel_Return_1w"].rolling(window, min_periods=window).std()
    angle = 2.0 * math.pi * merged["Date"].dt.month / 12.0
    merged["Month_Sin"] = np.sin(angle)
    merged["Month_Cos"] = np.cos(angle)


def build_fuel_feature_frame_xgb(
    brent_df: pd.DataFrame,
    eurusd_df: pd.DataFrame | None,
    weekly_fuel_df: pd.DataFrame,
    us_inventory_df: pd.DataFrame | None,
    crack_df: pd.DataFrame | None,
    fuel_name: str,
    *,
    forecast_horizon_weeks: int = DEFAULT_FORECAST_HORIZON_WEEKS,
) -> tuple[pd.DataFrame, str | None]:
    del eurusd_df, us_inventory_df, crack_df
    if forecast_horizon_weeks < 1:
        return pd.DataFrame(), "Ennustehorisontin pitää olla vähintään yksi viikko."
    if fuel_name not in FUEL_MODEL_CONFIGS:
        return pd.DataFrame(), f"Tuntematon polttoaine: {fuel_name}"
    brent = _clean_market_weekly(brent_df, "Close", "Brent_USD")
    fuel = _clean_fuel_weekly(weekly_fuel_df, fuel_name)
    if brent.empty:
        return pd.DataFrame(), "Brentin viikkosarjaa ei voitu muodostaa."
    if fuel.empty:
        return pd.DataFrame(), f"{fuel_name}-viikkosarjaa ei voitu muodostaa."
    merged = pd.merge(fuel, brent, on="Date", how="inner").sort_values("Date")
    if merged.empty:
        return pd.DataFrame(), f"{fuel_name}-hinnan ja Brent-datan välille ei löytynyt yhteisiä viikkoja."
    _add_candidate_features(merged)
    merged["ForecastDate"] = merged["Date"] + pd.to_timedelta(forecast_horizon_weeks * 7, unit="D")
    merged["Fuel_Target_Price"] = merged["FuelPrice"].shift(-forecast_horizon_weeks)
    merged["Fuel_Target_Change"] = merged["Fuel_Target_Price"] / merged["FuelPrice"] - 1.0
    merged["Baseline_NoChange"] = 0.0
    merged["Baseline_FuelTrend"] = (
        merged["FuelPrice"] / merged["FuelPrice"].shift(forecast_horizon_weeks) - 1.0
    ).clip(-MAX_ABSOLUTE_FORECAST_CHANGE, MAX_ABSOLUTE_FORECAST_CHANGE)
    merged["Baseline_BrentTrend"] = (
        merged["Brent_USD"] / merged["Brent_USD"].shift(forecast_horizon_weeks) - 1.0
    ).clip(-MAX_ABSOLUTE_FORECAST_CHANGE, MAX_ABSOLUTE_FORECAST_CHANGE)
    merged["Fuel"] = fuel_name
    candidate_features = [
        *[f"Brent_Return_1w_Lag_{lag}w" for lag in CANDIDATE_LAG_WEEKS],
        *[f"Fuel_Return_1w_Lag_{lag}w" for lag in CANDIDATE_LAG_WEEKS],
        "Brent_Momentum_4w", "Brent_Momentum_8w",
        "Brent_Volatility_4w", "Brent_Volatility_8w",
        "Fuel_Momentum_4w", "Fuel_Momentum_8w",
        "Fuel_Volatility_4w", "Fuel_Volatility_8w",
        "Month_Sin", "Month_Cos",
    ]
    cols = [
        "Date", "ForecastDate", "Fuel", "FuelPrice", "Fuel_Target_Price",
        "Fuel_Target_Change", "Brent_USD", "Baseline_NoChange",
        "Baseline_FuelTrend", "Baseline_BrentTrend", *candidate_features,
    ]
    out = (merged[cols].replace([np.inf, -np.inf], np.nan)
           .dropna(subset=["Date", "FuelPrice", *candidate_features])
           .sort_values("Date").reset_index(drop=True))
    if out.empty:
        return pd.DataFrame(), "XGBoost V6 -featuretaulukko jäi tyhjäksi."
    return out, None


def _create_direction_model_for_fuel(fuel_name: str, *, random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        tree_method="hist", n_jobs=1, random_state=random_state, verbosity=0,
        **_get_fuel_model_config(fuel_name)["params"],
    )


def _create_magnitude_model_for_fuel(fuel_name: str, *, random_state: int = 142) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:absoluteerror", tree_method="hist", n_jobs=1,
        random_state=random_state, verbosity=0,
        **_get_fuel_model_config(fuel_name)["params"],
    )


def _create_direction_model(*, random_state: int = 42) -> XGBClassifier:
    return _create_direction_model_for_fuel("95E10", random_state=random_state)


def _create_magnitude_model(*, random_state: int = 142) -> XGBRegressor:
    return _create_magnitude_model_for_fuel("95E10", random_state=random_state)


def _direction_from_change(values: np.ndarray, *, neutral_limit: float = DIRECTION_NEUTRAL_LIMIT) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(values > neutral_limit, 1, np.where(values < -neutral_limit, -1, 0))


def _classify_direction(values: np.ndarray, *, neutral_limit: float = DIRECTION_NEUTRAL_LIMIT) -> np.ndarray:
    return _direction_from_change(values, neutral_limit=neutral_limit)


def _direction_to_class(values: np.ndarray) -> np.ndarray:
    return np.array([DIRECTION_TO_CLASS[int(x)] for x in _direction_from_change(values)], dtype=int)


def _balanced_sample_weights(classes: np.ndarray) -> np.ndarray:
    classes = np.asarray(classes, dtype=int)
    unique, counts = np.unique(classes, return_counts=True)
    total = len(classes)
    weights = {int(c): total / (max(len(unique), 1) * int(n)) for c, n in zip(unique, counts) if n > 0}
    return np.array([weights.get(int(c), 1.0) for c in classes], dtype=float)


def _combine_stage_predictions(class_probabilities: np.ndarray, magnitude: float) -> tuple[float, int, float]:
    p = np.asarray(class_probabilities, dtype=float)
    if p.shape != (3,):
        raise ValueError("Suuntamallin pitää palauttaa kolme luokkatodennäköisyyttä.")
    predicted_class = int(np.argmax(p))
    direction = int(CLASS_TO_DIRECTION[predicted_class])
    confidence = float(p[predicted_class])
    expected_direction = float(p[2] - p[0])
    magnitude = float(np.clip(magnitude, 0.0, MAX_ABSOLUTE_FORECAST_CHANGE))
    change = float(np.clip(expected_direction * magnitude, -MAX_ABSOLUTE_FORECAST_CHANGE, MAX_ABSOLUTE_FORECAST_CHANGE))
    return change, direction, confidence


def _classification_metrics(actual_change: np.ndarray, predicted_change: np.ndarray) -> dict[str, Any]:
    actual = _classify_direction(actual_change)
    predicted = _classify_direction(predicted_change)
    labels = np.array([-1, 0, 1], dtype=int)
    balanced = float(balanced_accuracy_score(actual, predicted))
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=labels, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=labels, average="macro", zero_division=0
    )
    names = ("down", "flat", "up")
    class_metrics = {
        name: {
            "precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "support": int(support[i]),
        }
        for i, name in enumerate(names)
    }
    return {
        "balanced_accuracy": balanced,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "class_metrics": class_metrics,
    }


def _regression_metrics(backtest_df: pd.DataFrame) -> dict[str, Any]:
    if backtest_df is None or backtest_df.empty:
        return {}
    actual_change = backtest_df["ActualChange"].to_numpy(float)
    predicted_change = backtest_df["PredictedChange"].to_numpy(float)
    actual_price = backtest_df["ActualPrice"].to_numpy(float)
    predicted_price = backtest_df["PredictedPrice"].to_numpy(float)
    residual = (actual_price - predicted_price) * 100.0
    actual_dir = _classify_direction(actual_change)
    predicted_dir = _classify_direction(predicted_change)
    active = actual_dir != 0
    ss_res = float(np.sum((actual_change - predicted_change) ** 2))
    ss_total = float(np.sum((actual_change - np.mean(actual_change)) ** 2))
    return {
        "mae_cents": float(np.mean(np.abs(residual))),
        "rmse_cents": float(np.sqrt(np.mean(residual ** 2))),
        "median_error_cents": float(np.median(np.abs(residual))),
        "direction_accuracy": float(np.mean(actual_dir == predicted_dir)),
        "direction_accuracy_active": float(np.mean(actual_dir[active] == predicted_dir[active])) if np.any(active) else None,
        "r_squared": 1.0 - ss_res / ss_total if ss_total > 0 else None,
        "residual_std_cents": float(np.std(residual, ddof=1)) if len(residual) > 1 else None,
        **_classification_metrics(actual_change, predicted_change),
        "observations": int(len(backtest_df)),
    }


def build_direction_confusion_matrix(backtest_df: pd.DataFrame, *, prediction_column: str = "PredictedChange") -> pd.DataFrame:
    required = {"ActualChange", prediction_column}
    if backtest_df is None or backtest_df.empty or not required.issubset(backtest_df.columns):
        return pd.DataFrame()
    matrix = confusion_matrix(
        _classify_direction(backtest_df["ActualChange"].to_numpy(float)),
        _classify_direction(backtest_df[prediction_column].to_numpy(float)),
        labels=[-1, 0, 1],
    )
    return pd.DataFrame(
        matrix,
        index=["Todellinen: lasku", "Todellinen: vakaa", "Todellinen: nousu"],
        columns=["Ennuste: lasku", "Ennuste: vakaa", "Ennuste: nousu"],
    )


def build_model_comparison_table(backtest_df: pd.DataFrame) -> pd.DataFrame:
    if backtest_df is None or backtest_df.empty:
        return pd.DataFrame()
    models = {
        "XGBoost": "PredictedChange",
        "Ei muutosta": "BaselineNoChange",
        "Polttoainetrendi": "BaselineFuelTrend",
        "Brent-trendi": "BaselineBrentTrend",
    }
    rows = []
    for label, col in models.items():
        if col not in backtest_df.columns:
            continue
        temp = backtest_df.copy()
        temp["PredictedChange"] = temp[col]
        temp["PredictedPrice"] = temp["FuelPrice"] * (1.0 + temp[col])
        m = _regression_metrics(temp)
        rows.append({
            "Malli": label,
            "MAE_snt_l": m.get("mae_cents"),
            "RMSE_snt_l": m.get("rmse_cents"),
            "Suuntatarkkuus": m.get("direction_accuracy"),
            "Tasapainotettu_tarkkuus": m.get("balanced_accuracy"),
            "Macro_F1": m.get("macro_f1"),
            "Havaintoja": m.get("observations"),
        })
    return pd.DataFrame(rows)


def _walk_forward_backtest_xgb(
    model_df: pd.DataFrame,
    *,
    initial_train_size: int,
    forecast_horizon_weeks: int,
    refit_every: int = DEFAULT_REFIT_EVERY,
    feature_columns: list[str] | tuple[str, ...] | None = None,
    fuel_name: str | None = None,
) -> tuple[pd.DataFrame, str | None]:
    if feature_columns is None:
        feature_columns = (
            list(_get_fuel_model_config(fuel_name)["feature_columns"])
            if fuel_name is not None else FEATURE_COLUMNS
        )
    feature_columns = list(feature_columns)
    missing = [c for c in feature_columns if c not in model_df.columns]
    if missing:
        return pd.DataFrame(), f"Walk-forward-testistä puuttuvat featuret: {missing}"
    if len(model_df) <= initial_train_size:
        return pd.DataFrame(), "XGBoost-testille ei jäänyt testihavaintoja."
    inferred_fuel = fuel_name or str(model_df["Fuel"].dropna().iloc[0])
    rows = []
    direction_model = None
    magnitude_model = None
    since_refit = refit_every
    for index in range(initial_train_size, len(model_df)):
        row = model_df.iloc[[index]]
        prediction_date = pd.to_datetime(row.iloc[0]["Date"])
        train = model_df.iloc[:index].loc[
            lambda frame: pd.to_datetime(frame["ForecastDate"]) <= prediction_date
        ].copy()
        if len(train) < max(1, initial_train_size - forecast_horizon_weeks):
            continue
        if direction_model is None or magnitude_model is None or since_refit >= refit_every:
            x_train = train[feature_columns]
            y_change = train["Fuel_Target_Change"].to_numpy(float)
            y_direction = _direction_to_class(y_change)
            if len(np.unique(y_direction)) < 3:
                continue
            direction_model = _create_direction_model_for_fuel(inferred_fuel, random_state=42 + index)
            magnitude_model = _create_magnitude_model_for_fuel(inferred_fuel, random_state=142 + index)
            direction_model.fit(x_train, y_direction, sample_weight=_balanced_sample_weights(y_direction))
            magnitude_model.fit(x_train, np.abs(y_change))
            since_refit = 0
        x_row = row[feature_columns]
        probabilities = direction_model.predict_proba(x_row)[0]
        magnitude = float(magnitude_model.predict(x_row)[0])
        predicted_change, predicted_direction, confidence = _combine_stage_predictions(probabilities, magnitude)
        current_price = float(row.iloc[0]["FuelPrice"])
        actual_change = float(row.iloc[0]["Fuel_Target_Change"])
        actual_price = float(row.iloc[0]["Fuel_Target_Price"])
        predicted_price = current_price * (1.0 + predicted_change)
        rows.append({
            "Date": row.iloc[0]["Date"],
            "ForecastDate": row.iloc[0]["ForecastDate"],
            "Fuel": row.iloc[0]["Fuel"],
            "FuelPrice": current_price,
            "TrainObservations": int(len(train)),
            "FeatureCount": len(feature_columns),
            "ActualChange": actual_change,
            "PredictedChange": predicted_change,
            "BaselineNoChange": float(row.iloc[0]["Baseline_NoChange"]),
            "BaselineFuelTrend": float(row.iloc[0]["Baseline_FuelTrend"]),
            "BaselineBrentTrend": float(row.iloc[0]["Baseline_BrentTrend"]),
            "ActualDirection": int(_classify_direction(np.array([actual_change]))[0]),
            "PredictedDirection": predicted_direction,
            "DirectionConfidence": confidence,
            "ProbabilityDown": float(probabilities[0]),
            "ProbabilityFlat": float(probabilities[1]),
            "ProbabilityUp": float(probabilities[2]),
            "PredictedMagnitude": magnitude,
            "ActualPrice": actual_price,
            "PredictedPrice": predicted_price,
            "ErrorCents": abs(actual_price - predicted_price) * 100.0,
        })
        since_refit += 1
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(), "Purged XGBoost walk-forward -testi ei tuottanut ennusteita."
    return out.sort_values("ForecastDate").reset_index(drop=True), None


def build_forecast_diagnostics(backtest_df: pd.DataFrame, *, max_shift_weeks: int = 6):
    required = {"Date", "ForecastDate", "Fuel", "FuelPrice", "ActualChange", "PredictedChange", "ActualPrice", "PredictedPrice"}
    if backtest_df is None or backtest_df.empty or not required.issubset(backtest_df.columns):
        missing = required.difference(backtest_df.columns) if backtest_df is not None else required
        return {}, pd.DataFrame(), pd.DataFrame(), f"Diagnostiikkaan tarvittavia sarakkeita puuttuu: {sorted(missing)}"
    df = backtest_df.copy()
    df["SignedErrorCents"] = (df["PredictedPrice"] - df["ActualPrice"]) * 100.0
    df["AbsoluteErrorCents"] = df["SignedErrorCents"].abs()
    df["ActualDirection"] = _classify_direction(df["ActualChange"].to_numpy(float))
    df["PredictedDirection"] = _classify_direction(df["PredictedChange"].to_numpy(float))
    df["DirectionCorrect"] = df["ActualDirection"] == df["PredictedDirection"]
    timing_rows = []
    for shift in range(-max_shift_weeks, max_shift_weeks + 1):
        comparison = pd.DataFrame({
            "Actual": df["ActualChange"],
            "Predicted": df["PredictedChange"].shift(shift),
        }).dropna()
        if len(comparison) < 15:
            continue
        corr = comparison["Actual"].corr(comparison["Predicted"])
        timing_rows.append({
            "ShiftWeeks": shift,
            "Correlation": float(corr) if pd.notna(corr) else None,
            "DirectionAccuracy": float(np.mean(
                _classify_direction(comparison["Actual"].to_numpy(float)) ==
                _classify_direction(comparison["Predicted"].to_numpy(float))
            )),
            "Observations": len(comparison),
        })
    timing_df = pd.DataFrame(timing_rows)
    if timing_df.empty:
        return {}, pd.DataFrame(), df, "Ajoitusanalyysi ei tuottanut tuloksia."
    valid = timing_df.dropna(subset=["Correlation"])
    best = valid.sort_values(["Correlation", "DirectionAccuracy"], ascending=[False, False]).iloc[0] if not valid.empty else None
    best_shift = int(best["ShiftWeeks"]) if best is not None else 0
    best_corr = float(best["Correlation"]) if best is not None else None
    timing_text = (
        f"Ennuste näyttää olevan noin {best_shift} viikkoa liian aikaisin."
        if best_shift > 0 else
        f"Ennuste näyttää olevan noin {abs(best_shift)} viikkoa myöhässä."
        if best_shift < 0 else
        "Paras kohdistus löytyy ilman ajallista siirtoa."
    )
    mean_signed = float(df["SignedErrorCents"].mean())
    bias_text = (
        "Malli ennustaa keskimäärin liian korkeaa hintaa." if mean_signed > 1.0 else
        "Malli ennustaa keskimäärin liian matalaa hintaa." if mean_signed < -1.0 else
        "Mallissa ei näy merkittävää jatkuvaa hintaharhaa."
    )
    diagnostics = {
        "best_shift_weeks": best_shift,
        "best_correlation": best_corr,
        "timing_interpretation": timing_text,
        "mean_signed_error_cents": mean_signed,
        "median_absolute_error_cents": float(df["AbsoluteErrorCents"].median()),
        "mean_absolute_error_cents": float(df["AbsoluteErrorCents"].mean()),
        "direction_accuracy_unshifted": float(df["DirectionCorrect"].mean()),
        "bias_interpretation": bias_text,
        "observations": int(len(df)),
    }
    return diagnostics, timing_df, df, None


def _build_feature_importance(direction_model, magnitude_model, feature_columns: list[str]) -> pd.DataFrame:
    direction = direction_model.feature_importances_
    magnitude = magnitude_model.feature_importances_
    combined = 0.65 * direction + 0.35 * magnitude
    out = pd.DataFrame({
        "Feature": feature_columns,
        "Label": [FEATURE_LABELS.get(f, f) for f in feature_columns],
        "DirectionImportance": direction,
        "MagnitudeImportance": magnitude,
        "Importance": combined,
    })
    total = float(out["Importance"].sum())
    out["ImportanceShare"] = out["Importance"] / total if total > 0 else 0.0
    return out.sort_values("Importance", ascending=False).reset_index(drop=True)


def _build_latest_contributions(magnitude_model, latest_row, current_price: float, predicted_direction: int, feature_columns: list[str]) -> pd.DataFrame:
    latest_features = latest_row[feature_columns]
    matrix = xgb.DMatrix(latest_features, feature_names=feature_columns)
    contribs = magnitude_model.get_booster().predict(matrix, pred_contribs=True)[0]
    rows = []
    for feature, value, contrib in zip(feature_columns, latest_features.iloc[0].to_numpy(float), contribs[:-1]):
        signed = float(contrib) * float(predicted_direction)
        rows.append({
            "Feature": feature,
            "Label": FEATURE_LABELS.get(feature, feature),
            "Value": float(value),
            "ContributionPctPoints": signed * 100.0,
            "ContributionCents": signed * current_price * 100.0,
        })
    return pd.DataFrame(rows).sort_values("ContributionCents", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def _confidence_label(*, mae_cents, direction_accuracy, observations, r_squared) -> str:
    if mae_cents is None or direction_accuracy is None or observations < 30:
        return "Heikko"
    if mae_cents <= 5.5 and direction_accuracy >= 0.70 and r_squared is not None and r_squared >= 0.10:
        return "Hyvä"
    if mae_cents <= 8.0 and direction_accuracy >= 0.60:
        return "Kohtalainen"
    return "Heikko"


def _direction_label(predicted_change_cents, direction_accuracy, class_confidence) -> str:
    if predicted_change_cents is None or pd.isna(predicted_change_cents):
        return "Ei arviota"
    if direction_accuracy is None or direction_accuracy < 0.55 or class_confidence is None or class_confidence < 0.45:
        return "Epävarma"
    if predicted_change_cents >= 2.0:
        return "Nousupaine"
    if predicted_change_cents <= -2.0:
        return "Laskupaine"
    return "Epävarma / melko vakaa"


def calculate_fuel_forecast_xgb(
    brent_df: pd.DataFrame,
    eurusd_df: pd.DataFrame | None,
    weekly_fuel_df: pd.DataFrame,
    us_inventory_df: pd.DataFrame | None,
    crack_df: pd.DataFrame | None,
    fuel_name: str,
    *,
    forecast_horizon_weeks: int = DEFAULT_FORECAST_HORIZON_WEEKS,
    train_share: float = DEFAULT_TRAIN_SHARE,
    min_train_observations: int = DEFAULT_MIN_TRAIN_OBSERVATIONS,
):
    config = _get_fuel_model_config(fuel_name)
    features = list(config["feature_columns"])
    feature_frame, message = build_fuel_feature_frame_xgb(
        brent_df, eurusd_df, weekly_fuel_df, us_inventory_df, crack_df,
        fuel_name, forecast_horizon_weeks=forecast_horizon_weeks,
    )
    if feature_frame.empty:
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), message
    model_df = feature_frame.dropna(subset=["Fuel_Target_Price", "Fuel_Target_Change", *features]).copy()
    if len(model_df) < min_train_observations:
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"{fuel_name}-malliin jäi vain {len(model_df)} tavoitehavaintoa."
    initial_train_size = min(max(min_train_observations, int(len(model_df) * train_share)), len(model_df) - 30)
    backtest, message = _walk_forward_backtest_xgb(
        model_df,
        initial_train_size=initial_train_size,
        forecast_horizon_weeks=forecast_horizon_weeks,
        refit_every=DEFAULT_REFIT_EVERY,
        feature_columns=features,
        fuel_name=fuel_name,
    )
    if backtest.empty:
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), message
    metrics = _regression_metrics(backtest)
    x_final = model_df[features]
    y_change = model_df["Fuel_Target_Change"].to_numpy(float)
    y_direction = _direction_to_class(y_change)
    if len(np.unique(y_direction)) < 3:
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"{fuel_name}-suuntamallista puuttuu vähintään yksi luokka."
    direction_model = _create_direction_model_for_fuel(fuel_name, random_state=42)
    magnitude_model = _create_magnitude_model_for_fuel(fuel_name, random_state=142)
    direction_model.fit(x_final, y_direction, sample_weight=_balanced_sample_weights(y_direction))
    magnitude_model.fit(x_final, np.abs(y_change))
    latest_row = feature_frame.iloc[[-1]]
    probabilities = direction_model.predict_proba(latest_row[features])[0]
    magnitude = float(magnitude_model.predict(latest_row[features])[0])
    predicted_change, predicted_direction, confidence = _combine_stage_predictions(probabilities, magnitude)
    latest_price = float(latest_row.iloc[0]["FuelPrice"])
    predicted_price = latest_price * (1.0 + predicted_change)
    predicted_change_cents = (predicted_price - latest_price) * 100.0
    residual_std = metrics.get("residual_std_cents") or metrics.get("mae_cents")
    margin = 1.28 * float(residual_std) if residual_std is not None else None
    importance_df = _build_feature_importance(direction_model, magnitude_model, features)
    contribution_df = _build_latest_contributions(magnitude_model, latest_row, latest_price, predicted_direction, features)
    summary = {
        "model_version": "XGBoost V6 – lukittu optimoitu tuotantomalli",
        "fuel": fuel_name,
        "feature_count": len(features),
        "model_params": dict(config["params"]),
        "forecast_horizon_weeks": forecast_horizon_weeks,
        "forecast_horizon_text": f"seuraavat {forecast_horizon_weeks} viikkoa",
        "latest_date": pd.to_datetime(latest_row.iloc[0]["Date"]),
        "forecast_date": pd.to_datetime(latest_row.iloc[0]["ForecastDate"]),
        "latest_fuel_price": latest_price,
        "latest_brent_usd": float(latest_row.iloc[0]["Brent_USD"]),
        "predicted_change_pct": predicted_change * 100.0,
        "predicted_change_cents": predicted_change_cents,
        "predicted_price": predicted_price,
        "predicted_direction_class": predicted_direction,
        "direction_confidence": confidence,
        "probability_down": float(probabilities[0]),
        "probability_flat": float(probabilities[1]),
        "probability_up": float(probabilities[2]),
        "predicted_magnitude_pct": magnitude * 100.0,
        "interval_low_price": predicted_price - margin / 100.0 if margin is not None else None,
        "interval_high_price": predicted_price + margin / 100.0 if margin is not None else None,
        "direction": _direction_label(predicted_change_cents, metrics.get("direction_accuracy"), confidence),
        "confidence": _confidence_label(
            mae_cents=metrics.get("mae_cents"),
            direction_accuracy=metrics.get("direction_accuracy"),
            observations=int(metrics.get("observations", 0)),
            r_squared=metrics.get("r_squared"),
        ),
        "walk_forward_mae_cents": metrics.get("mae_cents"),
        "walk_forward_rmse_cents": metrics.get("rmse_cents"),
        "walk_forward_median_error_cents": metrics.get("median_error_cents"),
        "direction_accuracy": metrics.get("direction_accuracy"),
        "direction_accuracy_active": metrics.get("direction_accuracy_active"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "macro_f1": metrics.get("macro_f1"),
        "class_metrics": metrics.get("class_metrics"),
        "r_squared": metrics.get("r_squared"),
        "test_observations": int(metrics.get("observations", 0)),
        "train_observations": int(len(model_df)),
    }
    return summary, backtest, importance_df, contribution_df, None


def calculate_all_fuel_forecasts_xgb(
    brent_df: pd.DataFrame,
    eurusd_df: pd.DataFrame | None,
    weekly_fuel_df: pd.DataFrame,
    us_inventory_df: pd.DataFrame | None,
    crack_df: pd.DataFrame | None,
    fuels: list[str] | None = None,
    *,
    forecast_horizon_weeks: int = DEFAULT_FORECAST_HORIZON_WEEKS,
):
    selected = fuels or ["95E10", "Diesel"]
    summaries, backtests, importances, contributions, messages = {}, {}, {}, {}, []
    for fuel in selected:
        summary, backtest, importance, contribution, message = calculate_fuel_forecast_xgb(
            brent_df, eurusd_df, weekly_fuel_df, us_inventory_df, crack_df,
            fuel, forecast_horizon_weeks=forecast_horizon_weeks,
        )
        if summary:
            summaries[fuel] = summary
        if not backtest.empty:
            backtests[fuel] = backtest
        if not importance.empty:
            importances[fuel] = importance
        if not contribution.empty:
            contributions[fuel] = contribution
        if message:
            messages.append(message)
    return summaries, backtests, importances, contributions, messages


def compare_forecast_horizons_xgb(
    brent_df: pd.DataFrame,
    eurusd_df: pd.DataFrame | None,
    weekly_fuel_df: pd.DataFrame,
    us_inventory_df: pd.DataFrame | None,
    crack_df: pd.DataFrame | None,
    fuel_name: str,
    horizons: tuple[int, ...] = (1, 2, 4),
):
    rows, backtests, messages = [], {}, []
    for horizon in horizons:
        summary, backtest, _, _, message = calculate_fuel_forecast_xgb(
            brent_df, eurusd_df, weekly_fuel_df, us_inventory_df, crack_df,
            fuel_name, forecast_horizon_weeks=horizon,
        )
        if message:
            messages.append(f"{horizon} vk: {message}")
            continue
        if not summary or backtest.empty:
            continue
        backtests[horizon] = backtest
        comparison = build_model_comparison_table(backtest)
        xgb_row = comparison.loc[comparison["Malli"] == "XGBoost"]
        if xgb_row.empty:
            continue
        r = xgb_row.iloc[0]
        rows.append({
            "Horisontti_vk": horizon,
            "MAE_snt_l": r["MAE_snt_l"],
            "RMSE_snt_l": r["RMSE_snt_l"],
            "Suuntatarkkuus": r["Suuntatarkkuus"],
            "Tasapainotettu_tarkkuus": r["Tasapainotettu_tarkkuus"],
            "Macro_F1": r["Macro_F1"],
            "Havaintoja": r["Havaintoja"],
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("Horisontti_vk").reset_index(drop=True)
    return result, backtests, messages
