import logging

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

PRIMARY_USES = [
    "Education", "Lodging/residential", "Office", "Entertainment/public assembly",
    "Other", "Retail", "Parking", "Public services", "Warehouse/storage",
    "Food sales and service", "Religious worship", "Healthcare", "Utility",
    "Technology/science", "Manufacturing/industrial", "Services",
]

TIME_FEATURES = ["hour", "dayofweek", "month", "dayofmonth", "is_weekend", "hour_sin", "hour_cos"]
BUILDING_FEATURES = ["meter", "site_id", "primary_use_code", "log_square_feet",
                     "building_age", "floor_count"]
WEATHER_FEATURES = ["air_temperature", "dew_temperature", "cloud_coverage",
                    "precip_depth_1_hr", "sea_level_pressure", "wind_speed",
                    "wind_dir_sin", "wind_dir_cos", "temp_dew_spread"]
CATEGORICAL_FEATURES = ["meter", "site_id", "primary_use_code"]

REQUIRED = {"building_id", "meter", "timestamp", "meter_reading", "site_id",
            "primary_use", "square_feet", "year_built", "floor_count",
            *config.WEATHER_COLS}


def lag_columns(lags=config.LAGS):
    return [f"lag_{h}h" for h in lags]


def feature_columns(use_lags=config.USE_LAGS, lags=config.LAGS):
    cols = TIME_FEATURES + BUILDING_FEATURES + WEATHER_FEATURES
    return cols + (lag_columns(lags) if use_lags else [])


def add_time_features(df):
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour.astype("int8")
    df["dayofweek"] = ts.dt.dayofweek.astype("int8")
    df["month"] = ts.dt.month.astype("int8")
    df["dayofmonth"] = ts.dt.day.astype("int8")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype("float32")
    return df


def add_building_features(df):
    df["log_square_feet"] = np.log1p(df["square_feet"]).astype("float32")
    year_ref = df["timestamp"].dt.year
    df["building_age"] = (year_ref - df["year_built"]).astype("float32")   # NaN stays NaN
    uses = pd.Categorical(df["primary_use"].astype("object"), categories=PRIMARY_USES)
    codes = uses.codes.astype("int16")
    codes[codes == -1] = len(PRIMARY_USES)           # unknown category
    df["primary_use_code"] = codes
    return df


def add_weather_features(df):
    rad = np.deg2rad(df["wind_direction"])
    df["wind_dir_sin"] = np.sin(rad).astype("float32")
    df["wind_dir_cos"] = np.cos(rad).astype("float32")
    df["temp_dew_spread"] = (df["air_temperature"] - df["dew_temperature"]).astype("float32")
    return df


def add_lag_features(df, lags=config.LAGS):
    """lag_Xh = the reading of the same building/meter exactly X hours earlier (NaN if unseen)."""
    keys = ["building_id", "meter", "timestamp"]
    base = df[[*keys, "meter_reading"]]
    for h in lags:
        shifted = base.copy()
        shifted["timestamp"] = shifted["timestamp"] + pd.Timedelta(hours=h)
        shifted = shifted.rename(columns={"meter_reading": f"lag_{h}h"})
        df = df.merge(shifted, on=keys, how="left")
    return df


def build_features(df, use_lags=config.USE_LAGS, lags=config.LAGS):
    missing = REQUIRED - set(df.columns)
    if missing:
        raise KeyError(f"build_features: missing columns {sorted(missing)}")
    df = df.drop_duplicates(["building_id", "meter", "timestamp"]).copy()
    df = add_time_features(df)
    df = add_building_features(df)
    df = add_weather_features(df)
    if use_lags:
        df = add_lag_features(df, lags)
    df = df.sort_values(["building_id", "meter", "timestamp"]).reset_index(drop=True)
    log.info("Built %d feature columns", len(feature_columns(use_lags, lags)))
    return df
