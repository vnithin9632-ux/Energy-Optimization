"""Module 2 - Preprocessing (turns the EDA findings into rules).

Input : merged DataFrame from ingest.load_raw()
Output: (clean DataFrame, report dict)
Rules : 1. drop NaN / negative readings
        2. drop known outlier (building, meter) pairs (config.EXCLUDE_PAIRS, e.g. building 1099 steam)
        3. drop long runs of zero readings on electricity (site 0 reads 0 until ~May 2016)
        4. fill weather gaps: interpolate short gaps, then site median, then global median
        5. *flag* (log only) any building whose mean reading is extreme
Fails : never silently - every step records how many rows it touched in the report.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)


def drop_invalid_readings(df):
    keep = df["meter_reading"].notna() & (df["meter_reading"] >= 0)
    return df[keep], int((~keep).sum())


def drop_excluded_pairs(df, pairs):
    if not pairs:
        return df, 0
    drop = pd.Series(False, index=df.index)
    for building_id, meter in pairs:
        drop |= (df["building_id"] == building_id) & (df["meter"] == meter)
    return df[~drop], int(drop.sum())


def remove_zero_streaks(df, min_len=config.ZERO_STREAK_HOURS,
                        meters=config.ZERO_STREAK_METERS):
    """Drop runs of >= min_len consecutive zero rows per (building, meter)."""
    df = df.sort_values(["building_id", "meter", "timestamp"]).reset_index(drop=True)
    key = df["building_id"].astype("int64") * 10 + df["meter"].astype("int64")
    is_zero = df["meter_reading"].eq(0)
    new_run = (is_zero != is_zero.shift()) | (key != key.shift())
    run_id = new_run.cumsum()
    run_len = is_zero.groupby(run_id).transform("size")
    drop = is_zero & run_len.ge(min_len) & df["meter"].isin(meters)
    return df[~drop].reset_index(drop=True), int(drop.sum())


def flag_extreme_buildings(df, factor=config.EXTREME_FACTOR):
    """Log (do not drop) buildings whose mean reading is >> the typical building."""
    means = df.groupby(["meter", "building_id"])["meter_reading"].mean()
    flagged = []
    for meter, s in means.groupby(level="meter"):
        median = s.median()
        if median > 0:
            for (_, building_id), value in s[s > factor * median].items():
                flagged.append({"building_id": int(building_id), "meter": int(meter),
                                "mean_reading": float(value)})
    for f in flagged:
        log.warning("Extreme building flagged: %s", f)
    return flagged


def fill_weather(df):
    """Fill weather gaps per site; returns (df, missing-share before, after)."""
    cols = [c for c in config.WEATHER_COLS if c in df.columns]
    before = {c: float(df[c].isna().mean()) for c in cols}

    w = (df[["site_id", "timestamp", *cols]]
         .drop_duplicates(["site_id", "timestamp"])
         .sort_values(["site_id", "timestamp"]).reset_index(drop=True))
    w[cols] = w.groupby("site_id")[cols].transform(
        lambda s: s.interpolate(limit=config.WEATHER_INTERP_LIMIT, limit_direction="both"))
    w[cols] = w[cols].fillna(w.groupby("site_id")[cols].transform("median"))
    w[cols] = w[cols].fillna(w[cols].median())      # e.g. a site with no pressure sensor

    df = df.drop(columns=cols).merge(w, on=["site_id", "timestamp"], how="left")
    after = {c: float(df[c].isna().mean()) for c in cols}
    return df, before, after


def clean(df):
    report = {"rows_in": int(len(df))}

    df, n = drop_invalid_readings(df)
    report["invalid_readings_removed"] = n

    df, n = drop_excluded_pairs(df, config.EXCLUDE_PAIRS)
    report["excluded_pair_rows_removed"] = n

    df, n = remove_zero_streaks(df)
    report["zero_streak_rows_removed"] = n

    df, before, after = fill_weather(df)
    report["weather_missing_before"] = before
    report["weather_missing_after"] = after

    df = df.sort_values(["building_id", "meter", "timestamp"]).reset_index(drop=True)
    report["extreme_buildings_flagged"] = flag_extreme_buildings(df)
    report["rows_out"] = int(len(df))
    log.info("Cleaning: %s -> %s rows", f"{report['rows_in']:,}", f"{report['rows_out']:,}")
    return df, report


def save_report(report, out_dir=config.OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cleaning_report.json").write_text(json.dumps(report, indent=2))
