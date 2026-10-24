import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)


def detect_anomalies(pred, z_thresh=config.ANOMALY_Z, min_log_dev=config.ANOMALY_MIN_LOG_DEV):
    pred = pred.copy()
    pred["residual_log"] = np.log1p(pred["actual"]) - np.log1p(pred["predicted"])
    grp = pred.groupby(["building_id", "meter"])["residual_log"]
    med = grp.transform("median")
    mad = (pred["residual_log"] - med).abs().groupby(
        [pred["building_id"], pred["meter"]]).transform("median")
    pred["robust_z"] = 0.6745 * (pred["residual_log"] - med) / np.maximum(mad, 0.05)
    pred["is_anomaly"] = (pred["robust_z"].abs() > z_thresh) & (pred["residual_log"].abs() >= min_log_dev)
    pred["direction"] = np.where(pred["residual_log"] > 0, "above expected", "below expected")
    pred["excess_kwh"] = np.where(pred["is_anomaly"] & (pred["residual_log"] > 0),
                                  pred["actual"] - pred["predicted"], 0.0)
    return pred


def suppress_lag_echoes(pred, lags=config.LAGS):
    """A spike is 'remembered' by lag features, so the model over-predicts exactly
    24h / 168h later and produces false 'below expected' flags. Drop those echoes."""
    above = pred.loc[pred["is_anomaly"] & (pred["residual_log"] > 0),
                     ["building_id", "meter", "timestamp"]]
    echo = pd.Series(False, index=pred.index)
    if above.empty:
        return echo
    keys = pd.MultiIndex.from_frame(pred[["building_id", "meter", "timestamp"]])
    for h in lags:
        shifted = above.assign(timestamp=above["timestamp"] + pd.Timedelta(hours=h))
        echo |= keys.isin(pd.MultiIndex.from_frame(shifted))
    return echo & pred["is_anomaly"] & (pred["residual_log"] < 0)


def building_summary(pred):
    g = pred.groupby(["building_id", "meter"])
    out = g.agg(total_kwh=("actual", "sum"),
                hours=("actual", "size"),
                anomalies=("is_anomaly", "sum"),
                excess_kwh=("excess_kwh", "sum")).reset_index()
    out["anomaly_rate_%"] = 100 * out["anomalies"] / out["hours"]
    out["excess_share_%"] = 100 * out["excess_kwh"] / out["total_kwh"].replace(0, np.nan)
    return out.sort_values("excess_kwh", ascending=False).reset_index(drop=True)


def daily_peaks(pred):
    """Portfolio-level daily peak: the hour with the highest summed consumption."""
    hourly = pred.groupby("timestamp")[["actual", "predicted"]].sum()
    hourly["date"] = hourly.index.normalize()
    idx = hourly.groupby("date")["actual"].idxmax()
    peaks = hourly.loc[idx].reset_index().rename(columns={"timestamp": "peak_time"})
    peaks["peak_hour"] = peaks["peak_time"].dt.hour
    return peaks[["date", "peak_time", "peak_hour", "actual", "predicted"]]


def run_analytics(predictions, out_dir=config.OUT_DIR, lag_model=config.USE_LAGS):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if predictions.empty:
        log.warning("No predictions to analyse.")
        return predictions, pd.DataFrame(), pd.DataFrame()

    scored = detect_anomalies(predictions)
    if lag_model:
        echo = suppress_lag_echoes(scored)
        scored["is_anomaly"] &= ~echo
        log.info("Suppressed %d lag-echo flags", int(echo.sum()))
    summary = building_summary(scored)
    peaks = daily_peaks(scored)

    anomalies = scored[scored["is_anomaly"]][
        ["building_id", "meter", "timestamp", "actual", "predicted",
         "residual_log", "robust_z", "direction", "excess_kwh"]]
    anomalies.to_csv(out_dir / "anomalies.csv", index=False)
    summary.to_csv(out_dir / "building_summary.csv", index=False)
    peaks.to_csv(out_dir / "daily_peaks.csv", index=False)
    scored.to_parquet(out_dir / "predictions_scored.parquet", index=False)
    log.info("Anomalies: %d flagged hours in %d buildings",
             len(anomalies), anomalies["building_id"].nunique())
    return scored, summary, peaks
