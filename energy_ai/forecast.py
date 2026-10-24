import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

import config
import features as F

log = logging.getLogger(__name__)


def time_split(df, test_fraction=config.TEST_FRACTION):
    t0, t1 = df["timestamp"].min(), df["timestamp"].max()
    cutoff = t0 + (t1 - t0) * (1 - test_fraction)
    train, test = df[df["timestamp"] < cutoff], df[df["timestamp"] >= cutoff]
    if train.empty or test.empty:
        raise ValueError("Time split produced an empty train or test set.")
    return train.copy(), test.copy(), cutoff


def regression_metrics(y_true, y_pred):
    y = np.asarray(y_true, dtype="float64")
    p = np.clip(np.asarray(y_pred, dtype="float64"), 0, None)
    err = y - p
    nz = y > 1
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "RMSLE": float(np.sqrt(np.mean((np.log1p(y) - np.log1p(p)) ** 2))),
        "MAPE_%": float(np.mean(np.abs(err[nz] / y[nz])) * 100) if nz.any() else float("nan"),
        "R2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
    }


def _hour_of_week(df):
    return (df["dayofweek"].astype("int16") * 24 + df["hour"].astype("int16"))


def add_profile_baseline(train, test):
    """Mean reading per building/meter/hour-of-week learned on the training period only."""
    prof = (train.assign(how=_hour_of_week(train))
                 .groupby(["building_id", "meter", "how"])["meter_reading"].mean()
                 .rename("baseline_profile").reset_index())
    test = test.assign(how=_hour_of_week(test)).merge(
        prof, on=["building_id", "meter", "how"], how="left").drop(columns="how")
    return test


def train_model(X_train, y_train_log, features):
    cat_mask = [f in F.CATEGORICAL_FEATURES for f in features]
    model = HistGradientBoostingRegressor(categorical_features=cat_mask, **config.MODEL_PARAMS)
    model.fit(X_train, y_train_log)
    return model


def run_forecast(df, use_lags=config.USE_LAGS, test_fraction=config.TEST_FRACTION,
                 out_dir=config.OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = F.feature_columns(use_lags)
    train, test, cutoff = time_split(df, test_fraction)
    log.info("Train rows %s | test rows %s | cutoff %s",
             f"{len(train):,}", f"{len(test):,}", cutoff)

    X_train = train[feats].astype("float32")
    y_train = np.log1p(train["meter_reading"].to_numpy())
    model = train_model(X_train, y_train, feats)

    test = add_profile_baseline(train, test)
    test["predicted"] = np.clip(np.expm1(model.predict(test[feats].astype("float32"))), 0, None)
    test = test.rename(columns={"meter_reading": "actual"})

    baselines = ["baseline_profile"]
    if use_lags and "lag_168h" in test.columns:
        test["baseline_naive_168h"] = test["lag_168h"]
        baselines.append("baseline_naive_168h")

    # evaluate model and baselines on the SAME rows
    mask = test["actual"].notna()
    for b in baselines:
        mask &= test[b].notna()
    ev = test[mask]
    if ev.empty:
        raise ValueError("No rows left for evaluation (baselines have no overlap).")

    metrics = {
        "n_train": int(len(train)), "n_test": int(len(test)), "n_eval": int(len(ev)),
        "cutoff": str(cutoff), "use_lags": bool(use_lags),
        "model": regression_metrics(ev["actual"], ev["predicted"]),
    }
    for b in baselines:
        metrics[b] = regression_metrics(ev["actual"], ev[b])
    best_base = min(metrics[b]["MAE"] for b in baselines)
    metrics["mae_improvement_vs_best_baseline_%"] = float(
        100 * (best_base - metrics["model"]["MAE"]) / best_base)

    # permutation importance on a sample (answers "which features matter?")
    sample = ev.sample(min(20000, len(ev)), random_state=config.RANDOM_SEED)
    imp = permutation_importance(
        model, sample[feats].astype("float32"), np.log1p(sample["actual"]),
        scoring="neg_mean_absolute_error", n_repeats=3,
        random_state=config.RANDOM_SEED, n_jobs=1)
    importance = (pd.DataFrame({"feature": feats, "importance": imp.importances_mean})
                    .sort_values("importance", ascending=False))

    keep = ["building_id", "meter", "site_id", "primary_use", "timestamp",
            "actual", "predicted", *baselines]
    predictions = test[keep].copy()
    predictions["primary_use"] = predictions["primary_use"].astype("object")

    joblib.dump({"model": model, "features": feats}, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    importance.to_csv(out_dir / "feature_importance.csv", index=False)
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    log.info("Model MAE %.3f | best baseline MAE %.3f | improvement %.1f%%",
             metrics["model"]["MAE"], best_base, metrics["mae_improvement_vs_best_baseline_%"])
    return predictions, metrics
