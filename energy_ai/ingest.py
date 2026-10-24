import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "train.csv": {"building_id", "meter", "timestamp", "meter_reading"},
    "building_metadata.csv": {"site_id", "building_id", "primary_use",
                              "square_feet", "year_built", "floor_count"},
    "weather_train.csv": {"site_id", "timestamp", *config.WEATHER_COLS},
}


def check_inputs(data_dir) -> None:
    """Fail early and clearly if an input file or column is missing."""
    data_dir = Path(data_dir)
    for fname, needed in REQUIRED_COLUMNS.items():
        path = data_dir / fname
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing input file: {path}. Download the ASHRAE Great Energy "
                f"Predictor III data and place {fname} in '{data_dir}'."
            )
        header = set(pd.read_csv(path, nrows=0).columns)
        missing = needed - header
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def _select_buildings(path, building, meters, n_buildings, seed, chunk_rows):
    """First pass over train.csv: which buildings actually have the wanted meters?"""
    pairs = []
    for chunk in pd.read_csv(path, usecols=["building_id", "meter"],
                             dtype={"building_id": "int32", "meter": "int8"},
                             chunksize=chunk_rows):
        pairs.append(chunk[chunk["meter"].isin(meters)].drop_duplicates())
    pairs = pd.concat(pairs).drop_duplicates()
    eligible = np.intersect1d(pairs["building_id"].unique(),
                              building["building_id"].unique())
    if len(eligible) == 0:
        raise ValueError(f"No buildings have meter(s) {meters} in the selected sites.")
    if n_buildings and len(eligible) > n_buildings:
        rng = np.random.default_rng(seed)
        eligible = np.sort(rng.choice(eligible, n_buildings, replace=False))
    return eligible


def load_raw(data_dir=config.DATA_DIR, meters=config.METERS, site_ids=config.SITE_IDS,
             n_buildings=config.N_BUILDINGS, seed=config.RANDOM_SEED,
             chunk_rows=config.CHUNK_ROWS) -> pd.DataFrame:
    """Read, subset and merge the three ASHRAE tables."""
    data_dir = Path(data_dir)
    check_inputs(data_dir)

    building = pd.read_csv(data_dir / "building_metadata.csv")
    if site_ids:
        building = building[building["site_id"].isin(site_ids)]

    ids = _select_buildings(data_dir / "train.csv", building, meters,
                            n_buildings, seed, chunk_rows)
    building = building[building["building_id"].isin(ids)]
    log.info("Selected %d buildings, meters %s", len(ids), list(meters))

    parts = []
    for chunk in pd.read_csv(data_dir / "train.csv",
                             dtype={"building_id": "int32", "meter": "int8",
                                    "meter_reading": "float32"},
                             chunksize=chunk_rows):
        chunk = chunk[chunk["building_id"].isin(ids) & chunk["meter"].isin(meters)]
        if len(chunk):
            parts.append(chunk)
    if not parts:
        raise ValueError("No meter readings left after applying the subset filters.")
    train = pd.concat(parts, ignore_index=True)
    train["timestamp"] = pd.to_datetime(train["timestamp"])

    weather = pd.read_csv(data_dir / "weather_train.csv")
    weather = weather[weather["site_id"].isin(building["site_id"].unique())]
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])

    df = (train.merge(building, on="building_id", how="left")
               .merge(weather, on=["site_id", "timestamp"], how="left"))

    # memory-friendly dtypes (same idea as in the EDA notebook)
    df["site_id"] = df["site_id"].astype("int8")
    df["square_feet"] = df["square_feet"].astype("int32")
    for c in ["year_built", "floor_count", *config.WEATHER_COLS]:
        df[c] = df[c].astype("float32")
    df["primary_use"] = df["primary_use"].astype("category")

    df = df.sort_values(["building_id", "meter", "timestamp"]).reset_index(drop=True)
    log.info("Loaded %s rows x %s columns", f"{len(df):,}", df.shape[1])
    return df
