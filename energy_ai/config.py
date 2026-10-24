from pathlib import Path

# ---- paths -----------------------------------------------------------------
DATA_DIR = Path("data")      # folder holding the ASHRAE csv files
OUT_DIR = Path("outputs")    # everything the pipeline writes goes here

# ---- data subset (keeps memory manageable on a laptop) ---------------------
METERS = [0]          # 0 electricity, 1 chilledwater, 2 steam, 3 hotwater
SITE_IDS = None       # e.g. [0, 2]; None = all sites
N_BUILDINGS = 100     # random sample of buildings; None = all (needs lots of RAM)
RANDOM_SEED = 42
CHUNK_ROWS = 2_000_000

WEATHER_COLS = [
    "air_temperature", "cloud_coverage", "dew_temperature",
    "precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed",
]

# ---- preprocessing ---------------------------------------------------------
ZERO_STREAK_HOURS = 48        # a run of zeros at least this long is treated as "no data"
ZERO_STREAK_METERS = [0]      # zeros only mean "no data" for electricity
EXCLUDE_PAIRS = [(1099, 2)]   # (building_id, meter) pairs dropped as known outliers (EDA finding)
EXTREME_FACTOR = 20           # building mean > 20x the median building mean is *flagged* (logged only)
WEATHER_INTERP_LIMIT = 6      # max consecutive hours of weather gap that are interpolated

# ---- features / forecasting -------------------------------------------------
LAGS = (24, 168)              # hours; 24h = same hour yesterday, 168h = same hour last week
USE_LAGS = True               # True = day-ahead forecast using readings observed 24h/168h earlier
TEST_FRACTION = 0.2           # last 20 % of the time range is held out
MODEL_PARAMS = dict(
    max_iter=300,
    learning_rate=0.1,
    max_leaf_nodes=63,
    l2_regularization=1.0,
    random_state=RANDOM_SEED,
)

# ---- analytics ---------------------------------------------------------------
ANOMALY_Z = 4.0               # robust z-score threshold on forecast residuals
ANOMALY_MIN_LOG_DEV = 0.5     # and the residual must be at least this large in log space (~65 %)
