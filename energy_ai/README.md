# AI-Driven Smart Energy Analytics - Design Project 1 (Review 2)

Pipeline on the ASHRAE Great Energy Predictor III data: ingest -> clean -> features ->
forecast -> analytics -> dashboard.

Only `train.csv`, `building_metadata.csv` and `weather_train.csv` are needed. The Kaggle
`test.csv` has no labels, so it cannot be used for evaluation.
Start with a sample of buildings (`--n-buildings 100`); the full file has ~20 M rows.

## Modules (one owner each)
| # | Module | File | Input -> Output | Failure behaviour |
|---|---|---|---|---|
| 1 | Data ingestion | `ingest.py` | csv files -> merged DataFrame | clear error if a file/column is missing or subset is empty |
| 2 | Preprocessing | `preprocess.py` | merged -> clean + report | every rule logs rows touched in `cleaning_report.json` |
| 3 | Feature engineering | `features.py` | clean -> model features | `KeyError` listing missing columns |
| 4 | Forecasting | `forecast.py` | features -> predictions, metrics | `ValueError` on empty train/test split |
| 5 | Energy analytics | `analytics.py` | predictions -> anomalies, peaks, summary | returns empty tables, no crash |
| 6 | Optimization | *not built yet (Review 3)* | | |
| 7 | Dashboard | `app.py` | outputs/ -> charts | shows an error + instruction if results are missing |

All tunable settings (subset size, cleaning rules, lags, thresholds) are in `config.py`.

## Design decisions you can defend in the viva
- **Time-based hold-out** (last 20 % of the time range), not a random split - random splits
  leak neighbouring hours of the same building into the test set.
- **log1p target**: readings are heavily skewed; this also matches the competition's RMSLE.
- **Two baselines on identical rows**: same hour last week, and a building/hour-of-week profile.
  The model has to beat them to justify itself.
- **Lag features = day-ahead forecasting**: `lag_24h` / `lag_168h` use readings observed one
  day / one week earlier. Use `--no-lags` for a weather/calendar-only model.
- **Cleaning from EDA**: zero streaks >= 48 h on electricity (site 0 before ~May 2016) are removed;
  building 1099 steam is excluded; weather gaps are interpolated (<= 6 h), then site median,
  then global median (site 5 has no pressure sensor).
- **HistGradientBoosting** handles missing values and categorical features natively and is much
  faster than a RandomForest on millions of rows.
- **Anomalies** = robust z-score of forecast residuals + minimum deviation. Flags 24 h / 168 h
  after a spike are suppressed (lag echo).

## Known limitations (say these honestly in the review)
- Anomaly flags are statistical, not ground truth. "above expected" flags are more reliable than
  "below expected"; rank buildings by `excess_kwh` rather than by raw flag count.
- The dataset has no solar/wind generation, so renewable integration and optimization need assumed
  profiles or a different data source.
- Results on the synthetic data are only a pipeline test, not project results.
