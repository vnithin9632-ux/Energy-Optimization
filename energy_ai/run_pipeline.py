"""Runs modules 1-5 end to end:  python run_pipeline.py --data-dir data"""
import argparse
import logging
import time

import analytics
import config
import features
import forecast
import ingest
import preprocess


def main():
    p = argparse.ArgumentParser(description="Energy analytics pipeline")
    p.add_argument("--data-dir", default=str(config.DATA_DIR))
    p.add_argument("--out-dir", default=str(config.OUT_DIR))
    p.add_argument("--meters", type=int, nargs="+", default=config.METERS,
                   help="0 electricity, 1 chilledwater, 2 steam, 3 hotwater")
    p.add_argument("--sites", type=int, nargs="+", default=config.SITE_IDS)
    p.add_argument("--n-buildings", type=int, default=config.N_BUILDINGS,
                   help="random sample size; 0 = all buildings")
    p.add_argument("--no-lags", action="store_true", help="forecast without lag features")
    p.add_argument("--test-fraction", type=float, default=config.TEST_FRACTION)
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("pipeline")
    use_lags = not a.no_lags

    def step(name, fn, *args, **kw):
        t = time.time()
        log.info("== %s", name)
        result = fn(*args, **kw)
        log.info("   done in %.1fs", time.time() - t)
        return result

    df = step("1/5 ingest", ingest.load_raw, a.data_dir, a.meters, a.sites,
              a.n_buildings or None)
    df, report = step("2/5 preprocess", preprocess.clean, df)
    preprocess.save_report(report, a.out_dir)
    df = step("3/5 features", features.build_features, df, use_lags)
    preds, metrics = step("4/5 forecast", forecast.run_forecast, df, use_lags,
                          a.test_fraction, a.out_dir)
    step("5/5 analytics", analytics.run_analytics, preds, a.out_dir, use_lags)

    m, base = metrics["model"], metrics["baseline_profile"]
    print("\n---- RESULT --------------------------------------------------")
    print(f"Model    MAE {m['MAE']:.2f}  RMSE {m['RMSE']:.2f}  RMSLE {m['RMSLE']:.3f}  R2 {m['R2']:.3f}")
    print(f"Baseline MAE {base['MAE']:.2f}  RMSE {base['RMSE']:.2f}  (hour-of-week profile)")
    print(f"MAE improvement vs best baseline: {metrics['mae_improvement_vs_best_baseline_%']:.1f}%")
    print(f"Outputs written to: {a.out_dir}")
    print("Dashboard: streamlit run app.py")


if __name__ == "__main__":
    main()
