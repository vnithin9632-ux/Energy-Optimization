"""Module 7 - Dashboard.   Run:  streamlit run app.py
Reads the files written by run_pipeline.py (default folder: outputs/, override with
the ENERGY_OUT environment variable)."""
import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

OUT = Path(os.environ.get("ENERGY_OUT", "outputs"))
METER_NAMES = {0: "electricity", 1: "chilled water", 2: "steam", 3: "hot water"}

st.set_page_config(page_title="Smart Energy Analytics", layout="wide")
alt.data_transformers.disable_max_rows()


def show(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:                       # older Streamlit versions
        st.altair_chart(chart, use_container_width=True)


@st.cache_data
def load(out: str):
    out = Path(out)
    needed = ["predictions_scored.parquet", "metrics.json", "building_summary.csv",
              "daily_peaks.csv", "anomalies.csv"]
    missing = [f for f in needed if not (out / f).is_file()]
    if missing:
        return None, missing
    data = {
        "pred": pd.read_parquet(out / "predictions_scored.parquet"),
        "metrics": json.loads((out / "metrics.json").read_text()),
        "summary": pd.read_csv(out / "building_summary.csv"),
        "peaks": pd.read_csv(out / "daily_peaks.csv", parse_dates=["date", "peak_time"]),
        "anomalies": pd.read_csv(out / "anomalies.csv", parse_dates=["timestamp"]),
    }
    imp = out / "feature_importance.csv"
    data["importance"] = pd.read_csv(imp) if imp.is_file() else None
    return data, []


data, missing = load(str(OUT))
st.title("AI-Driven Smart Energy Analytics")
if data is None:                            # failure handling: results not available
    st.error(f"Results not found in '{OUT}'. Missing: {', '.join(missing)}. "
             "Run `python run_pipeline.py --data-dir data` first.")
    st.stop()

pred, metrics, summary = data["pred"], data["metrics"], data["summary"]
m, base = metrics["model"], metrics["baseline_profile"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model MAE (kWh)", f"{m['MAE']:.2f}")
c2.metric("Model RMSE (kWh)", f"{m['RMSE']:.2f}")
c3.metric("R²", f"{m['R2']:.3f}")
c4.metric("MAE gain vs best baseline", f"{metrics['mae_improvement_vs_best_baseline_%']:.1f}%")
st.caption(f"Hold-out period starts {metrics['cutoff'][:16]} · {metrics['n_eval']:,} evaluated hours · "
           f"baseline (hour-of-week profile) MAE {base['MAE']:.2f}")

tab1, tab2, tab3 = st.tabs(["Building forecast", "Portfolio & peaks", "Anomalies"])

with tab1:
    pairs = (pred[["building_id", "meter"]].drop_duplicates()
             .sort_values(["building_id", "meter"]).itertuples(index=False))
    options = {f"Building {b} · {METER_NAMES.get(mt, mt)}": (b, mt) for b, mt in pairs}
    choice = st.selectbox("Building / meter", list(options))
    b, mt = options[choice]
    sel = pred[(pred["building_id"] == b) & (pred["meter"] == mt)].sort_values("timestamp")
    long = sel.melt(id_vars="timestamp", value_vars=["actual", "predicted"],
                    var_name="series", value_name="kWh")
    line = alt.Chart(long).mark_line().encode(
        x=alt.X("timestamp:T", title=None), y=alt.Y("kWh:Q"), color="series:N")
    pts = alt.Chart(sel[sel["is_anomaly"] & (sel["direction"] == "above expected")]).mark_point(
        color="red", size=70, filled=True).encode(x="timestamp:T", y="actual:Q")
    show(line + pts)
    st.caption("Red dots = hours with consumption well above the forecast.")

with tab2:
    left, right = st.columns(2)
    with left:
        st.subheader("Daily portfolio peak")
        peaks = data["peaks"]
        show(alt.Chart(peaks).mark_bar().encode(
            x=alt.X("date:T", title=None), y=alt.Y("actual:Q", title="Peak-hour kWh"),
            tooltip=["peak_time", "actual", "predicted"]))
        st.caption(f"Most common peak hour: {int(peaks['peak_hour'].mode().iloc[0])}:00")
    with right:
        st.subheader("Top buildings by estimated excess kWh")
        top = summary.head(10).assign(label=lambda d: d["building_id"].astype(str) + "/m" + d["meter"].astype(str))
        show(alt.Chart(top).mark_bar().encode(
            x=alt.X("excess_kwh:Q", title="Excess kWh (flagged hours)"),
            y=alt.Y("label:N", sort="-x", title="building/meter")))
    if data["importance"] is not None:
        st.subheader("What drives the forecast")
        imp = data["importance"].head(12)
        show(alt.Chart(imp).mark_bar().encode(
            x="importance:Q", y=alt.Y("feature:N", sort="-x")))

with tab3:
    direction = st.radio("Show", ["above expected", "below expected", "both"], horizontal=True)
    an = data["anomalies"]
    if direction != "both":
        an = an[an["direction"] == direction]
    st.write(f"{len(an):,} flagged hours")
    st.dataframe(an.sort_values("robust_z", key=abs, ascending=False).head(500), hide_index=True)
    st.subheader("Building summary")
    st.dataframe(summary.round(2), hide_index=True)
