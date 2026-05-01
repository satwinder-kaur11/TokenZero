from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from core.settings import get_settings
from db.queries import get_recent_requests, get_stats

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # pragma: no cover - optional runtime dependency
    st_autorefresh = None

st.set_page_config(page_title="Smart Router", layout="wide")
st.title("Smart Router")
st.caption("Live routing metrics and ROI proof from SQLite telemetry.")

if st_autorefresh is not None:
    st_autorefresh(interval=30_000, key="smart-router-autorefresh")

settings = get_settings()
default_db = settings.sqlite_path

with st.sidebar:
    st.subheader("Filters")
    sqlite_path = st.text_input("SQLite path", value=default_db)
    hours = st.slider("Lookback (hours)", min_value=1, max_value=168, value=24)

db_exists = Path(sqlite_path).exists()
if not db_exists:
    st.warning(f"Database not found at `{sqlite_path}`. Start the API and send traffic first.")
    st.stop()


@st.cache_data(ttl=30)
def load_dashboard_payload(db_path: str, lookback_hours: int) -> tuple[dict[str, Any], pd.DataFrame]:
    stats = asyncio.run(get_stats(db_path, lookback_hours))
    rows = asyncio.run(get_recent_requests(db_path, lookback_hours))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["ts"] = pd.to_datetime(frame["ts"], unit="s")
        frame["total_tokens"] = frame["prompt_tokens"].fillna(0) + frame["completion_tokens"].fillna(0)
    return stats, frame


stats, df = load_dashboard_payload(sqlite_path, hours)

top_1, top_2, top_3, top_4 = st.columns(4)
top_1.metric("Total Requests", f'{stats["total_requests"]:,}')
top_2.metric("Total Cost", f'${stats["total_cost_usd"]:.4f}')
top_3.metric("Savings vs Always-70B", f'${stats["savings_usd"]:.4f}', f'{stats["savings_pct"]:.2f}%')
top_4.metric("Average Latency", f'{stats["avg_latency_ms"]:.1f} ms')

if df.empty:
    st.info("No request data in the selected window yet.")
    st.stop()

st.subheader("Cost Over Time by Tier")
cost_ts = (
    df.assign(minute=df["ts"].dt.floor("min"))
    .groupby(["minute", "tier"], as_index=False)["cost_usd"]
    .sum()
)
cost_chart = (
    alt.Chart(cost_ts)
    .mark_line(point=True)
    .encode(
        x=alt.X("minute:T", title="Time"),
        y=alt.Y("cost_usd:Q", title="Cost (USD)"),
        color=alt.Color("tier:N", title="Tier"),
        tooltip=["minute:T", "tier:N", alt.Tooltip("cost_usd:Q", format=".6f")],
    )
    .properties(height=280)
)
st.altair_chart(cost_chart, use_container_width=True)

left, right = st.columns(2)

with left:
    st.subheader("Latency by Model (P50/P95)")
    latency_summary = (
        df.groupby("model")["latency_ms"]
        .agg(
            p50=lambda s: float(s.quantile(0.50)),
            p95=lambda s: float(s.quantile(0.95)),
        )
        .reset_index()
        .melt(id_vars="model", value_vars=["p50", "p95"], var_name="metric", value_name="latency_ms")
    )
    latency_chart = (
        alt.Chart(latency_summary)
        .mark_bar()
        .encode(
            x=alt.X("model:N", title="Model"),
            y=alt.Y("latency_ms:Q", title="Latency (ms)"),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=["model:N", "metric:N", alt.Tooltip("latency_ms:Q", format=".2f")],
        )
        .properties(height=280)
    )
    st.altair_chart(latency_chart, use_container_width=True)

with right:
    st.subheader("Complexity Score Distribution")
    complexity_chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("complexity_score:Q", bin=alt.Bin(step=0.1), title="Complexity score"),
            y=alt.Y("count():Q", title="Requests"),
            tooltip=[alt.Tooltip("count():Q", title="Requests")],
        )
        .properties(height=280)
    )
    st.altair_chart(complexity_chart, use_container_width=True)

lower_left, lower_right = st.columns(2)

with lower_left:
    st.subheader("A/B Variant Comparison")
    ab_rows = []
    for variant, info in stats["ab_comparison"].items():
        ab_rows.append(
            {
                "variant": variant,
                "requests": int(info["requests"]),
                "avg_latency_ms": float(info["avg_latency_ms"]),
                "avg_cost_usd": float(info["avg_cost_usd"]),
                "avg_completion_tokens": float(info["avg_completion_tokens"]),
            }
        )
    if ab_rows:
        st.dataframe(pd.DataFrame(ab_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No A/B data yet.")

with lower_right:
    st.subheader("Cost by Tier")
    cost_rows = [{"tier": tier, "cost_usd": value} for tier, value in stats["cost_by_tier"].items()]
    if cost_rows:
        cost_df = pd.DataFrame(cost_rows)
        tier_chart = (
            alt.Chart(cost_df)
            .mark_bar()
            .encode(
                x=alt.X("tier:N", title="Tier"),
                y=alt.Y("cost_usd:Q", title="Cost (USD)"),
                color=alt.Color("tier:N", legend=None),
                tooltip=["tier:N", alt.Tooltip("cost_usd:Q", format=".6f")],
            )
            .properties(height=220)
        )
        st.altair_chart(tier_chart, use_container_width=True)
    else:
        st.caption("No tier-level cost data yet.")
