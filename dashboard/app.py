from __future__ import annotations
import sys
import asyncio
from pathlib import Path
from typing import Any

import altair as alt
import httpx
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.settings import get_settings
from db.queries import get_recent_requests, get_stats

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TokenZero — Smart Router",
    page_icon="⚡",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Gradient header */
    .hero-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(99, 179, 237, 0.2);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #63b3ed, #90cdf4, #bee3f8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-sub {
        color: #a0aec0;
        font-size: 1rem;
        margin-top: 0.3rem;
    }

    /* Tier badge colours */
    .tier-small  { background:#276749; color:#c6f6d5; padding:3px 10px; border-radius:20px; font-weight:700; font-size:0.85rem; }
    .tier-medium { background:#744210; color:#fefcbf; padding:3px 10px; border-radius:20px; font-weight:700; font-size:0.85rem; }
    .tier-large  { background:#44337a; color:#e9d8fd; padding:3px 10px; border-radius:20px; font-weight:700; font-size:0.85rem; }

    /* Route result card */
    .route-card {
        background: linear-gradient(135deg, #1a202c, #2d3748);
        border: 1px solid rgba(99,179,237,0.25);
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin-top: 1rem;
    }
    .route-card h3 { color:#90cdf4; margin:0 0 0.8rem 0; font-size:1.1rem; }
    .route-model  { font-size:1.6rem; font-weight:800; color:#fff; }
    .route-meta   { color:#a0aec0; font-size:0.88rem; margin-top:0.4rem; }

    /* Schema box */
    .schema-box {
        background:#1a202c;
        border:1px solid #2d3748;
        border-radius:10px;
        padding:1rem 1.2rem;
        font-family:monospace;
        font-size:0.82rem;
        color:#e2e8f0;
        overflow-x:auto;
        white-space:pre;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg,#1a202c,#2d3748);
        border:1px solid rgba(99,179,237,0.2);
        border-radius:12px;
        padding:1rem;
    }
    [data-testid="stMetricValue"] { color:#90cdf4 !important; font-size:1.8rem !important; }

    /* Tab styling */
    button[data-baseweb="tab"] { font-weight:600; }

    /* Chat messages */
    .stChatMessage { border-radius:12px; }
</style>
""", unsafe_allow_html=True)

if st_autorefresh is not None:
    st_autorefresh(interval=30_000, key="smart-router-autorefresh")

settings  = get_settings()
API_BASE  = "http://127.0.0.1:8000"

# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <p class="hero-title">⚡ TokenZero — Smart Router</p>
  <p class="hero-sub">Route every prompt to the right model at the right cost · 100% local · powered by Ollama</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    sqlite_path    = st.text_input("SQLite path", value=settings.sqlite_path)
    hours          = st.slider("Lookback (hours)", 1, 168, 24)
    router_api_url = st.text_input("Router API URL", value=f"{API_BASE}/v1/chat/completions")
    budget_hint    = st.selectbox("Budget hint", ["balanced", "cheap", "quality"])
    st.divider()
    st.markdown("**Model Tiers**")
    st.markdown(f"🟢 Small  → `{settings.small_model}`")
    st.markdown(f"🟡 Medium → `{settings.medium_model}`")
    st.markdown(f"🔴 Large  → `{settings.large_model}`")
    st.divider()
    st.caption("Auto-refreshes every 30 s")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_route, tab_chat, tab_metrics = st.tabs([
    "🛠️ Route Inspector (MCP Tool)",
    "💬 Router Chat",
    "📊 Live Metrics",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Route Inspector  (shows the route_request MCP tool to the user)
# ══════════════════════════════════════════════════════════════════════════════
with tab_route:
    st.markdown("#### 🛠️ `route_request` — MCP Tool Inspector")
    st.caption(
        "Type any prompt below. TokenZero scores its complexity and picks the optimal "
        "Ollama model tier — exactly what an AI agent does when it calls this tool."
    )

    # ── JSON Schema display ────────────────────────────────────────────────────
    with st.expander("📄 View JSON Schema (what the agent sees)", expanded=False):
        schema_col1, schema_col2 = st.columns(2)
        with schema_col1:
            st.markdown("**Input Schema**")
            st.markdown("""<div class="schema-box">{
  "name": "route_request",
  "input_schema": {
    "type": "object",
    "properties": {
      "prompt": {
        "type": "string",
        "description": "User prompt to route"
      },
      "budget_hint": {
        "type": "string",
        "enum": ["cheap","balanced","quality"],
        "default": "balanced"
      }
    },
    "required": ["prompt"]
  }
}</div>""", unsafe_allow_html=True)
        with schema_col2:
            st.markdown("**Output Schema**")
            st.markdown("""<div class="schema-box">{
  "output_schema": {
    "type": "object",
    "properties": {
      "selected_model":   {"type":"string"},
      "tier":             {"type":"string",
                           "enum":["small","medium","large"]},
      "complexity_score": {"type":"number",
                           "minimum":0.0,
                           "maximum":1.0},
      "estimated_cost_usd":{"type":"number"}
    }
  }
}</div>""", unsafe_allow_html=True)

    st.divider()

    # ── Quick prompt buttons ───────────────────────────────────────────────────
    st.markdown("**Try a sample prompt:**")
    qcol1, qcol2, qcol3 = st.columns(3)
    prefill = ""
    if qcol1.button("🟢 Simple — What is 2+2?"):
        prefill = "What is 2 + 2?"
    if qcol2.button("🟡 Medium — REST vs GraphQL"):
        prefill = "Summarize the key differences between REST and GraphQL APIs and explain when to use each."
    if qcol3.button("🔴 Complex — Fraud detection system"):
        prefill = (
            "Design a multi-agent orchestration architecture for a distributed "
            "financial fraud detection system with real-time stream processing, "
            "explainability requirements, and sub-100ms latency SLAs."
        )

    prompt_input = st.text_area(
        "Prompt",
        value=prefill,
        height=110,
        placeholder="Type your prompt here…",
        label_visibility="collapsed",
    )
    route_budget = st.radio(
        "Budget hint",
        ["cheap", "balanced", "quality"],
        horizontal=True,
        index=1,
        key="route_budget",
    )

    if st.button("⚡ Route this prompt", type="primary", use_container_width=True):
        if not prompt_input.strip():
            st.warning("Please enter a prompt first.")
        else:
            with st.spinner("Scoring complexity and selecting model…"):
                try:
                    resp = httpx.post(
                        f"{API_BASE}/tools/route_request",
                        json={"prompt": prompt_input, "budget_hint": route_budget},
                        timeout=10.0,
                    )
                    resp.raise_for_status()
                    result = resp.json()

                    tier       = result.get("tier", "")
                    model      = result.get("selected_model", "")
                    score      = float(result.get("complexity_score", 0))
                    cost       = float(result.get("estimated_cost_usd", 0))
                    badge_cls  = f"tier-{tier}"

                    # ── Result card ────────────────────────────────────────────
                    st.markdown(f"""
<div class="route-card">
  <h3>🤖 Agent Decision</h3>
  <div class="route-model">{model}</div>
  <div class="route-meta" style="margin-top:0.6rem;">
    Tier: &nbsp;<span class="{badge_cls}">{tier.upper()}</span>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Complexity score: <strong style="color:#fff">{score:.4f}</strong>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Est. cost: <strong style="color:#68d391">${cost:.6f}</strong>
  </div>
  <div class="route-meta" style="margin-top:0.8rem; font-style:italic; color:#718096;">
    "Agent selected [{model}] at [{tier}] tier — estimated cost ${cost:.6f}"
  </div>
</div>
""", unsafe_allow_html=True)

                    # ── Complexity gauge ───────────────────────────────────────
                    st.markdown(" ")
                    gcol1, gcol2, gcol3 = st.columns([1, 2, 1])
                    with gcol2:
                        st.markdown("**Complexity Score**")
                        st.progress(score)
                        threshold_note = (
                            "Below medium threshold (0.35) → small tier"
                            if score < 0.35
                            else "Below large threshold (0.70) → medium tier"
                            if score < 0.70
                            else "Above large threshold (0.70) → large tier"
                        )
                        st.caption(f"Score: `{score:.4f}` — {threshold_note}")

                    # ── Raw JSON for the agent ─────────────────────────────────
                    with st.expander("🔍 Raw JSON response (what the agent receives)"):
                        st.json(result)

                except httpx.HTTPError as exc:
                    st.error(f"❌ Could not reach the API server: {exc}\n\nMake sure the server is running on port 8000.")
                except Exception as exc:
                    st.error(f"❌ Unexpected error: {exc}")

    # ── How it works explainer ─────────────────────────────────────────────────
    st.divider()
    st.markdown("#### How routing works")
    hcol1, hcol2, hcol3 = st.columns(3)
    hcol1.info("**🟢 Small**\n\n`llama3.2:1b` · 1.3 GB\n\nScore < **0.35**\n\nGreetings, maths, simple lookup")
    hcol2.warning("**🟡 Medium**\n\n`llama3.2:3b` · 2.0 GB\n\nScore 0.35 – **0.70**\n\nSummaries, comparisons, short code")
    hcol3.error("**🔴 Large**\n\n`llama3.1:8b` · 4.9 GB\n\nScore ≥ **0.70**\n\nArchitecture, deep reasoning, long code")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Router Chat
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("#### 💬 Router Chat")
    st.caption("Chat with the Smart Router — it picks the right model for each message automatically.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    def call_router_chat(
        api_url: str,
        messages: list[dict[str, str]],
        budget: str,
    ) -> tuple[str, dict[str, str], str | None]:
        payload = {"messages": messages, "budget_hint": budget, "stream": False}
        try:
            response = httpx.post(
                api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120.0,
            )
        except httpx.HTTPError as exc:
            return "", {}, f"Network error: {exc}"

        if response.status_code != 200:
            return "", {}, f"Router returned {response.status_code}: {response.text[:500]}"

        try:
            body = response.json()
        except ValueError:
            return "", {}, "Router returned non-JSON response."

        answer  = ""
        choices = body.get("choices", []) if isinstance(body, dict) else []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                answer = str(message.get("content", "")).strip()

        headers_info = {
            "x-router-model":      response.headers.get("x-router-model", ""),
            "x-tier":              response.headers.get("x-tier", ""),
            "x-complexity-score":  response.headers.get("x-complexity-score", ""),
            "x-ab-variant":        response.headers.get("x-ab-variant", ""),
        }
        return answer, headers_info, None

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            meta = msg.get("meta")
            if isinstance(meta, dict) and meta.get("tier"):
                st.caption(
                    f'Tier: `{meta.get("tier","")}` | Model: `{meta.get("model","")}` | '
                    f'Complexity: `{meta.get("score","")}` | Variant: `{meta.get("variant","")}`'
                )

    user_prompt = st.chat_input("Message the Smart Router…")
    if user_prompt:
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
        model_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_messages
            if m["role"] in {"user", "assistant"}
        ]
        answer, headers, error = call_router_chat(router_api_url, model_messages, budget_hint)
        if error:
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": f"⚠️ Request failed: {error}"}
            )
        else:
            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": answer or "(empty response)",
                    "meta": {
                        "tier":    headers.get("x-tier", ""),
                        "model":   headers.get("x-router-model", ""),
                        "score":   headers.get("x-complexity-score", ""),
                        "variant": headers.get("x-ab-variant", ""),
                    },
                }
            )
        st.rerun()

    if st.button("🗑️ Clear chat history"):
        st.session_state.chat_messages = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Live Metrics
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.markdown("#### 📊 Live Routing Metrics")

    db_exists = Path(sqlite_path).exists()
    if not db_exists:
        st.warning(
            f"Database not found at `{sqlite_path}`. "
            "Send some requests via the Chat or Route Inspector tab to populate metrics."
        )
        st.stop()

    @st.cache_data(ttl=30)
    def load_dashboard_payload(db_path: str, lookback_hours: int) -> tuple[dict[str, Any], pd.DataFrame]:
        stats = asyncio.run(get_stats(db_path, lookback_hours))
        rows  = asyncio.run(get_recent_requests(db_path, lookback_hours))
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["ts"]           = pd.to_datetime(frame["ts"], unit="s")
            frame["total_tokens"] = frame["prompt_tokens"].fillna(0) + frame["completion_tokens"].fillna(0)
        return stats, frame

    stats, df = load_dashboard_payload(sqlite_path, hours)

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Requests",       f'{stats["total_requests"]:,}')
    k2.metric("Total Cost",           f'${stats["total_cost_usd"]:.4f}')
    k3.metric("Savings vs Always-Large", f'${stats["savings_usd"]:.4f}', f'{stats["savings_pct"]:.2f}%')
    k4.metric("Avg Latency",          f'{stats["avg_latency_ms"]:.1f} ms')

    if df.empty:
        st.info("No request data in the selected window yet.")
        st.stop()

    # ── Cost over time ────────────────────────────────────────────────────────
    st.subheader("Cost Over Time by Tier")
    cost_ts = (
        df.assign(minute=df["ts"].dt.floor("min"))
        .groupby(["minute", "tier"], as_index=False)["cost_usd"]
        .sum()
    )
    st.altair_chart(
        alt.Chart(cost_ts).mark_line(point=True).encode(
            x=alt.X("minute:T", title="Time"),
            y=alt.Y("cost_usd:Q", title="Cost (USD)"),
            color=alt.Color("tier:N", title="Tier"),
            tooltip=["minute:T", "tier:N", alt.Tooltip("cost_usd:Q", format=".6f")],
        ).properties(height=260),
        use_container_width=True,
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Latency by Model (P50 / P95)")
        latency_summary = (
            df.groupby("model")["latency_ms"]
            .agg(
                p50=lambda s: float(s.quantile(0.50)),
                p95=lambda s: float(s.quantile(0.95)),
            )
            .reset_index()
            .melt(id_vars="model", value_vars=["p50", "p95"], var_name="metric", value_name="latency_ms")
        )
        st.altair_chart(
            alt.Chart(latency_summary).mark_bar().encode(
                x=alt.X("model:N", title="Model"),
                y=alt.Y("latency_ms:Q", title="Latency (ms)"),
                color=alt.Color("metric:N", title="Metric"),
                tooltip=["model:N", "metric:N", alt.Tooltip("latency_ms:Q", format=".2f")],
            ).properties(height=260),
            use_container_width=True,
        )

    with right:
        st.subheader("Complexity Score Distribution")
        st.altair_chart(
            alt.Chart(df).mark_bar().encode(
                x=alt.X("complexity_score:Q", bin=alt.Bin(step=0.1), title="Complexity score"),
                y=alt.Y("count():Q", title="Requests"),
                tooltip=[alt.Tooltip("count():Q", title="Requests")],
            ).properties(height=260),
            use_container_width=True,
        )

    lower_left, lower_right = st.columns(2)

    with lower_left:
        st.subheader("A/B Variant Comparison")
        ab_rows = [
            {
                "variant":               variant,
                "requests":              int(info["requests"]),
                "avg_latency_ms":        float(info["avg_latency_ms"]),
                "avg_cost_usd":          float(info["avg_cost_usd"]),
                "avg_completion_tokens": float(info["avg_completion_tokens"]),
            }
            for variant, info in stats["ab_comparison"].items()
        ]
        if ab_rows:
            st.dataframe(pd.DataFrame(ab_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No A/B data yet.")

    with lower_right:
        st.subheader("Cost by Tier")
        cost_rows = [{"tier": tier, "cost_usd": value} for tier, value in stats["cost_by_tier"].items()]
        if cost_rows:
            st.altair_chart(
                alt.Chart(pd.DataFrame(cost_rows)).mark_bar().encode(
                    x=alt.X("tier:N", title="Tier"),
                    y=alt.Y("cost_usd:Q", title="Cost (USD)"),
                    color=alt.Color("tier:N", legend=None),
                    tooltip=["tier:N", alt.Tooltip("cost_usd:Q", format=".6f")],
                ).properties(height=220),
                use_container_width=True,
            )
        else:
            st.caption("No tier-level cost data yet.")
