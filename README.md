# Dynamic LLM Routing API (Smart Router)

Smart Router is an OpenAI-compatible middleware that cuts inference cost and latency by routing each request to the right model tier. Simple prompts go to cheaper/faster models, complex prompts go to stronger models.

## What It Does

- Accepts standard `/v1/chat/completions` requests.
- Scores prompt complexity (`0.0` to `1.0`) with pluggable classifiers.
- Routes traffic by threshold + budget mode (`cheap`, `balanced`, `quality`).
- Controls context growth with sliding window + optional summarization.
- Logs cost/latency/token telemetry to SQLite.
- Exposes dashboard + Prometheus/Grafana for ROI proof.

## Architecture

```text
Client App
   |
   v
FastAPI Proxy (/v1/chat/completions)
   |
   +--> Context Manager (window + summarize)
   |
   +--> Classifier (heuristic | bert)
   |
   +--> Model Router (tier + A/B variant)
   |
   +--> LLM Client (Together API, retries)
   |
   +--> Metrics Recorder (SQLite) + Prometheus /metrics
   |
   v
OpenAI-Compatible Response (+ x-router-model, x-tier, x-complexity-score)
```

## Repo Layout

```text
api/                  FastAPI routes + app lifecycle
core/                 business logic (no HTTP coupling)
db/                   SQLite schema + query/aggregation logic
dashboard/            Streamlit telemetry dashboard
benchmarks/           benchmark runner + query sets + results artifacts
tests/                unit + integration tests
grafana/              provisioning + dashboards
prometheus.yml        scrape config
docker-compose.yml    full stack (router/dashboard/prometheus/grafana)
```

## Quick Start

1. Install dependencies.

```powershell
python -m pip install -U pip
python -m pip install -e .
```

2. Configure environment.

```powershell
copy .env.example .env
```

3. Start API locally.

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

4. Validate health.

```powershell
curl http://localhost:8000/health
```

## Docker Stack

```powershell
docker compose up --build
```

Exposed services:

- Router API: `http://localhost:8000`
- Streamlit dashboard: `http://localhost:8501`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (default `admin/admin`)

## How Routing Works

1. Context manager adds new messages and trims/summarizes old history when token budget is exceeded.
2. Classifier scores the latest user query complexity.
3. Router selects model tier from score thresholds or explicit budget hint.
4. LLM client calls Together API with retries for transient failures.
5. Telemetry is recorded (latency, cost, tokens, tier, A/B variant).
6. Response returns OpenAI-compatible JSON + router headers.

## API Reference

### POST `/v1/chat/completions`

Request:

```json
{
  "messages": [
    { "role": "user", "content": "Explain vector databases in simple terms." }
  ],
  "budget_hint": "balanced",
  "stream": false
}
```

Response headers:

- `x-router-model`
- `x-tier`
- `x-complexity-score`
- `x-ab-variant`

Budget hints:

- `cheap`: force small tier
- `balanced`: threshold-based routing
- `quality`: force large tier

## Configuration Reference

| Variable | Default | Purpose |
| --- | --- | --- |
| `TOGETHER_API_KEY` | `replace_me` | Together API auth key |
| `TOGETHER_BASE_URL` | `https://api.together.xyz` | Together API base |
| `ROUTER_MODE` | `heuristic` | Classifier type (`heuristic` or `bert`) |
| `AB_SPLIT_RATIO` | `0.10` | Variant traffic share |
| `BUDGET_DEFAULT` | `balanced` | Default budget mode |
| `CONTEXT_WINDOW_SIZE` | `5` | Recent turns to keep raw |
| `CONTEXT_MAX_TOKENS` | `4000` | Max context token budget |
| `SUMMARIZER_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | Cheap summarizer model |
| `SMALL_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | Small tier model |
| `MEDIUM_MODEL` | `meta-llama/Llama-2-13b-chat-hf` | Medium tier model |
| `LARGE_MODEL` | `meta-llama/Llama-2-70b-chat-hf` | Large tier model |
| `MEDIUM_THRESHOLD` | `0.35` | Medium routing boundary |
| `LARGE_THRESHOLD` | `0.70` | Large routing boundary |
| `SQLITE_PATH` | `./data/router.db` | Telemetry database path |

## Benchmark Suite

Run benchmark with 1000 requests (600 simple, 300 medium, 100 complex):

```powershell
python benchmarks/run.py ^
  --together-api-key YOUR_KEY ^
  --router-url http://localhost:8000/v1/chat/completions ^
  --together-url https://api.together.xyz/v1/chat/completions ^
  --baseline-model meta-llama/Llama-2-70b-chat-hf
```

Outputs:

- Terminal summary table (cost, avg latency, p95 latency, savings).
- JSON artifact: `benchmarks/results/benchmark_<timestamp>.json`.

### Benchmark Result Table (Fill After Run)

| Metric | Baseline (Always 70B) | Smart Router |
| --- | ---: | ---: |
| Successful requests | _TBD_ | _TBD_ |
| Failed requests | _TBD_ | _TBD_ |
| Total cost (USD) | _TBD_ | _TBD_ |
| Avg latency (ms) | _TBD_ | _TBD_ |
| P95 latency (ms) | _TBD_ | _TBD_ |
| Savings (USD) | - | _TBD_ |
| Savings (%) | - | _TBD_ |

Copy the numbers from the benchmark output artifact into this table.

## Observability

- `/metrics` exposes Prometheus counters/histograms:
  - `requests_total{tier=...}`
  - `cost_usd_total{tier=...}`
  - `latency_ms_histogram{tier=...}`
- Streamlit dashboard reads SQLite aggregates (cost by tier, p95, A/B comparison, complexity distribution).
- Grafana dashboard is pre-provisioned from `grafana/dashboards/smart-router.json`.

## Development

Run tests:

```powershell
python -m pytest -q
```

Current status:

- Section 1: foundation
- Section 2: classifier
- Section 3: context manager
- Section 4: model router + A/B
- Section 5: proxy pipeline + Together client
- Section 6: metrics DB + dashboard
- Section 7: Docker + Prometheus + Grafana
- Section 8: benchmark suite + README
