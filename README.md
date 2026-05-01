# Dynamic LLM Routing API (Smart Router)

Commercial-grade, OpenAI-compatible middleware that routes each request to the right model tier based on complexity so cost stays low and latency stays fast.

## Section 1 Complete: Foundation

This section establishes a production-grade repo layout and strict separation of concerns:

- `api/`: FastAPI transport layer only
- `core/`: pure routing/classification/context logic with no HTTP coupling
- `db/`: telemetry schema/query layer
- `dashboard/`: visual metrics layer
- `benchmarks/`: reproducible performance and ROI tests
- `tests/`: unit and integration suites

## Quickstart

```powershell
poetry install
copy .env.example .env
poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Health check:

```powershell
curl http://localhost:8000/health
```

Completions bootstrap endpoint:

```powershell
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

## Current status

- Section 1: foundation scaffold complete
- Section 2+: implementation details will be added incrementally with tests
