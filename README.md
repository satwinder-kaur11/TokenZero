# TokenZero — Dynamic LLM Smart Router

> Route every prompt to the **right model at the right cost** — automatically.  
> 100 % local · no API keys · powered by [Ollama](https://ollama.com)

TokenZero is an **OpenAI-compatible middleware API** that sits between your application and your local LLMs. It scores each incoming prompt for complexity, then routes it to a small, medium, or large Ollama model — cutting unnecessary inference cost and latency without sacrificing quality on hard tasks.

---

## Table of Contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [MCP Tool — `route_request`](#mcp-tool--route_request)
4. [Model Tiers](#model-tiers)
5. [Full Setup — New Machine](#full-setup--new-machine)
6. [Running the Project](#running-the-project)
7. [API Reference](#api-reference)
8. [Testing](#testing)
9. [Telemetry Dashboard](#telemetry-dashboard)
10. [Docker (Optional)](#docker-optional)
11. [How AI Tools Were Used](#how-ai-tools-were-used)

---

## What it does

Most LLM applications send every query to the same big model — even when a tiny one would do the job perfectly. TokenZero fixes that:

| Prompt | Complexity Score | Routed to |
|--------|-----------------|-----------|
| `"What is 2 + 2?"` | 0.05 | `llama3.2:1b` (Small) |
| `"Summarise the key differences between REST and GraphQL."` | 0.27 | `llama3.2:3b` (Medium) |
| `"Design a multi-agent fraud detection system with sub-100ms SLAs."` | 0.72 | `llama3.1:8b` (Large) |

**Key features:**

- 🔀 **Heuristic or BERT-based** complexity scoring (configurable)
- 💰 **Budget hints** — `cheap` / `balanced` / `quality` per request
- 📊 **A/B variant tracking** for model comparison experiments
- 🗜️ **Sliding-window context manager** with automatic summarisation
- 📈 **Prometheus metrics + Streamlit dashboard** for real-time observability
- 🛠️ **MCP-compatible tool** (`route_request`) — any AI agent can call it
- 🔒 **Zero external API keys** — everything runs locally via Ollama

---

## Architecture

```
Your App / AI Agent
        │
        ▼
┌───────────────────────────────────────┐
│         TokenZero  (FastAPI)          │
│                                       │
│  POST /v1/chat/completions            │
│  POST /tools/route_request  ◄── MCP   │
│                                       │
│  ┌─────────────┐  ┌───────────────┐  │
│  │  Classifier │  │ Context Mgr   │  │
│  │  (scoring)  │  │ (sliding win) │  │
│  └──────┬──────┘  └───────────────┘  │
│         │                            │
│  ┌──────▼──────────────────────────┐ │
│  │         Model Router            │ │
│  │  cheap → small                  │ │
│  │  balanced → heuristic tier      │ │
│  │  quality → large                │ │
│  └──────┬──────────────────────────┘ │
│         │                            │
│  ┌──────▼──────────────────────────┐ │
│  │     LLM Client (httpx)          │ │
│  │  ollama_base_url/v1/chat/...    │ │
│  └─────────────────────────────────┘ │
└───────────────────────────────────────┘
        │
        ▼
  Ollama  (localhost:11434)
  ├── llama3.2:1b   ← small
  ├── llama3.2:3b   ← medium
  └── llama3.1:8b   ← large
```

---

## MCP Tool — `route_request`

TokenZero exposes a **Model Context Protocol (MCP)-compatible tool** at `GET /tools/schema` + `POST /tools/route_request`. Any AI agent (Claude, LangChain, AutoGen, etc.) can discover and call this tool autonomously.

### JSON Schema Definition

```json
{
  "name": "route_request",
  "description": "Routes a prompt to the optimal LLM tier based on complexity scoring. Returns the selected model, tier, complexity score, and estimated cost.",
  "input_schema": {
    "type": "object",
    "properties": {
      "prompt": {
        "type": "string",
        "description": "The user prompt text to route."
      },
      "budget_hint": {
        "type": "string",
        "enum": ["cheap", "balanced", "quality"],
        "default": "balanced"
      }
    },
    "required": ["prompt"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "tool":              { "type": "string" },
      "selected_model":   { "type": "string" },
      "tier":             { "type": "string", "enum": ["small", "medium", "large"] },
      "complexity_score": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
      "estimated_cost_usd": { "type": "number" },
      "budget_hint":      { "type": "string" }
    },
    "required": ["tool", "selected_model", "tier", "complexity_score", "estimated_cost_usd"]
  }
}
```

### Example Tool Call

```bash
# Discover the tool (what an agent does on startup)
curl http://localhost:8000/tools/schema

# Call the tool
curl -X POST http://localhost:8000/tools/route_request \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain transformer attention mechanisms.", "budget_hint": "balanced"}'
```

**Example response:**

```json
{
  "tool": "route_request",
  "selected_model": "llama3.2:3b",
  "tier": "medium",
  "complexity_score": 0.3600,
  "estimated_cost_usd": 0.00015,
  "budget_hint": "balanced"
}
```

---

## Model Tiers

| Tier | Default Model | RAM | Use Case |
|------|--------------|-----|----------|
| **Small** | `llama3.2:1b` | ~1.3 GB | Greetings, maths, lookup, yes/no |
| **Medium** | `llama3.2:3b` | ~2.0 GB | Summaries, comparisons, short code |
| **Large** | `llama3.1:8b` | ~4.9 GB | Architecture, reasoning, long code |

Thresholds (configurable in `.env`):

```
score < 0.35  → small
score < 0.70  → medium
score ≥ 0.70  → large
```

---

## Full Setup — New Machine

Follow these steps **exactly** on a brand-new Windows, macOS, or Linux machine.

### Step 1 — Install Prerequisites

#### Python 3.11+
Download from [python.org](https://www.python.org/downloads/) and make sure it's on your PATH.

```powershell
python --version   # should print 3.11.x or higher
```

#### Poetry (Python dependency manager)

```powershell
# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# macOS / Linux
curl -sSL https://install.python-poetry.org | python3 -
```

After install, add Poetry to your PATH (the installer prints the exact path).  
Verify:

```powershell
poetry --version
```

#### Git

Download from [git-scm.com](https://git-scm.com/downloads) if not already installed.

```powershell
git --version
```

#### Ollama

Download and install from [ollama.com/download](https://ollama.com/download).  
Ollama runs a local LLM server at `http://localhost:11434`.

```powershell
ollama --version
```

---

### Step 2 — Clone the Repository

```powershell
git clone https://github.com/your-username/TokenZero.git
cd TokenZero
```

---

### Step 3 — Pull Ollama Models

Start Ollama (if it is not already running as a service):

```powershell
ollama serve
```

Open a **new terminal** in the `TokenZero` folder and pull all three tier models:

```powershell
# Small tier  — fastest, 1.3 GB
ollama pull llama3.2:1b

# Medium tier — balanced, 2.0 GB
ollama pull llama3.2:3b

# Large tier  — highest quality, 4.9 GB
ollama pull llama3.1:8b
```

Verify all three are available:

```powershell
ollama list
```

You should see `llama3.2:1b`, `llama3.2:3b`, and `llama3.1:8b` in the list.

---

### Step 4 — Install Python Dependencies

```powershell
# From inside the TokenZero directory
poetry install
```

This creates an isolated virtual environment and installs all packages from `pyproject.toml`.

---

### Step 5 — Configure Environment

```powershell
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

The default `.env` is already configured for local Ollama — **no changes needed**. It looks like this:

```ini
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434

SMALL_MODEL=llama3.2:1b
MEDIUM_MODEL=llama3.2:3b
LARGE_MODEL=llama3.1:8b
SUMMARIZER_MODEL=llama3.2:1b

ROUTER_MODE=heuristic
MEDIUM_THRESHOLD=0.35
LARGE_THRESHOLD=0.70
BUDGET_DEFAULT=balanced

API_HOST=0.0.0.0
API_PORT=8000
SQLITE_PATH=./data/router.db
```

Create the data directory for SQLite:

```powershell
mkdir data
```

---

### Step 6 — Start the API Server

```powershell
poetry run python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

You should see:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Running the Project

### Quick Verification

Open a **second terminal** and run:

```powershell
# 1. Health check
Invoke-WebRequest -Uri http://localhost:8000/health -UseBasicParsing | Select-Object -ExpandProperty Content
# Expected: {"status":"ok"}

# 2. Fetch the MCP tool schema
Invoke-WebRequest -Uri http://localhost:8000/tools/schema -UseBasicParsing | Select-Object -ExpandProperty Content

# 3. Route a simple prompt → should pick llama3.2:1b (small)
$body = '{"prompt": "What is 2 + 2?", "budget_hint": "cheap"}'
(Invoke-WebRequest -Uri http://localhost:8000/tools/route_request -Method POST -ContentType "application/json" -Body $body -UseBasicParsing).Content

# 4. Route a complex prompt → should pick llama3.1:8b (large)
$body = '{"prompt": "Design a distributed fraud detection system with real-time stream processing, explain and compare each architectural trade-off step by step.", "budget_hint": "quality"}'
(Invoke-WebRequest -Uri http://localhost:8000/tools/route_request -Method POST -ContentType "application/json" -Body $body -UseBasicParsing).Content

# 5. Run the full MCP Agent Demo
poetry run python agent_demo.py
```

### OpenAI-Compatible Chat Endpoint

Send a chat message just like you would to OpenAI:

```powershell
# Windows PowerShell
$body = @"
{
  "messages": [{"role": "user", "content": "Explain vector databases in simple terms."}],
  "budget_hint": "balanced"
}
"@
Invoke-WebRequest -Uri http://localhost:8000/v1/chat/completions `
  -Method POST `
  -ContentType "application/json" `
  -Body $body `
  -UseBasicParsing | Select-Object -ExpandProperty Content
```

```bash
# macOS / Linux (curl)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain vector databases in simple terms."}],
    "budget_hint": "balanced"
  }'
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/tools/schema` | MCP JSON Schema for `route_request` |
| `POST` | `/tools/route_request` | Route a prompt, get model + tier decision |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat (routes + calls Ollama) |
| `GET` | `/metrics` | Prometheus metrics scrape endpoint |

**Interactive docs** (auto-generated by FastAPI):
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Testing

Run the full test suite (34 tests, no server required):

```powershell
poetry run pytest tests/ -v
```

Expected output:

```
34 passed in ~1s
```

Run a specific test file:

```powershell
poetry run pytest tests/test_llm_client.py -v
poetry run pytest tests/test_router.py -v
poetry run pytest tests/test_context_manager.py -v
poetry run pytest tests/test_settings.py -v
```

---

## Telemetry Dashboard

The Streamlit dashboard shows live request counts, tier distribution, cost savings, latency, and A/B variant results.

```powershell
poetry run streamlit run dashboard/app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501) in your browser. The dashboard auto-refreshes every 5 seconds.

---

## Docker (Optional)

Run the entire stack (API + Dashboard + Prometheus + Grafana) with one command:

```powershell
docker compose up --build
```

| Service | URL |
|---------|-----|
| Router API | http://localhost:8000 |
| Streamlit Dashboard | http://localhost:8501 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

> **Note for Docker on Windows/macOS:** If Ollama is running on your host machine (not inside Docker), update `.env`:
> ```ini
> OLLAMA_BASE_URL=http://host.docker.internal:11434
> ```

---

## How AI Tools Were Used

This project was scaffolded and iterated on with the help of AI coding assistants — specifically **Claude** (via the **Antigravity / Cursor** environment).

### What AI helped with

| Area | AI Contribution |
|------|----------------|
| **Boilerplate generation** | Initial FastAPI app structure, router, settings, and Pydantic models were scaffolded via prompts like *"create an OpenAI-compatible FastAPI endpoint that routes to different models based on complexity"* |
| **Ollama migration** | All Gemini API references were replaced with Ollama's `/v1/chat/completions` endpoint — AI identified every affected file (`llm_client.py`, `context_manager.py`, `settings.py`, tests) and rewrote them consistently |
| **Test rewriting** | Gemini-specific test cases (`test_llm_client.py`, `test_context_manager.py`, `test_settings.py`) were updated to test Ollama behaviour, with new mock-transport patterns |
| **Bug fixing** | The `HeuristicClassifier(settings=settings)` instantiation bug in `tools.py` was caught and fixed; the missing `pythonpath = ["."]` in `pyproject.toml` that caused pytest collection errors was added |
| **README & docs** | This README was written with AI assistance — structure, command examples, JSON schema docs, and the architecture diagram were all generated and refined through prompting |

### Workflow used

1. **Describe intent** → *"Switch the entire project from Gemini to Ollama, keeping MCP tools working"*
2. **AI reads the codebase** → scans all files, identifies dependencies and side-effects
3. **AI makes targeted edits** → replaces only what needs changing, preserves existing logic
4. **Human reviews diffs** → each file change was inspected before accepting
5. **Run tests to verify** → `poetry run pytest tests/ -v` confirmed 34/34 passing after every change
6. **Live end-to-end test** → API server started, all endpoints hit manually, agent demo confirmed working

> AI tools dramatically accelerated the migration from ~2 hours of manual search-and-replace to a fully tested, working state in one session — without introducing regressions.

---

## Project Structure

```
TokenZero/
├── api/
│   ├── main.py                 # FastAPI app factory + lifespan
│   └── routes/
│       ├── completions.py      # POST /v1/chat/completions
│       ├── metrics.py          # GET /metrics (Prometheus)
│       └── tools.py            # GET /tools/schema · POST /tools/route_request
├── core/
│   ├── classifier.py           # HeuristicClassifier + BERTClassifier
│   ├── classifier_factory.py   # Factory: returns correct classifier from settings
│   ├── context_manager.py      # Sliding-window context + Ollama summariser
│   ├── llm_client.py           # HTTP client for Ollama / Together / mock
│   ├── prometheus_metrics.py   # Prometheus counters and histograms
│   ├── router.py               # ModelRouter: score → tier → model decision
│   └── settings.py             # Pydantic-settings config (reads .env)
├── db/
│   ├── queries.py              # Async SQLite read/write helpers
│   └── schema.py               # DB schema definition
├── dashboard/
│   └── app.py                  # Streamlit telemetry dashboard
├── tests/                      # 34 pytest tests (no Ollama required)
├── benchmarks/                 # Benchmark runner utilities
├── agent_demo.py               # MCP agent demo script
├── .env                        # Your local config (not committed)
├── .env.example                # Template — copy to .env
├── pyproject.toml              # Poetry dependencies + pytest config
├── docker-compose.yml          # Full observability stack
└── prometheus.yml              # Prometheus scrape config
```

---

## License

MIT — use freely, attribution appreciated.
