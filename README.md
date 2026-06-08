# Dynamic LLM Routing API (Smart Router)

Smart Router is an OpenAI-compatible middleware that cuts inference cost and latency by routing each request to the right model tier. Simple prompts go to cheaper/faster models, complex prompts go to stronger models.

## Prerequisites

To run this project correctly on any device without errors, you will need the following installed:

1. **Python 3.9+**: [Download and install Python](https://www.python.org/downloads/).
2. **Git**: [Download and install Git](https://git-scm.com/downloads).
3. **Ollama**: [Download and install Ollama](https://ollama.com/download) for running local LLM models.
4. **Docker & Docker Compose (Optional)**: Needed if you want to run the full observability stack (Prometheus + Grafana).

## 1. Setup Instructions


### 2. Install Python Dependencies



# Upgrade pip and install the project dependencies
python -m pip install -U pip
python -m pip install -e .
pip install prometheus_client
python -m pip install aiosqlite  # Required for async DB operations
```

### 3. Install and Setup Ollama Models

Start the Ollama server in a background terminal if it is not already running:

```powershell
ollama serve
```

Open a new terminal and pull the required models for your routing tiers. Make sure you pull the models that correspond to your small, medium, and large tiers:

```powershell
# Pull the summarizer and small tier model
ollama pull mistral
SUMMARIZER_MODE: ollama pull llama3.2:latest

SMALL_MODEL: ollama pull llama3.2:1b      # Super fast, 1.3 GB edge model
MEDIUM_MODEL: ollama pull llama3.2:latest    # Balanced, 2.0 GB 3B model
LARGE_MODEL: ollama pull llama3.1:8b
```
*(You can use any models you prefer, just make sure to update your `.env` configuration to match).*

### 4. Configure Environment Variables

Copy the example environment file:

```powershell
copy .env.example .env
```

Open the `.env` file and adjust it to point to your local Ollama instance (since Ollama provides an OpenAI-compatible API):

```ini
# Point to your local Ollama server's OpenAI compatibility endpoint
TOGETHER_BASE_URL="http://localhost:11434/v1"
TOGETHER_API_KEY="ollama" # Ollama does not require a real API key

# Update model names to exactly match the tags you pulled in Ollama
SUMMARIZER_MODEL="mistral"
SMALL_MODEL="mistral"
MEDIUM_MODEL="llama3"
LARGE_MODEL="llama3:70b"

SQLITE_PATH="./data/router.db"
```

### 5. Start the Smart Router API

Ensure your virtual environment is activated, then run:

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Validate the health of the API:

```powershell
curl http://localhost:8000/health
```

### 6. Run the Telemetry Dashboard

Open another terminal session and run the Streamlit dashboard:

```powershell
# Windows

streamlit run dashboard/app.py
```
This will open the dashboard UI at `http://localhost:8501`.

### 7. Run Full Stack with Docker (Optional)

To start the API, Dashboard, Prometheus, and Grafana all at once using Docker:

```powershell
docker compose up --build
```

**Exposed services in Docker:**
- Router API: `http://localhost:8000`
- Streamlit dashboard: `http://localhost:8501`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (default login: `admin` / `admin`)

*(Important: If running in Docker but your Ollama is running on the host machine, you may need to change your `.env` to `TOGETHER_BASE_URL=http://host.docker.internal:11434/v1` so the container can reach your host's Ollama).*

## API Usage Example

Once the server is running, you can test the routing by sending a standard OpenAI-style request:

```powershell
curl -X POST http://localhost:8000/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d '{
    "messages": [
      { "role": "user", "content": "Explain vector databases in simple terms." }
    ],
    "budget_hint": "balanced"
  }'
```
