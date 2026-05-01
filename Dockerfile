FROM python:3.11-slim AS builder

ENV POETRY_VERSION=1.8.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://install.python-poetry.org | python -

WORKDIR /app
COPY pyproject.toml README.md /app/
RUN /opt/poetry/bin/poetry export -f requirements.txt --output /app/requirements.txt --without-hashes

FROM python:3.11-slim AS final

ENV TOGETHER_API_KEY="" \
    ROUTER_MODE="heuristic" \
    AB_SPLIT_RATIO="0.1" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd -m -u 10001 appuser
WORKDIR /app

COPY --from=builder /app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
