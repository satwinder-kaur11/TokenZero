SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    model TEXT,
    tier TEXT,
    complexity_score REAL,
    latency_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd REAL,
    ab_variant TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
"""
