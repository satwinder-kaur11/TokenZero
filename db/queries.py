from __future__ import annotations

from pathlib import Path

import aiosqlite

from db.schema import SCHEMA_SQL


async def init_db(sqlite_path: str) -> None:
    path = Path(sqlite_path)
    if path.parent and str(path.parent) not in {"", "."}:
        path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(path) as conn:
        await conn.execute(SCHEMA_SQL)
        await conn.commit()


async def insert_request(
    sqlite_path: str,
    *,
    ts: float,
    model: str,
    tier: str,
    complexity_score: float,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    ab_variant: str,
) -> None:
    async with aiosqlite.connect(sqlite_path) as conn:
        await conn.execute(
            """
            INSERT INTO requests (
                ts, model, tier, complexity_score, latency_ms,
                prompt_tokens, completion_tokens, cost_usd, ab_variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                model,
                tier,
                complexity_score,
                latency_ms,
                prompt_tokens,
                completion_tokens,
                cost_usd,
                ab_variant,
            ),
        )
        await conn.commit()


class MetricsRecorder:
    def __init__(self, sqlite_path: str) -> None:
        self.sqlite_path = sqlite_path

    async def init(self) -> None:
        await init_db(self.sqlite_path)

    async def record(
        self,
        *,
        ts: float,
        model: str,
        tier: str,
        complexity_score: float,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        ab_variant: str,
    ) -> None:
        await insert_request(
            self.sqlite_path,
            ts=ts,
            model=model,
            tier=tier,
            complexity_score=complexity_score,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            ab_variant=ab_variant,
        )
