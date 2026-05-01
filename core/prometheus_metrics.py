from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest


class RouterPrometheusMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests_total = Counter(
            "requests_total",
            "Total routed requests",
            labelnames=("tier",),
            registry=self.registry,
        )
        self.cost_usd_total = Counter(
            "cost_usd_total",
            "Total routed cost in USD",
            labelnames=("tier",),
            registry=self.registry,
        )
        self.latency_ms_histogram = Histogram(
            "latency_ms_histogram",
            "Request latency in milliseconds",
            labelnames=("tier",),
            buckets=(25, 50, 100, 200, 400, 800, 1200, 2000, 5000),
            registry=self.registry,
        )

    def observe_completion(self, *, tier: str, cost_usd: float, latency_ms: float) -> None:
        safe_tier = tier or "unknown"
        self.requests_total.labels(tier=safe_tier).inc()
        self.cost_usd_total.labels(tier=safe_tier).inc(max(0.0, cost_usd))
        self.latency_ms_histogram.labels(tier=safe_tier).observe(max(0.0, latency_ms))

    def render(self) -> bytes:
        return generate_latest(self.registry)
