from core.settings import get_settings


def test_settings_defaults() -> None:
    settings = get_settings()
    assert settings.router_mode in {"heuristic", "bert"}
    assert settings.llm_backend in {"together", "gemini", "mock"}
    assert 0.0 <= settings.ab_split_ratio <= 1.0
    assert settings.context_window_size >= 1
