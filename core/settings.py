from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    together_api_key: str = Field(default="replace_me", alias="TOGETHER_API_KEY")
    together_base_url: str = Field(default="https://api.together.xyz", alias="TOGETHER_BASE_URL")

    router_mode: Literal["heuristic", "bert"] = Field(default="heuristic", alias="ROUTER_MODE")
    ab_split_ratio: float = Field(default=0.10, ge=0.0, le=1.0, alias="AB_SPLIT_RATIO")
    budget_default: Literal["cheap", "balanced", "quality"] = Field(
        default="balanced", alias="BUDGET_DEFAULT"
    )

    context_window_size: int = Field(default=5, ge=1, le=20, alias="CONTEXT_WINDOW_SIZE")
    context_max_tokens: int = Field(default=4000, ge=256, alias="CONTEXT_MAX_TOKENS")
    summarizer_model: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.2", alias="SUMMARIZER_MODEL"
    )

    small_model: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.2", alias="SMALL_MODEL"
    )
    medium_model: str = Field(default="meta-llama/Llama-2-13b-chat-hf", alias="MEDIUM_MODEL")
    large_model: str = Field(default="meta-llama/Llama-2-70b-chat-hf", alias="LARGE_MODEL")
    medium_threshold: float = Field(default=0.35, ge=0.0, le=1.0, alias="MEDIUM_THRESHOLD")
    large_threshold: float = Field(default=0.70, ge=0.0, le=1.0, alias="LARGE_THRESHOLD")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, ge=1, le=65535, alias="API_PORT")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )

    sqlite_path: str = Field(default="./data/router.db", alias="SQLITE_PATH")


@lru_cache
def get_settings() -> Settings:
    return Settings()
