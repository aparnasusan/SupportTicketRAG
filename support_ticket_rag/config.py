"""Validated operational configuration shared by every entry point."""

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, SettingsError

from support_ticket_rag.validation import validate_model

PROJECT_DIR = Path(__file__).resolve().parent.parent
# Changing embeddings requires rebuilding and evaluating the index.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def validate_origin(value: str) -> str:
    """Accept an explicit HTTP origin, never a path, wildcard, or credential."""
    value = value.strip()
    parts = urlsplit(value)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.path not in {"", "/"}
        or parts.query
        or parts.fragment
        or "*" in value
        or any(c.isspace() for c in value)
        or "?" in value
        or "#" in value
        or "\\" in value
    ):
        raise ValueError("must be an explicit HTTP(S) origin")
    _ = parts.port  # Also rejects malformed and out-of-range ports.
    return value.rstrip("/")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
    )
    chroma_path: Path = PROJECT_DIR / "storage" / "chroma"
    chroma_collection: str = "support_tickets"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    healthcheck_timeout_seconds: float = Field(default=3, gt=0, allow_inf_nan=False)
    cors_allowed_origins: tuple[str, ...] = ()

    @field_validator("chroma_path", mode="before")
    @classmethod
    def resolve_path(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("must not be blank")
        path = Path(value).expanduser()
        return (path if path.is_absolute() else PROJECT_DIR / path).resolve()

    @field_validator("ollama_model")
    @classmethod
    def model_name(cls, value: str) -> str:
        return validate_model(value)

    @field_validator("chroma_collection")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ord(c) < 32 for c in value):
            raise ValueError("must be nonblank text without control characters")
        return value

    @field_validator("ollama_base_url")
    @classmethod
    def base_url(cls, value: str) -> str:
        return validate_origin(value)

    @field_validator("cors_allowed_origins")
    @classmethod
    def origins(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(validate_origin(value) for value in values))


class ConfigurationError(ValueError):
    """A safe startup error containing field names, never configuration values."""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as error:
        fields = sorted({str(item["loc"][0]) for item in error.errors()})
        raise ConfigurationError("Invalid configuration fields: " + ", ".join(fields)) from None
    except SettingsError:
        raise ConfigurationError(
            "Invalid environment configuration; check .env value formats."
        ) from None
