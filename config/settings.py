"""Central configuration: constants, secrets loading, and YAML helpers.

Precedence for every setting: environment variable > st.secrets > default.
Secrets are never hardcoded and never logged.
"""
from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, TypeVar

import yaml

from config.errors import ConfigError

try:  # .env is optional; only used for local development
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

__all__ = [
    "BASE_DIR",
    "CATEGORIES",
    "PRIORITIES",
    "STATUSES",
    "STATUS_TRANSITIONS",
    "ConfigError",
    "Settings",
    "get_settings",
    "load_yaml",
    "configure_logging",
    "cache_resource",
]

BASE_DIR: Path = Path(__file__).resolve().parent.parent

CATEGORIES: tuple[str, ...] = (
    "Account",
    "Payment",
    "Technical",
    "Refund",
    "Delivery",
    "Subscription",
    "General",
    "Other",
)
PRIORITIES: tuple[str, ...] = ("High", "Medium", "Low")
STATUSES: tuple[str, ...] = (
    "new",
    "triaged",
    "needs_review",
    "in_progress",
    "resolved",
    "closed",
)
STATUS_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "new": ("triaged", "needs_review"),
    "triaged": ("in_progress",),
    "needs_review": ("triaged", "in_progress"),
    "in_progress": ("resolved",),
    "resolved": ("closed", "in_progress"),
    "closed": (),
}

# Input limits (mirrored by CHECK constraints in sql/schema.sql)
MAX_NAME_LEN = 100
MAX_EMAIL_LEN = 254
MAX_SUBJECT_LEN = 200
MAX_MESSAGE_LEN = 5000
MIN_MESSAGE_LEN = 10

BASE_MODEL_NAME = "distilbert-base-uncased"

T = TypeVar("T")


def cache_resource(func: Callable[..., T]) -> Callable[..., T]:
    """Singleton cache: ``st.cache_resource`` if Streamlit is installed, else lru_cache."""
    try:
        import streamlit as st
    except ImportError:
        return functools.lru_cache(maxsize=None)(func)
    return st.cache_resource(show_spinner=False)(func)


def _get_secret(name: str, default: str | None = None) -> str | None:
    """Read a setting from env vars, then st.secrets, then the default."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets file / not running under Streamlit
        pass
    return default


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its content as a dict."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path.name}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path.name}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} must contain a mapping at top level")
    return data


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    supabase_url: str | None
    supabase_key: str | None
    confidence_threshold: float
    page_size: int
    max_length: int
    top_k: int
    log_level: str
    model_name: str
    model_dir: Path
    labels_path: Path
    priority_rules_path: Path
    response_templates_path: Path

    @property
    def supabase_configured(self) -> bool:
        """True when both Supabase settings are present (no network call)."""
        return bool(self.supabase_url and self.supabase_key)

    def require_supabase(self) -> tuple[str, str]:
        """Return (url, key) or raise ConfigError if either is missing."""
        if not self.supabase_url or not self.supabase_key:
            raise ConfigError(
                "Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY "
                "in .env or .streamlit/secrets.toml."
            )
        return self.supabase_url, self.supabase_key


def _parse_float(name: str, raw: str | None, default: float) -> float:
    try:
        return float(raw) if raw is not None else default
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc


def _parse_int(name: str, raw: str | None, default: int) -> int:
    try:
        return int(raw) if raw is not None else default
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build and validate settings once per process."""
    threshold = _parse_float(
        "CONFIDENCE_THRESHOLD", _get_secret("CONFIDENCE_THRESHOLD"), 0.60
    )
    if not 0.0 <= threshold <= 1.0:
        raise ConfigError("CONFIDENCE_THRESHOLD must be between 0 and 1")

    page_size = _parse_int("PAGE_SIZE", _get_secret("PAGE_SIZE"), 20)
    if page_size < 1 or page_size > 100:
        raise ConfigError("PAGE_SIZE must be between 1 and 100")

    return Settings(
        supabase_url=_get_secret("SUPABASE_URL"),
        supabase_key=_get_secret("SUPABASE_KEY"),
        confidence_threshold=threshold,
        page_size=page_size,
        max_length=128,
        top_k=3,
        log_level=(_get_secret("LOG_LEVEL", "INFO") or "INFO").upper(),
        model_name=BASE_MODEL_NAME,
        model_dir=BASE_DIR / "models" / "ticket_classifier",
        labels_path=BASE_DIR / "ml" / "labels_bitext8.json",
        priority_rules_path=BASE_DIR / "config" / "priority_rules.yaml",
        response_templates_path=BASE_DIR / "config" / "response_templates.yaml",
    )


def configure_logging() -> None:
    """Configure root logging once. Never log personal data or secrets."""
    level = getattr(logging, get_settings().log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
