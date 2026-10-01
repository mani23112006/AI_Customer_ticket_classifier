"""Template-based suggested replies (YAML templates, {customer_name}/{ticket_no})."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from config.errors import ConfigError
from config.settings import get_settings, load_yaml


class _SafeDict(dict):
    """Leaves unknown placeholders untouched instead of raising KeyError."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class ResponseService:
    """Fills a per-category template. Values are inserted, never re-parsed."""

    def __init__(self, templates: dict[str, str], fallback: str) -> None:
        self._templates = templates
        self._fallback = fallback

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ResponseService":
        templates = config.get("templates")
        fallback = config.get("fallback")
        if not isinstance(templates, dict) or not isinstance(fallback, str):
            raise ConfigError("response_templates.yaml needs 'templates' and 'fallback'")
        return cls({str(k): str(v) for k, v in templates.items()}, fallback)

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> "ResponseService":
        return cls.from_config(load_yaml(path or get_settings().response_templates_path))

    def suggest(self, category: str | None, customer_name: str, ticket_no: str) -> str:
        """Return the reply text for a category (fallback if unknown/None)."""
        values = _SafeDict(customer_name=customer_name, ticket_no=ticket_no)
        template = self._templates.get(category or "", self._fallback)
        try:
            return template.format_map(values).strip()
        except (ValueError, IndexError, AttributeError):  # malformed template braces
            return self._fallback.format_map(values).strip()
