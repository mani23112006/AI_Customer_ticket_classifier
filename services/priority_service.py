"""Rule-based priority engine. Independent of the ML model; rules live in YAML."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.errors import ConfigError
from config.settings import CATEGORIES, PRIORITIES, get_settings, load_yaml


@dataclass(frozen=True)
class PriorityResult:
    """Assigned priority plus the rule that produced it."""

    priority: str
    rule_id: str
    description: str


@dataclass(frozen=True)
class _HighRule:
    rule_id: str
    description: str
    patterns: tuple[re.Pattern[str], ...]


class PriorityEngine:
    """High (keyword rules) -> Medium (by category) -> Low (default)."""

    def __init__(
        self,
        high_rules: list[_HighRule],
        medium_categories: frozenset[str],
        default_priority: str,
    ) -> None:
        self._high_rules = high_rules
        self._medium_categories = medium_categories
        self._default = default_priority

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PriorityEngine":
        """Build the engine from parsed YAML, validating everything up front."""
        high_rules: list[_HighRule] = []
        try:
            for raw in config.get("high_priority_rules", []):
                compiled = tuple(re.compile(p, re.IGNORECASE) for p in raw["patterns"])
                high_rules.append(
                    _HighRule(str(raw["id"]), str(raw.get("description", "")), compiled)
                )
        except (KeyError, TypeError, re.error) as exc:
            raise ConfigError("Invalid high_priority_rules in priority_rules.yaml") from exc

        medium = frozenset(config.get("medium_categories", []))
        unknown = medium - set(CATEGORIES)
        if unknown:
            raise ConfigError(f"Unknown categories in medium_categories: {sorted(unknown)}")

        default = config.get("default_priority", "Low")
        if default not in PRIORITIES:
            raise ConfigError("default_priority must be High, Medium or Low")
        return cls(high_rules, medium, default)

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> "PriorityEngine":
        """Load rules from YAML (default path from settings)."""
        return cls.from_config(load_yaml(path or get_settings().priority_rules_path))

    def assign(self, subject: str, message: str, category: str | None) -> PriorityResult:
        """Return the priority and matched rule for a ticket."""
        text = f"{subject or ''} {message or ''}"
        for rule in self._high_rules:
            if any(p.search(text) for p in rule.patterns):
                return PriorityResult("High", f"high:{rule.rule_id}", rule.description)
        if category in self._medium_categories:
            return PriorityResult(
                "Medium", f"medium:category:{category}", f"Category {category} defaults to Medium"
            )
        return PriorityResult(self._default, "default", "No high-priority rule or medium category matched")
