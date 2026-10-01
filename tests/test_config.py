"""Smoke tests for configuration, YAML files and label consistency."""
import json
import re

import pytest

from config.settings import (
    BASE_DIR,
    CATEGORIES,
    ConfigError,
    STATUS_TRANSITIONS,
    STATUSES,
    get_settings,
    load_yaml,
)


def test_settings_defaults():
    get_settings.cache_clear()
    settings = get_settings()
    assert 0.0 <= settings.confidence_threshold <= 1.0
    assert settings.page_size >= 1


def test_invalid_threshold(monkeypatch):
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "1.5")
    get_settings.cache_clear()
    with pytest.raises(ConfigError):
        get_settings()
    monkeypatch.delenv("CONFIDENCE_THRESHOLD")
    get_settings.cache_clear()


def test_require_supabase_raises_when_missing(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    if not settings.supabase_configured:
        with pytest.raises(ConfigError):
            settings.require_supabase()
    get_settings.cache_clear()


def test_status_transitions_cover_all_statuses():
    assert set(STATUS_TRANSITIONS) == set(STATUSES)
    for targets in STATUS_TRANSITIONS.values():
        assert set(targets) <= set(STATUSES)


def test_priority_rules_compile():
    rules = load_yaml(get_settings().priority_rules_path)
    for rule in rules["high_priority_rules"]:
        for pattern in rule["patterns"]:
            re.compile(pattern, re.IGNORECASE)
    assert set(rules["medium_categories"]) <= set(CATEGORIES)


def test_templates_cover_all_categories():
    data = load_yaml(get_settings().response_templates_path)
    assert set(data["templates"]) == set(CATEGORIES)
    for text in list(data["templates"].values()) + [data["fallback"]]:
        assert "{customer_name}" in text and "{ticket_no}" in text


def test_labels_json_matches_categories():
    labels = json.loads((BASE_DIR / "ml" / "labels.json").read_text(encoding="utf-8"))["labels"]
    assert labels == list(CATEGORIES)
    bitext_labels = json.loads((BASE_DIR / "ml" / "labels_bitext8.json").read_text(encoding="utf-8"))["labels"]
    assert bitext_labels == list(CATEGORIES)
