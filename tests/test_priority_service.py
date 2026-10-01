"""Tests for the rule-based priority engine (uses the real YAML rules)."""
import pytest

from config.errors import ConfigError
from services.priority_service import PriorityEngine


@pytest.fixture(scope="module")
def engine():
    return PriorityEngine.from_yaml()


@pytest.mark.parametrize(
    "text,rule",
    [
        ("Someone hacked my account", "high:hacked"),
        ("There is an unauthorized charge", "high:unauthorized"),
        ("This looks like FRAUD to me", "high:fraud"),
        ("My package was stolen from the door", "high:stolen"),
        ("The account was compromised last night", "high:account_compromised"),
        ("Data breach notice received", "high:account_compromised"),
        ("Money was deducted twice from my bank", "high:money_deducted"),
        ("The payment of $50 got deducted but no receipt", "high:money_deducted"),
        ("Amount deducted from card", "high:money_deducted"),
    ],
)
def test_high_priority_rules(engine, text, rule):
    result = engine.assign("Help", text, "General")
    assert result.priority == "High"
    assert result.rule_id == rule


@pytest.mark.parametrize("category", ["Payment", "Refund", "Technical", "Account", "Delivery"])
def test_medium_categories(engine, category):
    result = engine.assign("Question", "please help with my thing", category)
    assert result.priority == "Medium"
    assert result.rule_id == f"medium:category:{category}"


@pytest.mark.parametrize("category", ["Subscription", "General", "Other", None])
def test_low_priority_default(engine, category):
    result = engine.assign("Question", "please help with my thing", category)
    assert result.priority == "Low"
    assert result.rule_id == "default"


def test_high_rule_beats_category(engine):
    assert engine.assign("Refund", "unauthorized transaction", "Refund").priority == "High"


def test_no_false_positive_on_similar_words(engine):
    assert engine.assign("Hi", "I will deduct nothing and the hackathon was fun", "General").priority == "Low"


def test_invalid_config_raises():
    with pytest.raises(ConfigError):
        PriorityEngine.from_config({"high_priority_rules": [{"id": "x", "patterns": ["("]}]})
    with pytest.raises(ConfigError):
        PriorityEngine.from_config({"medium_categories": ["Nonsense"]})
    with pytest.raises(ConfigError):
        PriorityEngine.from_config({"default_priority": "Urgent"})
