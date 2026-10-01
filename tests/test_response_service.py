"""Tests for template filling."""
from config.settings import CATEGORIES
from services.response_service import ResponseService


def test_fills_placeholders_for_every_category():
    service = ResponseService.from_yaml()
    for category in CATEGORIES:
        reply = service.suggest(category, "Asha", "TCK-000042")
        assert "Asha" in reply and "TCK-000042" in reply
        assert "{" not in reply and "}" not in reply


def test_unknown_or_missing_category_uses_fallback():
    service = ResponseService.from_yaml()
    for category in (None, "Nonexistent"):
        reply = service.suggest(category, "Asha", "TCK-000001")
        assert "Asha" in reply and "TCK-000001" in reply


def test_customer_name_with_braces_is_not_interpreted():
    service = ResponseService({"Payment": "Hi {customer_name} ({ticket_no})"}, "fallback")
    reply = service.suggest("Payment", "{ticket_no}{0}", "TCK-1")
    assert reply == "Hi {ticket_no}{0} (TCK-1)"


def test_unknown_placeholder_is_left_alone():
    service = ResponseService({"Payment": "Hi {customer_name} {unknown}"}, "fallback")
    assert service.suggest("Payment", "Asha", "TCK-1") == "Hi Asha {unknown}"


def test_malformed_template_falls_back():
    service = ResponseService({"Payment": "Broken {customer_name"}, "Fallback {customer_name}")
    assert service.suggest("Payment", "Asha", "TCK-1") == "Fallback Asha"
