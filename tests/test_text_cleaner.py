"""Tests for text cleaning and input validation."""
import pytest

from config.errors import ValidationError
from preprocessing.text_cleaner import (
    build_model_text,
    clean_text,
    normalize_text,
    validate_ticket_input,
)


def test_clean_text_removes_html_urls_and_extra_space():
    raw = "  Hello <b>team</b>,\n\n check https://example.com/x?a=1   now &amp; help  "
    assert clean_text(raw) == "Hello team , check now & help"


def test_clean_text_handles_none_and_empty():
    assert clean_text(None) == ""
    assert clean_text("   ") == ""


def test_clean_text_removes_control_characters():
    assert clean_text("bad\x00text\x07here") == "bad text here"


def test_normalize_text_multiline_keeps_paragraphs():
    assert normalize_text("a  b\r\n\r\n\r\n\r\nc", multiline=True) == "a b\n\nc"


def test_normalize_text_single_line_collapses_newlines():
    assert normalize_text("a\n\n b\t c") == "a b c"


def test_build_model_text_joins_subject_and_message():
    assert build_model_text("Refund", "Where is my money?") == "Refund. Where is my money?"
    assert build_model_text("", "Only message here") == "Only message here"
    assert build_model_text(None, None) == ""


def test_validate_ok_normalises_fields():
    ticket = validate_ticket_input("  Asha  Rao ", " ASHA@Example.COM ", " Refund  help ", "I need my money back please")
    assert ticket.name == "Asha Rao"
    assert ticket.email == "asha@example.com"
    assert ticket.subject == "Refund help"


@pytest.mark.parametrize(
    "name,email,subject,message,bad_field",
    [
        ("", "a@b.com", "s", "long enough message", "name"),
        ("A", "not-an-email", "s", "long enough message", "email"),
        ("A", "a@b.com", "", "long enough message", "subject"),
        ("A", "a@b.com", "s", "short", "message"),
        ("A", "a@b.com", "s", "x" * 5001, "message"),
        ("A" * 101, "a@b.com", "s", "long enough message", "name"),
        ("A", "a@b.com", "s" * 201, "long enough message", "subject"),
    ],
)
def test_validate_rejects_bad_input(name, email, subject, message, bad_field):
    with pytest.raises(ValidationError) as info:
        validate_ticket_input(name, email, subject, message)
    assert bad_field in info.value.errors


def test_validate_collects_all_errors():
    with pytest.raises(ValidationError) as info:
        validate_ticket_input("", "", "", "")
    assert set(info.value.errors) == {"name", "email", "subject", "message"}
