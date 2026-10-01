"""Text normalisation, model-input cleaning, and ticket input validation.

Two levels of cleaning are used on purpose:
* ``normalize_text``: light clean for what we STORE (keeps URLs and newlines
  so agents see what the customer wrote).
* ``clean_text``: heavier clean for what the MODEL sees (strips HTML and URLs).
The same ``build_model_text`` is used in training and inference.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass

from config.errors import ValidationError
from config.settings import (
    MAX_EMAIL_LEN,
    MAX_MESSAGE_LEN,
    MAX_NAME_LEN,
    MAX_SUBJECT_LEN,
    MIN_MESSAGE_LEN,
)

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t\f\v]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_ANY_WHITESPACE = re.compile(r"\s+")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass(frozen=True)
class TicketInput:
    """Validated, normalised ticket fields ready to be stored."""

    name: str
    email: str
    subject: str
    message: str


def normalize_text(text: str | None, multiline: bool = False) -> str:
    """Remove control characters and tidy whitespace, keeping the content intact."""
    if not text:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub(" ", text)
    if multiline:
        text = _SPACES.sub(" ", text)
        text = re.sub(r" ?\n ?", "\n", text)
        text = _MANY_NEWLINES.sub("\n\n", text)
    else:
        text = _ANY_WHITESPACE.sub(" ", text)
    return text.strip()


def clean_text(text: str | None) -> str:
    """Clean text for the model: unescape entities, drop HTML tags and URLs."""
    if not text:
        return ""
    text = html.unescape(str(text))
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _CONTROL.sub(" ", text)
    return _ANY_WHITESPACE.sub(" ", text).strip()


def build_model_text(subject: str | None, message: str | None) -> str:
    """Combine subject and message into the single string the model consumes."""
    subject_c = clean_text(subject)
    message_c = clean_text(message)
    if subject_c and message_c:
        return f"{subject_c}. {message_c}"
    return subject_c or message_c


def validate_ticket_input(
    name: str | None, email: str | None, subject: str | None, message: str | None
) -> TicketInput:
    """Validate and normalise the submission form. Raises ValidationError."""
    name_n = normalize_text(name)
    email_n = normalize_text(email).lower()
    subject_n = normalize_text(subject)
    message_n = normalize_text(message, multiline=True)

    errors: dict[str, str] = {}
    if not name_n:
        errors["name"] = "Name is required."
    elif len(name_n) > MAX_NAME_LEN:
        errors["name"] = f"Name must be at most {MAX_NAME_LEN} characters."

    if not email_n:
        errors["email"] = "Email is required."
    elif len(email_n) > MAX_EMAIL_LEN or not _EMAIL.match(email_n):
        errors["email"] = "Please enter a valid email address."

    if not subject_n:
        errors["subject"] = "Subject is required."
    elif len(subject_n) > MAX_SUBJECT_LEN:
        errors["subject"] = f"Subject must be at most {MAX_SUBJECT_LEN} characters."

    if not message_n:
        errors["message"] = "Message is required."
    elif len(message_n) < MIN_MESSAGE_LEN:
        errors["message"] = f"Message must be at least {MIN_MESSAGE_LEN} characters."
    elif len(message_n) > MAX_MESSAGE_LEN:
        errors["message"] = f"Message must be at most {MAX_MESSAGE_LEN} characters."

    if errors:
        raise ValidationError(errors)
    return TicketInput(name=name_n, email=email_n, subject=subject_n, message=message_n)
