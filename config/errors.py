"""Custom exceptions shared by all layers.

Messages on these exceptions are safe to show to users: they never contain
secrets or raw database/driver error text.
"""
from __future__ import annotations


class TicketAppError(Exception):
    """Base class for expected, user-presentable application errors."""


class ConfigError(TicketAppError):
    """Configuration is missing or invalid."""


class ValidationError(TicketAppError):
    """User input failed validation. ``errors`` maps field name -> message."""

    def __init__(self, errors: dict[str, str]) -> None:
        super().__init__("; ".join(errors.values()))
        self.errors = errors


class ModelUnavailableError(TicketAppError):
    """The trained model is missing or could not be loaded."""


class ClassificationError(TicketAppError):
    """The model loaded but produced unusable output."""


class DatabaseError(TicketAppError):
    """A database operation failed."""


class ConflictError(DatabaseError):
    """The record changed since it was read (optimistic concurrency check)."""


class InvalidTransitionError(TicketAppError):
    """Requested ticket status change is not allowed."""
