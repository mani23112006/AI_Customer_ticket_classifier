"""Supabase data access. The only module that talks to the database.

All calls are wrapped so driver errors become DatabaseError with a generic
message; only the exception TYPE is logged (never row data or PII).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, TypeVar

from config.errors import ConfigError, ConflictError, DatabaseError
from config.settings import cache_resource, get_settings

logger = logging.getLogger(__name__)
T = TypeVar("T")

_TICKET_COLUMNS = (
    "id,ticket_no,customer_name,customer_email,subject,message,category,priority,"
    "priority_rule,status,suggested_reply,created_at,updated_at,"
    "ticket_predictions(predicted_category,confidence,top3,model_name)"
)
_SEARCH_STRIP = re.compile(r'[,()*%_\\"]')
_TICKET_NO = re.compile(r"^(?:tck-?)?0*(\d{1,12})$", re.IGNORECASE)
_DASHBOARD_BATCH = 1000  # Supabase's default max rows per request
_DASHBOARD_MAX_ROWS = 20_000


@dataclass
class TicketFilters:
    """Filters for the history / review queue. Empty list = no filter."""

    categories: list[str] = field(default_factory=list)
    priorities: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    date_from: date | None = None
    date_to: date | None = None
    search: str = ""


def build_search_clause(term: str) -> str | None:
    """Build a PostgREST ``or`` expression for free-text search, or None.

    Characters that would break the filter syntax (commas, parentheses,
    wildcards, quotes) are stripped, so user input cannot inject filters.
    """
    cleaned = re.sub(r"\s+", " ", _SEARCH_STRIP.sub(" ", term or "")).strip()[:100]
    if not cleaned:
        return None
    clauses = [
        f"subject.ilike.*{cleaned}*",
        f"message.ilike.*{cleaned}*",
        f"customer_name.ilike.*{cleaned}*",
        f"customer_email.ilike.*{cleaned}*",
    ]
    match = _TICKET_NO.match(cleaned)
    if match:
        clauses.append(f"ticket_no.eq.{int(match.group(1))}")
    return ",".join(clauses)


def _run(operation: str, func: Callable[[], T]) -> T:
    """Run a DB call, converting any driver/network error to DatabaseError."""
    try:
        return func()
    except DatabaseError:
        raise
    except Exception as exc:
        logger.error("database operation failed: op=%s error=%s", operation, type(exc).__name__)
        raise DatabaseError("The database is unavailable right now. Please try again.") from exc


class TicketRepository:
    """CRUD and query methods for tickets, predictions and feedback."""

    def __init__(self, client: Any) -> None:
        self._client = client

    # ------------------------------------------------------------- writes
    def create_ticket(self, row: dict[str, Any]) -> dict[str, Any]:
        """Insert a ticket and return the stored row (includes id, ticket_no)."""
        result = _run("create_ticket", lambda: self._client.table("tickets").insert(row).execute())
        if not result.data:
            raise DatabaseError("The ticket could not be saved.")
        return result.data[0]

    def create_prediction(
        self,
        ticket_id: str,
        predicted_category: str,
        confidence: float,
        top3: list[dict[str, Any]],
        model_name: str,
    ) -> None:
        """Store the model's original prediction for a ticket."""
        payload = {
            "ticket_id": ticket_id,
            "predicted_category": predicted_category,
            "confidence": confidence,
            "top3": top3,
            "model_name": model_name,
        }
        _run("create_prediction", lambda: self._client.table("ticket_predictions").insert(payload).execute())

    def update_ticket(
        self, ticket_id: str, updates: dict[str, Any], expected_status: str | None = None
    ) -> dict[str, Any]:
        """Update a ticket. If ``expected_status`` is given, only update when unchanged."""
        payload = {**updates, "updated_at": datetime.now(timezone.utc).isoformat()}

        def call() -> Any:
            query = self._client.table("tickets").update(payload).eq("id", ticket_id)
            if expected_status is not None:
                query = query.eq("status", expected_status)
            return query.execute()

        result = _run("update_ticket", call)
        if not result.data:
            if expected_status is not None:
                raise ConflictError("This ticket was changed by someone else. Refresh and retry.")
            raise DatabaseError("Ticket not found.")
        return result.data[0]

    def add_feedback(
        self,
        ticket_id: str,
        field_name: str,
        old_value: str | None,
        new_value: str,
        model_confidence: float | None,
        note: str | None,
    ) -> None:
        """Record an agent override for future retraining."""
        payload = {
            "ticket_id": ticket_id,
            "field": field_name,
            "old_value": old_value,
            "new_value": new_value,
            "model_confidence": model_confidence,
            "note": note or None,
        }
        _run("add_feedback", lambda: self._client.table("feedback_events").insert(payload).execute())

    # -------------------------------------------------------------- reads
    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        """Fetch one ticket (with embedded prediction) or None."""
        result = _run(
            "get_ticket",
            lambda: self._client.table("tickets")
            .select(_TICKET_COLUMNS)
            .eq("id", ticket_id)
            .limit(1)
            .execute(),
        )
        return result.data[0] if result.data else None

    def list_tickets(
        self,
        filters: TicketFilters,
        page: int,
        page_size: int,
        oldest_first: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return (rows, total_count) for one page using range queries."""
        offset = (max(page, 1) - 1) * page_size

        def call() -> Any:
            query = self._client.table("tickets").select(_TICKET_COLUMNS, count="exact")
            if filters.categories:
                query = query.in_("category", filters.categories)
            if filters.priorities:
                query = query.in_("priority", filters.priorities)
            if filters.statuses:
                query = query.in_("status", filters.statuses)
            if filters.date_from:
                query = query.gte("created_at", filters.date_from.isoformat())
            if filters.date_to:
                query = query.lt("created_at", (filters.date_to + timedelta(days=1)).isoformat())
            clause = build_search_clause(filters.search)
            if clause:
                query = query.or_(clause)
            query = query.order("created_at", desc=not oldest_first)
            return query.range(offset, offset + page_size - 1).execute()

        result = _run("list_tickets", call)
        total = result.count if result.count is not None else len(result.data or [])
        return list(result.data or []), int(total)

    def fetch_dashboard_rows(self, since: datetime | None) -> list[dict[str, Any]]:
        """Fetch lightweight rows for aggregation (batched, capped)."""
        rows: list[dict[str, Any]] = []
        start = 0
        while start < _DASHBOARD_MAX_ROWS:
            end = start + _DASHBOARD_BATCH - 1

            def call(a: int = start, b: int = end) -> Any:
                query = self._client.table("tickets").select(
                    "created_at,category,priority,status,ticket_predictions(confidence)"
                )
                if since is not None:
                    query = query.gte("created_at", since.isoformat())
                return query.order("created_at", desc=True).range(a, b).execute()

            batch = _run("fetch_dashboard_rows", call).data or []
            rows.extend(batch)
            if len(batch) < _DASHBOARD_BATCH:
                break
            start += _DASHBOARD_BATCH
        return rows


def create_supabase_client() -> Any:
    """Create a Supabase client from settings/secrets."""
    url, key = get_settings().require_supabase()
    try:
        from supabase import create_client
    except ImportError as exc:
        raise ConfigError("The 'supabase' package is not installed.") from exc
    try:
        return create_client(url, key)
    except Exception as exc:
        logger.error("supabase client creation failed: %s", type(exc).__name__)
        raise DatabaseError("Could not connect to the database.") from exc


@cache_resource
def get_repository() -> TicketRepository:
    """Process-wide repository singleton."""
    return TicketRepository(create_supabase_client())
