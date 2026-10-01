"""Business logic for tickets: submit, list, review, dashboard.

Pages call this module only. It coordinates validation, the classifier, the
priority and response engines, and the repository. Failures in optional steps
(model, follow-up writes) are downgraded to warnings so a customer's ticket is
never lost; only the first database write is mandatory.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pandas as pd

from config.errors import DatabaseError, InvalidTransitionError, ValidationError
from config.settings import (
    CATEGORIES,
    PRIORITIES,
    STATUS_TRANSITIONS,
    STATUSES,
    Settings,
    cache_resource,
    get_settings,
)
from preprocessing.text_cleaner import build_model_text, normalize_text, validate_ticket_input
from services.classifier_service import (
    ClassifierService,
    Prediction,
    decide_status,
    get_classifier_service,
)
from services.priority_service import PriorityEngine
from services.response_service import ResponseService
from services.ticket_repository import TicketFilters, TicketRepository, get_repository

logger = logging.getLogger(__name__)

MODEL_UNAVAILABLE_WARNING = (
    "Automatic classification is unavailable, so this ticket was sent to manual review."
)


def format_ticket_no(number: int) -> str:
    """Display form of the numeric ticket id, e.g. TCK-000042."""
    return f"TCK-{int(number):06d}"


@dataclass
class SubmissionResult:
    """Everything the UI needs to confirm a submission."""

    ticket_id: str
    ticket_no: str
    category: str | None
    confidence: float | None
    top_k: list[tuple[str, float]]
    priority: str
    priority_rule: str
    status: str
    suggested_reply: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class TicketPage:
    """One page of tickets plus paging info."""

    items: list[dict[str, Any]]
    total: int
    page: int
    page_size: int

    @property
    def total_pages(self) -> int:
        return max(1, math.ceil(self.total / self.page_size))


@dataclass
class ReviewResult:
    """Outcome of an agent review action."""

    ticket: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    """Lift the embedded prediction into flat ``confidence`` / ``top3`` keys."""
    flat = dict(row)
    pred = flat.pop("ticket_predictions", None)
    if isinstance(pred, list):
        pred = pred[0] if pred else None
    flat["confidence"] = float(pred["confidence"]) if pred and pred.get("confidence") is not None else None
    flat["top3"] = (pred or {}).get("top3") or []
    flat["predicted_category"] = (pred or {}).get("predicted_category")
    if flat.get("ticket_no") is not None:
        flat["ticket_label"] = format_ticket_no(flat["ticket_no"])
    return flat


def _confidence_of(row: dict[str, Any]) -> float | None:
    """Confidence from an embedded prediction (dict or list) or an already-flat row."""
    pred = row.get("ticket_predictions")
    if isinstance(pred, list):
        pred = pred[0] if pred else None
    if isinstance(pred, dict) and pred.get("confidence") is not None:
        return float(pred["confidence"])
    value = row.get("confidence")
    return float(value) if value is not None else None


def build_dashboard(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Aggregate flat rows into the numbers and tables the dashboard shows."""
    empty = pd.DataFrame
    if not rows:
        return {
            "total": 0,
            "avg_confidence": None,
            "low_confidence_rate": None,
            "by_category": empty(columns=["category", "count"]),
            "by_priority": empty(columns=["priority", "count"]),
            "by_status": empty(columns=["status", "count"]),
            "over_time": empty(columns=["date", "count"]),
        }

    df = pd.DataFrame(rows)
    df["confidence"] = pd.to_numeric(
        pd.Series([_confidence_of(r) for r in rows], index=df.index), errors="coerce"
    )
    df["category"] = df["category"].fillna("Unclassified")

    def counts(column: str, order: list[str] | None = None) -> pd.DataFrame:
        table = df[column].value_counts().rename_axis(column).reset_index(name="count")
        if order:
            rank = {v: i for i, v in enumerate(order)}
            table = table.sort_values(by=column, key=lambda s: s.map(lambda v: rank.get(v, len(rank))))
        return table.reset_index(drop=True)

    scored = df["confidence"].dropna()
    days = pd.to_datetime(df["created_at"], utc=True, errors="coerce").dt.date.dropna()
    per_day = days.value_counts().sort_index()
    if len(per_day):
        full_range = pd.date_range(per_day.index.min(), per_day.index.max(), freq="D").date
        per_day = per_day.reindex(full_range, fill_value=0)
    over_time = per_day.rename_axis("date").reset_index(name="count")

    return {
        "total": int(len(df)),
        "avg_confidence": float(scored.mean()) if len(scored) else None,
        "low_confidence_rate": float((scored < threshold).mean()) if len(scored) else None,
        "by_category": counts("category"),
        "by_priority": counts("priority", list(PRIORITIES)),
        "by_status": counts("status", list(STATUSES)),
        "over_time": over_time,
    }


class TicketService:
    """Coordinates all ticket workflows."""

    def __init__(
        self,
        repository: TicketRepository,
        classifier_provider: Callable[[], ClassifierService],
        priority_engine: PriorityEngine,
        response_service: ResponseService,
        settings: Settings | None = None,
    ) -> None:
        self._repo = repository
        self._classifier_provider = classifier_provider
        self._priority = priority_engine
        self._responses = response_service
        self._settings = settings or get_settings()

    # ------------------------------------------------------------- submit
    def submit_ticket(self, name: str, email: str, subject: str, message: str) -> SubmissionResult:
        """Validate, classify, prioritise and store a new ticket.

        Raises ValidationError for bad input and DatabaseError if the ticket
        itself could not be saved. Everything else degrades to warnings.
        """
        data = validate_ticket_input(name, email, subject, message)
        warnings: list[str] = []

        prediction = self._classify(build_model_text(data.subject, data.message), warnings)
        category = prediction.category if prediction else None
        status = (
            decide_status(prediction.confidence, self._settings.confidence_threshold)
            if prediction
            else "needs_review"
        )
        priority = self._priority.assign(data.subject, data.message, category)

        row = self._repo.create_ticket(
            {
                "customer_name": data.name,
                "customer_email": data.email,
                "subject": data.subject,
                "message": data.message,
                "category": category,
                "priority": priority.priority,
                "priority_rule": priority.rule_id,
                "status": status,
            }
        )
        ticket_no = format_ticket_no(row["ticket_no"])
        reply = self._responses.suggest(category, data.name, ticket_no)
        self._store_followups(row["id"], prediction, reply, warnings)

        return SubmissionResult(
            ticket_id=row["id"],
            ticket_no=ticket_no,
            category=category,
            confidence=prediction.confidence if prediction else None,
            top_k=prediction.top_k if prediction else [],
            priority=priority.priority,
            priority_rule=priority.rule_id,
            status=status,
            suggested_reply=reply,
            warnings=warnings,
        )

    def _classify(self, text: str, warnings: list[str]) -> Prediction | None:
        try:
            return self._classifier_provider().classify(text)
        except Exception as exc:  # any model problem must not lose the ticket
            logger.warning("classification unavailable: %s", type(exc).__name__)
            warnings.append(MODEL_UNAVAILABLE_WARNING)
            return None

    def _store_followups(
        self, ticket_id: str, prediction: Prediction | None, reply: str, warnings: list[str]
    ) -> None:
        try:
            self._repo.update_ticket(ticket_id, {"suggested_reply": reply})
        except DatabaseError:
            warnings.append("The suggested reply could not be saved with the ticket.")
        if prediction is not None:
            try:
                self._repo.create_prediction(
                    ticket_id,
                    prediction.category,
                    prediction.confidence,
                    prediction.top_k_as_dicts(),
                    self._settings.model_name,
                )
            except DatabaseError:
                warnings.append("The model prediction details could not be saved.")

    # -------------------------------------------------------------- lists
    def list_tickets(
        self, filters: TicketFilters, page: int = 1, oldest_first: bool = False
    ) -> TicketPage:
        """Return one page of tickets (page size from settings)."""
        size = self._settings.page_size
        page = max(1, int(page))
        rows, total = self._repo.list_tickets(filters, page, size, oldest_first)
        last_page = max(1, math.ceil(total / size))
        if not rows and total > 0 and page > last_page:
            page = last_page
            rows, total = self._repo.list_tickets(filters, page, size, oldest_first)
        return TicketPage([_flatten(r) for r in rows], total, page, size)

    # ------------------------------------------------------------- review
    def allowed_next_statuses(self, current: str) -> list[str]:
        """Current status followed by the statuses it may move to."""
        return [current, *STATUS_TRANSITIONS.get(current, ())]

    def review_ticket(
        self,
        ticket_id: str,
        *,
        category: str | None,
        priority: str,
        status: str,
        reply: str,
        note: str = "",
    ) -> ReviewResult:
        """Apply an agent's overrides, status change and reply edit."""
        current_raw = self._repo.get_ticket(ticket_id)
        if current_raw is None:
            raise DatabaseError("Ticket not found.")
        current = _flatten(current_raw)

        errors: dict[str, str] = {}
        if category not in CATEGORIES:
            errors["category"] = "Please choose a category."
        if priority not in PRIORITIES:
            errors["priority"] = "Please choose a valid priority."
        if status not in STATUSES:
            errors["status"] = "Please choose a valid status."
        note_clean = normalize_text(note)
        if len(note_clean) > 500:
            errors["note"] = "Note must be at most 500 characters."
        if errors:
            raise ValidationError(errors)

        old_status = current["status"]
        if status != old_status and status not in STATUS_TRANSITIONS.get(old_status, ()):
            raise InvalidTransitionError(f"A ticket cannot move from '{old_status}' to '{status}'.")

        updates: dict[str, Any] = {}
        feedback: list[tuple[str, str | None, str]] = []
        if category != current.get("category"):
            updates["category"] = category
            feedback.append(("category", current.get("category"), category))
        if priority != current["priority"]:
            updates["priority"] = priority
            updates["priority_rule"] = "manual_override"
            feedback.append(("priority", current["priority"], priority))
        if status != old_status:
            updates["status"] = status
        reply_clean = normalize_text(reply, multiline=True)
        if reply_clean != (current.get("suggested_reply") or ""):
            updates["suggested_reply"] = reply_clean

        if not updates:
            return ReviewResult(current, ["No changes to save."])

        updated = self._repo.update_ticket(ticket_id, updates, expected_status=old_status)
        warnings: list[str] = []
        for field_name, old, new in feedback:
            try:
                self._repo.add_feedback(
                    ticket_id, field_name, old, new, current.get("confidence"), note_clean
                )
            except DatabaseError:
                warnings.append(f"The {field_name} change was saved but its feedback record was not.")
        return ReviewResult(_flatten(updated), warnings)

    # ---------------------------------------------------------- dashboard
    def get_dashboard_data(self, days: int | None) -> dict[str, Any]:
        """Aggregate ticket statistics for the last ``days`` days (None = all)."""
        since = datetime.now(timezone.utc) - timedelta(days=days) if days else None
        rows = self._repo.fetch_dashboard_rows(since)
        return build_dashboard(rows, self._settings.confidence_threshold)


def system_status() -> dict[str, bool]:
    """Cheap readiness checks (no network calls) for the home page."""
    settings = get_settings()
    return {
        "database_configured": settings.supabase_configured,
        "model_present": (settings.model_dir / "config.json").exists(),
    }


@cache_resource
def get_ticket_service() -> TicketService:
    """Process-wide service wired with real dependencies."""
    settings = get_settings()
    return TicketService(
        repository=get_repository(),
        classifier_provider=get_classifier_service,
        priority_engine=PriorityEngine.from_yaml(settings.priority_rules_path),
        response_service=ResponseService.from_yaml(settings.response_templates_path),
        settings=settings,
    )
