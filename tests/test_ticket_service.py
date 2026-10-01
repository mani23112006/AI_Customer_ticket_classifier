"""Tests for orchestration logic with a fake repository and a mocked classifier."""
from datetime import date

import pytest

from config.errors import (
    ClassificationError,
    ConflictError,
    DatabaseError,
    InvalidTransitionError,
    ModelUnavailableError,
    ValidationError,
)
from config.settings import CATEGORIES, get_settings
from services.classifier_service import ClassifierService
from services.priority_service import PriorityEngine
from services.response_service import ResponseService
from services.ticket_repository import TicketFilters
from services.ticket_service import (
    MODEL_UNAVAILABLE_WARNING,
    TicketService,
    build_dashboard,
    format_ticket_no,
)


class FakeBackend:
    def __init__(self, top_index=3, top_prob=0.9):
        rest = (1 - top_prob) / 7
        self.probs = [rest] * 8
        self.probs[top_index] = top_prob

    def predict_proba(self, text):
        return self.probs


class FakeRepo:
    def __init__(self, fail_create=False, fail_followups=False):
        self.tickets = {}
        self.predictions = []
        self.feedback = []
        self.fail_create = fail_create
        self.fail_followups = fail_followups
        self.list_calls = []
        self.counter = 0

    def create_ticket(self, row):
        if self.fail_create:
            raise DatabaseError("db down")
        self.counter += 1
        stored = {**row, "id": f"id-{self.counter}", "ticket_no": self.counter,
                  "created_at": "2026-09-01T10:00:00+00:00", "suggested_reply": None}
        self.tickets[stored["id"]] = stored
        return dict(stored)

    def update_ticket(self, ticket_id, updates, expected_status=None):
        if self.fail_followups and set(updates) == {"suggested_reply"}:
            raise DatabaseError("db down")
        ticket = self.tickets[ticket_id]
        if expected_status is not None and ticket["status"] != expected_status:
            raise ConflictError("changed")
        ticket.update(updates)
        return dict(ticket)

    def create_prediction(self, ticket_id, predicted_category, confidence, top3, model_name):
        if self.fail_followups:
            raise DatabaseError("db down")
        self.predictions.append((ticket_id, predicted_category, confidence, top3, model_name))

    def add_feedback(self, ticket_id, field_name, old_value, new_value, model_confidence, note):
        self.feedback.append((ticket_id, field_name, old_value, new_value, model_confidence, note))

    def get_ticket(self, ticket_id):
        ticket = self.tickets.get(ticket_id)
        if ticket is None:
            return None
        pred = next((p for p in self.predictions if p[0] == ticket_id), None)
        row = dict(ticket)
        row["ticket_predictions"] = (
            {"predicted_category": pred[1], "confidence": pred[2], "top3": pred[3]} if pred else None
        )
        return row

    def list_tickets(self, filters, page, page_size, oldest_first=False):
        self.list_calls.append((page, page_size))
        rows = [self.get_ticket(t) for t in self.tickets]
        start = (page - 1) * page_size
        return rows[start:start + page_size], len(rows)


def make_service(repo=None, backend=None, provider=None):
    repo = repo or FakeRepo()
    classifier = ClassifierService(backend or FakeBackend(), list(CATEGORIES))
    return TicketService(
        repository=repo,
        classifier_provider=provider or (lambda: classifier),
        priority_engine=PriorityEngine.from_yaml(),
        response_service=ResponseService.from_yaml(),
        settings=get_settings(),
    ), repo


VALID = ("Asha Rao", "asha@example.com", "Refund status", "I returned my order two weeks ago and need my money back")


def test_format_ticket_no():
    assert format_ticket_no(42) == "TCK-000042"


def test_submit_happy_path_triaged():
    service, repo = make_service()
    result = service.submit_ticket(*VALID)
    assert result.category == "Refund"
    assert result.status == "triaged"
    assert result.priority == "Medium"
    assert result.ticket_no == "TCK-000001"
    assert "Asha Rao" in result.suggested_reply and "TCK-000001" in result.suggested_reply
    assert result.warnings == []
    assert len(repo.predictions) == 1
    assert repo.tickets["id-1"]["suggested_reply"] == result.suggested_reply


def test_low_confidence_goes_to_needs_review():
    service, _ = make_service(backend=FakeBackend(top_index=3, top_prob=0.4))
    result = service.submit_ticket(*VALID)
    assert result.status == "needs_review"
    assert result.category == "Refund"


def test_high_priority_rule_is_independent_of_model():
    service, _ = make_service()
    result = service.submit_ticket("Asha", "a@b.com", "Help", "Someone hacked my account and I cannot log in")
    assert result.priority == "High"
    assert result.priority_rule == "high:hacked"


def test_model_unavailable_still_saves_ticket_for_manual_review():
    def broken():
        raise ModelUnavailableError("no model")

    service, repo = make_service(provider=broken)
    result = service.submit_ticket(*VALID)
    assert result.category is None and result.confidence is None
    assert result.status == "needs_review"
    assert MODEL_UNAVAILABLE_WARNING in result.warnings
    assert repo.tickets["id-1"]["category"] is None
    assert repo.predictions == []
    assert "Asha Rao" in result.suggested_reply  # fallback template still works


def test_unexpected_model_error_does_not_crash():
    def exploding():
        raise RuntimeError("out of memory")

    service, _ = make_service(provider=exploding)
    assert service.submit_ticket(*VALID).status == "needs_review"


def test_database_failure_on_create_raises_and_is_clean():
    service, _ = make_service(repo=FakeRepo(fail_create=True))
    with pytest.raises(DatabaseError):
        service.submit_ticket(*VALID)


def test_followup_failures_become_warnings_not_errors():
    service, repo = make_service(repo=FakeRepo(fail_followups=True))
    result = service.submit_ticket(*VALID)
    assert result.ticket_no == "TCK-000001"
    assert len(result.warnings) == 2
    assert "id-1" in repo.tickets


def test_invalid_input_never_reaches_repository():
    service, repo = make_service()
    with pytest.raises(ValidationError):
        service.submit_ticket("", "bad", "", "x")
    assert repo.tickets == {}


def _submitted(service):
    return service.submit_ticket(*VALID).ticket_id


def test_review_override_records_feedback_and_status():
    service, repo = make_service(backend=FakeBackend(top_index=3, top_prob=0.4))
    ticket_id = _submitted(service)
    outcome = service.review_ticket(
        ticket_id, category="Payment", priority="High", status="in_progress",
        reply="Custom reply", note="model confused refund and billing",
    )
    stored = repo.tickets[ticket_id]
    assert stored["category"] == "Payment" and stored["priority"] == "High"
    assert stored["priority_rule"] == "manual_override"
    assert stored["status"] == "in_progress" and stored["suggested_reply"] == "Custom reply"
    fields = {f[1]: (f[2], f[3]) for f in repo.feedback}
    assert fields == {"category": ("Refund", "Payment"), "priority": ("Medium", "High")}
    assert repo.feedback[0][4] == 0.4  # model confidence captured
    assert outcome.warnings == []


def test_review_without_changes_saves_nothing():
    service, repo = make_service(backend=FakeBackend(top_index=3, top_prob=0.4))
    ticket_id = _submitted(service)
    current = repo.tickets[ticket_id]
    outcome = service.review_ticket(
        ticket_id, category=current["category"], priority=current["priority"],
        status=current["status"], reply=current["suggested_reply"],
    )
    assert outcome.warnings == ["No changes to save."]
    assert repo.feedback == []


def test_review_rejects_invalid_transition():
    service, _ = make_service(backend=FakeBackend(top_index=3, top_prob=0.4))
    ticket_id = _submitted(service)
    with pytest.raises(InvalidTransitionError):
        service.review_ticket(ticket_id, category="Refund", priority="Medium", status="closed", reply="")


def test_review_requires_category_when_unclassified():
    def broken():
        raise ModelUnavailableError("no model")

    service, _ = make_service(provider=broken)
    ticket_id = _submitted(service)
    with pytest.raises(ValidationError):
        service.review_ticket(ticket_id, category=None, priority="Low", status="needs_review", reply="")


def test_review_unknown_ticket():
    service, _ = make_service()
    with pytest.raises(DatabaseError):
        service.review_ticket("nope", category="Payment", priority="Low", status="new", reply="")


def test_allowed_next_statuses():
    service, _ = make_service()
    assert service.allowed_next_statuses("needs_review") == ["needs_review", "triaged", "in_progress"]
    assert service.allowed_next_statuses("closed") == ["closed"]


def test_list_tickets_clamps_out_of_range_page():
    service, repo = make_service()
    _submitted(service)
    page = service.list_tickets(TicketFilters(), page=9)
    assert page.page == 1 and len(page.items) == 1
    assert page.total == 1 and page.total_pages == 1
    assert page.items[0]["ticket_label"] == "TCK-000001"
    assert page.items[0]["confidence"] == 0.9
    assert len(repo.list_calls) == 2


def test_build_dashboard_numbers():
    rows = [
        {"created_at": "2026-09-01T10:00:00+00:00", "category": "Payment", "priority": "Medium",
         "status": "triaged", "ticket_predictions": {"confidence": 0.9}},
        {"created_at": "2026-09-01T12:00:00+00:00", "category": "Refund", "priority": "High",
         "status": "needs_review", "ticket_predictions": [{"confidence": 0.5}]},
        {"created_at": "2026-09-03T09:00:00+00:00", "category": None, "priority": "Low",
         "status": "needs_review", "ticket_predictions": None},
    ]
    data = build_dashboard(rows, threshold=0.6)
    assert data["total"] == 3
    assert data["avg_confidence"] == pytest.approx(0.7)
    assert data["low_confidence_rate"] == pytest.approx(0.5)
    cats = dict(zip(data["by_category"]["category"], data["by_category"]["count"]))
    assert cats == {"Payment": 1, "Refund": 1, "Unclassified": 1}
    assert list(data["by_priority"]["priority"]) == ["High", "Medium", "Low"]
    over = data["over_time"]
    assert list(over["count"]) == [2, 0, 1]  # missing day filled with zero
    assert over["date"].iloc[0] == date(2026, 9, 1)


def test_build_dashboard_empty():
    data = build_dashboard([], threshold=0.6)
    assert data["total"] == 0 and data["avg_confidence"] is None
    assert data["by_category"].empty
