"""Repository tests against a recording fake of the supabase-py query builder.

These check that queries are built correctly (filters, range, ordering, search
sanitising, error wrapping). They do NOT prove behaviour against a live database.
"""
from datetime import date
from types import SimpleNamespace

import pytest

from config.errors import ConflictError, DatabaseError
from services.ticket_repository import TicketFilters, TicketRepository, build_search_clause


class FakeQuery:
    def __init__(self, result=None, error=None):
        self.calls = []
        self._result = result if result is not None else SimpleNamespace(data=[], count=0)
        self._error = error

    def __getattr__(self, name):
        def method(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self
        return method

    def execute(self):
        if self._error:
            raise self._error
        return self._result

    def names(self):
        return [c[0] for c in self.calls]

    def find(self, name):
        return [c for c in self.calls if c[0] == name]


class FakeClient:
    def __init__(self, query):
        self.query = query
        self.tables = []

    def table(self, name):
        self.tables.append(name)
        return self.query


def test_list_tickets_range_and_order():
    query = FakeQuery(SimpleNamespace(data=[{"id": "1"}], count=41))
    repo = TicketRepository(FakeClient(query))
    rows, total = repo.list_tickets(TicketFilters(), page=3, page_size=20)
    assert total == 41 and rows == [{"id": "1"}]
    assert query.find("range")[0][1] == (40, 59)
    assert query.find("order")[0] == ("order", ("created_at",), {"desc": True})
    assert query.find("select")[0][2] == {"count": "exact"}


def test_list_tickets_oldest_first():
    query = FakeQuery()
    TicketRepository(FakeClient(query)).list_tickets(TicketFilters(), 1, 20, oldest_first=True)
    assert query.find("order")[0][2] == {"desc": False}


def test_list_tickets_applies_all_filters():
    query = FakeQuery()
    filters = TicketFilters(
        categories=["Payment"], priorities=["High"], statuses=["needs_review"],
        date_from=date(2026, 9, 1), date_to=date(2026, 9, 30), search="refund",
    )
    TicketRepository(FakeClient(query)).list_tickets(filters, 1, 20)
    in_calls = {c[1][0]: c[1][1] for c in query.find("in_")}
    assert in_calls == {"category": ["Payment"], "priority": ["High"], "status": ["needs_review"]}
    assert query.find("gte")[0][1] == ("created_at", "2026-09-01")
    assert query.find("lt")[0][1] == ("created_at", "2026-10-01")  # end date is inclusive
    assert "or_" in query.names()


def test_search_clause_strips_filter_syntax_characters():
    clause = build_search_clause('a,b) or (status.eq.closed "x" 100%_')
    assert clause is not None
    body = clause.split(",")
    # exactly the 4 intended clauses: the user's commas/parens could not add more
    assert len([part for part in body if part.startswith(("subject.", "message.", "customer_name.", "customer_email."))]) == 4
    assert "(" not in clause and ")" not in clause and '"' not in clause


def test_search_clause_empty_and_ticket_number():
    assert build_search_clause("   ") is None
    assert build_search_clause("") is None
    assert "ticket_no.eq.42" in build_search_clause("TCK-000042")
    assert "ticket_no.eq.7" in build_search_clause("7")
    assert "ticket_no" not in build_search_clause("refund")


def test_database_errors_are_wrapped_without_leaking_details():
    query = FakeQuery(error=RuntimeError("password=hunter2 host=secret"))
    repo = TicketRepository(FakeClient(query))
    with pytest.raises(DatabaseError) as info:
        repo.get_ticket("abc")
    assert "hunter2" not in str(info.value)


def test_update_with_expected_status_conflict():
    query = FakeQuery(SimpleNamespace(data=[], count=None))
    repo = TicketRepository(FakeClient(query))
    with pytest.raises(ConflictError):
        repo.update_ticket("abc", {"status": "triaged"}, expected_status="needs_review")
    assert ("status", "needs_review") in [c[1] for c in query.find("eq")]


def test_update_without_match_and_no_expected_status_is_plain_error():
    query = FakeQuery(SimpleNamespace(data=[], count=None))
    with pytest.raises(DatabaseError):
        TicketRepository(FakeClient(query)).update_ticket("abc", {"status": "closed"})


def test_dashboard_rows_are_fetched_in_batches():
    full = SimpleNamespace(data=[{"created_at": "x"}] * 1000, count=None)
    tail = SimpleNamespace(data=[{"created_at": "y"}] * 5, count=None)

    class Paged(FakeQuery):
        def __init__(self):
            super().__init__()
            self.responses = [full, tail]

        def execute(self):
            return self.responses.pop(0)

    query = Paged()
    rows = TicketRepository(FakeClient(query)).fetch_dashboard_rows(None)
    assert len(rows) == 1005
    assert [c[1] for c in query.find("range")] == [(0, 999), (1000, 1999)]
