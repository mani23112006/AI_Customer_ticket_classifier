"""Review Queue page for agents. UI only: logic lives in services.ticket_service."""
import pandas as pd
import streamlit as st

from config.errors import ConflictError, InvalidTransitionError, TicketAppError, ValidationError
from config.settings import CATEGORIES, PRIORITIES, STATUSES, configure_logging
from services.ticket_repository import TicketFilters
from services.ticket_service import get_ticket_service

st.set_page_config(
    page_title="Review queue",
    page_icon=":material/fact_check:",
    layout="wide",
)
configure_logging()
st.title("Review queue", icon=":material/fact_check:")
st.caption("Check model suggestions, make agent decisions, and keep ticket status moving.")

try:
    service = get_ticket_service()
except TicketAppError as exc:
    st.error(f"The ticket system is not available right now: {exc}")
    st.stop()

flash = st.session_state.pop("review_flash", None)
if flash:
    st.success(flash, icon=":material/check_circle:")

with st.container(border=True):
    st.subheader("Queue controls", icon=":material/filter_list:")
    col_status, col_page = st.columns([3, 1])
    status_filter = col_status.selectbox(
        "Ticket status", STATUSES, index=STATUSES.index("needs_review")
    )
    page_number = col_page.number_input("Page", min_value=1, value=1, step=1)

try:
    page = service.list_tickets(TicketFilters(statuses=[status_filter]), int(page_number), oldest_first=True)
except TicketAppError as exc:
    st.error(str(exc))
    st.stop()

st.caption(f"{page.total} tickets · Page {page.page} of {page.total_pages} · Oldest first")
if not page.items:
    st.info("No tickets with this status.")
    st.stop()

table = pd.DataFrame(
    [
        {
            "Ticket": t["ticket_label"],
            "Subject": t["subject"],
            "Category": t["category"] or "Unclassified",
            "Confidence": t["confidence"],
            "Priority": t["priority"],
            "Status": t["status"],
            "Created": t["created_at"],
        }
        for t in page.items
    ]
)
st.dataframe(
    table,
    hide_index=True,
    column_config={
        "Confidence": st.column_config.NumberColumn(
            format="percent",
            help="Model confidence in its original category prediction.",
        )
    },
)

options = {f"{t['ticket_label']}: {t['subject'][:70]}": t for t in page.items}
ticket = options[st.selectbox("Open ticket", list(options))]

st.header("Ticket details", icon=":material/receipt_long:")
with st.container(border=True):
    left, right = st.columns([3, 2])
    with left:
        st.subheader(ticket["subject"])
        st.caption(f"From {ticket['customer_name']} · {ticket['customer_email']}")
        with st.container(horizontal=True):
            status_color = "orange" if ticket["status"] == "needs_review" else "blue"
            st.badge(ticket["status"].replace("_", " ").title(), color=status_color)
            priority_color = {"High": "red", "Medium": "orange", "Low": "green"}[ticket["priority"]]
            st.badge(f"{ticket['priority']} priority", color=priority_color)
        st.text(ticket["message"])
    with right:
        st.subheader("Model output", icon=":material/psychology:")
        if ticket["top3"]:
            st.dataframe(
                pd.DataFrame(ticket["top3"]),
                hide_index=True,
                column_config={
                    "confidence": st.column_config.NumberColumn(format="percent"),
                    "Confidence": st.column_config.NumberColumn(format="percent"),
                },
            )
        else:
            st.info("No model prediction stored for this ticket.")
        st.caption(f"Priority rule: {ticket['priority_rule']}")

with st.form(f"review_form_{ticket['id']}"):
    st.subheader("Agent decision", icon=":material/edit_note:")
    has_category = ticket["category"] in CATEGORIES
    category_options = list(CATEGORIES) if has_category else ["(choose)", *CATEGORIES]
    category = st.selectbox(
        "Category",
        category_options,
        index=category_options.index(ticket["category"]) if has_category else 0,
    )
    priority = st.selectbox("Priority", PRIORITIES, index=PRIORITIES.index(ticket["priority"]))
    status_options = service.allowed_next_statuses(ticket["status"])
    new_status = st.selectbox("Status", status_options, index=0)
    reply = st.text_area("Reply", value=ticket.get("suggested_reply") or "", height=200)
    note = st.text_input("Note for the feedback record (optional)", max_chars=500)
    save = st.form_submit_button("Save changes", type="primary", icon=":material/save:")

if save:
    try:
        outcome = service.review_ticket(
            ticket["id"],
            category=None if category == "(choose)" else category,
            priority=priority,
            status=new_status,
            reply=reply,
            note=note,
        )
    except ValidationError as exc:
        for problem in exc.errors.values():
            st.error(problem)
    except InvalidTransitionError as exc:
        st.error(str(exc))
    except ConflictError as exc:
        st.error(str(exc))
    except TicketAppError as exc:
        st.error(f"Could not save changes: {exc}")
    else:
        if outcome.warnings == ["No changes to save."]:
            message_parts = outcome.warnings
        else:
            message_parts = [f"Saved changes to {ticket['ticket_label']}.", *outcome.warnings]
        st.session_state["review_flash"] = " ".join(message_parts)
        st.rerun()
