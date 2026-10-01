"""Submit Ticket page. UI only: all logic lives in services.ticket_service."""
import pandas as pd
import streamlit as st

from config.errors import TicketAppError, ValidationError
from config.settings import (
    MAX_EMAIL_LEN,
    MAX_MESSAGE_LEN,
    MAX_NAME_LEN,
    MAX_SUBJECT_LEN,
    configure_logging,
)
from services.ticket_service import SubmissionResult, get_ticket_service

st.set_page_config(
    page_title="Submit a ticket",
    page_icon=":material/add_comment:",
    layout="wide",
)
configure_logging()
st.title("Submit a ticket", icon=":material/add_comment:")
st.caption("Capture the customer issue and let the classifier prepare a first-pass triage.")

try:
    service = get_ticket_service()
except TicketAppError as exc:
    st.error(f"The ticket system is not available right now: {exc}")
    st.stop()


def show_result(result: SubmissionResult) -> None:
    """Render the confirmation for a stored ticket."""
    st.success(f"Ticket **{result.ticket_no}** has been submitted.")
    for warning in result.warnings:
        st.warning(warning)

    with st.container(horizontal=True):
        st.metric("Category", result.category or "Pending review", border=True)
        st.metric(
            "Confidence",
            f"{result.confidence:.0%}" if result.confidence is not None else "n/a",
            border=True,
        )
        st.metric("Priority", result.priority, border=True)
    st.caption(f"Status: {result.status.replace('_', ' ').title()} · Priority rule: {result.priority_rule}")
    if result.status == "needs_review":
        st.info("The model was not confident enough, so an agent will review this ticket.")

    if result.top_k:
        st.subheader("Model confidence", icon=":material/leaderboard:")
        st.dataframe(
            pd.DataFrame(result.top_k, columns=["Category", "Confidence"]),
            hide_index=True,
            column_config={
                "Confidence": st.column_config.NumberColumn(format="percent")
            },
        )
    with st.container(border=True):
        st.subheader("Suggested reply", icon=":material/quickreply:")
        st.text_area(
            "Edit before sending",
            value=result.suggested_reply,
            height=220,
            key=f"reply_{result.ticket_no}",
        )


result = st.session_state.get("last_result")
if result is not None:
    show_result(result)
    if st.button("Submit another ticket"):
        st.session_state.pop("last_result", None)
        st.rerun()
    st.stop()

with st.form("ticket_form"):
    st.subheader("Customer details", icon=":material/person:")
    name_col, email_col = st.columns(2)
    name = name_col.text_input("Full name", max_chars=MAX_NAME_LEN)
    email = email_col.text_input("Email", max_chars=MAX_EMAIL_LEN)
    subject = st.text_input("Subject", max_chars=MAX_SUBJECT_LEN)
    message = st.text_area(
        "Describe the issue",
        height=180,
        max_chars=MAX_MESSAGE_LEN,
        placeholder="Include what happened and any steps you have already tried.",
    )
    submitted = st.form_submit_button(
        "Submit ticket",
        type="primary",
        icon=":material/send:",
    )

if submitted:
    try:
        with st.spinner("Analysing your ticket..."):
            outcome = service.submit_ticket(name, email, subject, message)
    except ValidationError as exc:
        for problem in exc.errors.values():
            st.error(problem)
    except TicketAppError as exc:
        # Widgets keep their values on this run, so nothing the user typed is lost.
        st.error(f"We could not save your ticket: {exc} Your text is still in the form; please try again.")
    else:
        st.session_state["last_result"] = outcome
        st.rerun()
