"""Ticket History page: filters, search, pagination. UI only."""
from datetime import date

import pandas as pd
import streamlit as st

from config.errors import TicketAppError
from config.settings import CATEGORIES, PRIORITIES, STATUSES, configure_logging
from services.ticket_repository import TicketFilters
from services.ticket_service import get_ticket_service

st.set_page_config(
    page_title="Ticket history",
    page_icon=":material/manage_search:",
    layout="wide",
)
configure_logging()
st.title("Ticket history", icon=":material/manage_search:")
st.caption("Search the full support record and inspect prior triage decisions.")

try:
    service = get_ticket_service()
except TicketAppError as exc:
    st.error(f"The ticket system is not available right now: {exc}")
    st.stop()

with st.sidebar:
    st.header("Filters", icon=":material/filter_alt:")
    categories = st.multiselect("Category", CATEGORIES)
    priorities = st.multiselect("Priority", PRIORITIES)
    statuses = st.multiselect("Status", STATUSES)
    use_dates = st.checkbox("Filter by date")
    date_range: tuple[date, ...] = ()
    if use_dates:
        picked = st.date_input("Created between", value=(date.today(), date.today()))
        date_range = tuple(picked) if isinstance(picked, (tuple, list)) else (picked,)
    search = st.text_input("Search (subject, message, name, email, ticket no.)", max_chars=100)

if use_dates and len(date_range) < 2:
    st.info("Pick an end date to apply the date filter.")

filters = TicketFilters(
    categories=list(categories),
    priorities=list(priorities),
    statuses=list(statuses),
    date_from=date_range[0] if len(date_range) == 2 else None,
    date_to=date_range[1] if len(date_range) == 2 else None,
    search=search,
)

st.header("Ticket records", icon=":material/table_rows:")
page_number = st.number_input("Page", min_value=1, value=1, step=1)
try:
    page = service.list_tickets(filters, int(page_number))
except TicketAppError as exc:
    st.error(str(exc))
    st.stop()

st.caption(f"{page.total} tickets · Page {page.page} of {page.total_pages} · {page.page_size} per page")
if not page.items:
    st.info("No tickets match these filters.")
    st.stop()

st.dataframe(
    pd.DataFrame(
        [
            {
                "Ticket": t["ticket_label"],
                "Created": t["created_at"],
                "Subject": t["subject"],
                "Customer": t["customer_name"],
                "Category": t["category"] or "Unclassified",
                "Confidence": t["confidence"],
                "Priority": t["priority"],
                "Status": t["status"],
            }
            for t in page.items
        ]
    ),
    hide_index=True,
    column_config={
        "Confidence": st.column_config.NumberColumn(
            format="percent",
            help="Model confidence in the original category prediction.",
        )
    },
)

options = {f"{t['ticket_label']}: {t['subject'][:70]}": t for t in page.items}
with st.expander("View ticket details", icon=":material/description:"):
    chosen = options[st.selectbox("Ticket", list(options))]
    st.caption(f"{chosen['customer_name']} <{chosen['customer_email']}> | priority rule `{chosen['priority_rule']}`")
    st.text(chosen["message"])  # st.text avoids rendering user-supplied markdown/HTML
    if chosen.get("suggested_reply"):
        st.write("Reply on file")
        st.text(chosen["suggested_reply"])
