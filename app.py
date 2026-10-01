"""Home page of the AI-Powered Customer Support Ticket Classifier.

Run:  streamlit run app.py
Pages (Submit Ticket, Review Queue, Ticket History, Dashboard) appear in the sidebar.
"""
import streamlit as st

from config.errors import TicketAppError
from config.settings import configure_logging, get_settings
from services.ticket_service import system_status

st.set_page_config(
    page_title="Ticket operations",
    page_icon=":material/support_agent:",
    layout="wide",
    initial_sidebar_state="expanded",
)
configure_logging()

st.title("Ticket operations", icon=":material/support_agent:")
st.caption(
    "AI-assisted support triage, agent review, and service insights in one workspace."
)

try:
    settings = get_settings()
    status = system_status()
except TicketAppError as exc:
    st.error(f"Configuration problem: {exc}")
    st.stop()

with st.container(horizontal=True):
    st.metric(
        "Supabase",
        "Configured" if status["database_configured"] else "Not configured",
        border=True,
        help="Credentials are present; the connection is checked when data is requested.",
    )
    st.metric(
        "Classifier model",
        "Available" if status["model_present"] else "Not trained",
        border=True,
    )
    st.metric("Review threshold", f"{settings.confidence_threshold:.0%}", border=True)

if not status["database_configured"]:
    st.warning(
        "Set SUPABASE_URL and SUPABASE_KEY in `.streamlit/secrets.toml` or a `.env` file, "
        "and run `sql/schema.sql` in the Supabase SQL editor."
    )
if not status["model_present"]:
    st.info(
        "No trained model found. Tickets can still be submitted; they will go to manual review. "
        "To train: `python -m ml.make_sample_data`, then `python -m ml.train`."
    )

st.header("Workspace", icon=":material/space_dashboard:")
submit_col, review_col = st.columns(2)
with submit_col.container(border=True):
    st.subheader("Submit a ticket", icon=":material/add_comment:")
    st.caption("Create a ticket and review its category, priority, and suggested reply.")
    st.page_link("pages/1_Submit_Ticket.py", label="Open ticket form", icon=":material/arrow_forward:")
with review_col.container(border=True):
    st.subheader("Review queue", icon=":material/fact_check:")
    st.caption("Resolve low-confidence tickets and record agent decisions.")
    st.page_link("pages/2_Review_Queue.py", label="Review tickets", icon=":material/arrow_forward:")

history_col, dashboard_col = st.columns(2)
with history_col.container(border=True):
    st.subheader("Ticket history", icon=":material/manage_search:")
    st.caption("Search and filter previously submitted tickets.")
    st.page_link("pages/3_Ticket_History.py", label="Browse history", icon=":material/arrow_forward:")
with dashboard_col.container(border=True):
    st.subheader("Service dashboard", icon=":material/analytics:")
    st.caption("Track ticket volume, priority, status, and model confidence.")
    st.page_link("pages/4_Dashboard.py", label="View dashboard", icon=":material/arrow_forward:")
