"""Dashboard page. Aggregates are cached for 60 seconds (st.cache_data)."""
import plotly.express as px
import streamlit as st

from config.errors import TicketAppError
from config.settings import configure_logging, get_settings
from services.ticket_service import get_ticket_service

st.set_page_config(
    page_title="Service dashboard",
    page_icon=":material/analytics:",
    layout="wide",
)
configure_logging()
st.title("Service dashboard", icon=":material/analytics:")
st.caption("A clear view of ticket volume, service priorities, and model confidence.")


@st.cache_data(ttl=60, show_spinner="Loading dashboard...")
def load_dashboard(days: int | None) -> dict:
    """Cached wrapper; exceptions are not cached, so failures retry next run."""
    return get_ticket_service().get_dashboard_data(days)


ranges = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90, "All time": None}
col_range, col_refresh = st.columns([4, 1], vertical_alignment="bottom")
choice = col_range.segmented_control(
    "Time range",
    options=list(ranges),
    default="Last 30 days",
    key="dashboard_time_range",
)
if col_refresh.button("Refresh", icon=":material/refresh:"):
    load_dashboard.clear()

try:
    data = load_dashboard(ranges[choice])
except TicketAppError as exc:
    st.error(f"The dashboard is not available right now: {exc}")
    st.stop()

if data["total"] == 0:
    st.info("No tickets in this time range yet.")
    st.stop()

threshold = get_settings().confidence_threshold
with st.container(horizontal=True):
    st.metric("Total tickets", data["total"], border=True)
    st.metric(
        "Average confidence",
        "n/a" if data["avg_confidence"] is None else f"{data['avg_confidence']:.1%}",
        border=True,
    )
    st.metric(
        f"Low-confidence rate (< {threshold:.0%})",
        "n/a" if data["low_confidence_rate"] is None else f"{data['low_confidence_rate']:.1%}",
        border=True,
    )

category_col, priority_col = st.columns(2)
with category_col.container(border=True):
    st.subheader("Tickets by category", icon=":material/category:")
    st.plotly_chart(
        px.bar(
            data["by_category"],
            x="category",
            y="count",
            labels={"category": "Category", "count": "Tickets"},
            color_discrete_sequence=["#187E73"],
        ),
        key="chart_category",
    )
with priority_col.container(border=True):
    st.subheader("Tickets by priority", icon=":material/priority_high:")
    st.plotly_chart(
        px.bar(
            data["by_priority"],
            x="priority",
            y="count",
            color="priority",
            labels={"priority": "Priority", "count": "Tickets"},
            color_discrete_map={"High": "#A95858", "Medium": "#C87845", "Low": "#187E73"},
        ),
        key="chart_priority",
    )

time_col, status_col = st.columns(2)
with time_col.container(border=True):
    st.subheader("Ticket volume over time", icon=":material/trending_up:")
    st.plotly_chart(
        px.line(
            data["over_time"],
            x="date",
            y="count",
            markers=True,
            labels={"date": "Date", "count": "Tickets"},
            color_discrete_sequence=["#187E73"],
        ),
        key="chart_time",
    )
with status_col.container(border=True):
    st.subheader("Tickets by status", icon=":material/track_changes:")
    st.plotly_chart(
        px.bar(
            data["by_status"],
            x="status",
            y="count",
            labels={"status": "Status", "count": "Tickets"},
            color_discrete_sequence=["#46789A"],
        ),
        key="chart_status",
    )
st.caption("Confidence includes tickets with a stored model prediction. Dashboard data refreshes every 60 seconds.")
