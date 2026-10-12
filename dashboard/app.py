"""TileLens AI — multi-page local Streamlit dashboard (Stage 5).

Run from your project root: streamlit run dashboard/app.py
Preserves existing inference and SQLite inspection-history components.
"""

import streamlit as st

from inference import load_models
from components.batch_inspection import render_batch_inspection
from components.history import render_history
from components.overview import render_overview
from components.single_inspection import render_single_inspection
from components.theme import apply_dashboard_styles, page_heading

st.set_page_config(
    page_title="TileLens AI | Inspection Console",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading TileLens AI models…")
def get_models():
    """Load weights only when an inspection actually runs."""
    return load_models()


apply_dashboard_styles()

PAGES = ["Overview", "Single Inspection", "Batch Inspection", "Inspection History"]
PAGE_ICONS = {
    "Overview": "◈",
    "Single Inspection": "▣",
    "Batch Inspection": "▦",
    "Inspection History": "◷",
}

with st.sidebar:
    st.markdown(
        '<div class="tl-brand"><span class="tl-brand-symbol">◈</span>'
        '<div><div class="tl-brand-name">TileLens <span>AI</span></div>'
        '<div class="tl-brand-tag">INSPECTION CONSOLE</div></div></div>',
        unsafe_allow_html=True,
    )
    st.caption("Ceramic tile surface intelligence")
    st.divider()

    if st.session_state.get("tl_nav") not in PAGES:
        st.session_state["tl_nav"] = "Overview"
    st.markdown('<p class="tl-sidebar-label">WORKSPACE</p>', unsafe_allow_html=True)
    selected_page = st.radio(
        "Workspace navigation",
        PAGES,
        key="tl_nav",
        format_func=lambda page: f"{PAGE_ICONS[page]}  {page}",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown('<p class="tl-sidebar-label">INSPECTION CONTROLS</p>', unsafe_allow_html=True)
    confidence = st.slider(
        "YOLO confidence threshold",
        min_value=0.10,
        max_value=0.85,
        step=0.05,
        value=0.25,
        key="tl_threshold",
        help="Minimum YOLO detection score to display. Not a calibrated probability.",
    )
    force_inspection = st.checkbox(
        "Allow single-image manual override",
        value=False,
        key="tl_single_override",
        help="Run YOLO even if MobileNetV3's tile score is below 0.80.",
    )
    st.caption("The threshold above applies to both single and batch inspections.")

    st.divider()
    st.markdown('<p class="tl-sidebar-label">ACTIVE MODELS</p>', unsafe_allow_html=True)
    st.markdown("**MobileNetV3-Small**  ·  Tile identification")
    st.markdown("**YOLO11n**  ·  Defect localization")
    st.caption("CPU inference · locally stored results")
    st.divider()
    st.warning("Research prototype. Confirm predictions by visual inspection.", icon="⚠️")

if selected_page == "Overview":
    page_heading("Overview", "Inspection activity, model context, and predicted defect trends.", "WORKSPACE / OVERVIEW")
    render_overview()
elif selected_page == "Single Inspection":
    page_heading("Single inspection", "Analyze one individual ceramic tile with both AI models.", "WORKSPACE / INSPECTION")
    render_single_inspection(get_models, confidence, force_inspection)
elif selected_page == "Batch Inspection":
    page_heading("Batch inspection", "Inspect up to ten tile photographs and compare results.", "WORKSPACE / BATCH")
    render_batch_inspection(get_models, confidence)
else:
    page_heading("Inspection history", "Review saved results, export PDFs, and manage records.", "WORKSPACE / HISTORY")
    render_history()

st.markdown('<div class="tl-footer">TileLens AI · Experimental ceramic tile inspection · Outputs require manual verification.</div>', unsafe_allow_html=True)
