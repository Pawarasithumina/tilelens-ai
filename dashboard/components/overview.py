"""Overview analytics sourced from the user's existing local SQLite records."""

import pandas as pd
import plotly.express as px
import streamlit as st

from components.history import _get_inspections

COLORS = {"Edge chipping": "#4ADBC6", "Hole": "#60A5FA", "Line": "#F4AE72"}


def _chart_style(fig, *, height=315):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=14, b=14),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#C6D7E7"),
        legend_title_text="",
    )
    return fig


def _navigate(page):
    st.session_state["tl_nav"] = page


def render_overview():
    st.markdown(
        '<div class="tl-hero"><strong>Visual inspection, in one workspace.</strong>'
        '<p>Review inspection activity and possible defect patterns from your saved runs. '
        'These charts summarize AI predictions, not verified ceramic defects.</p></div>',
        unsafe_allow_html=True,
    )

    frame = _get_inspections()
    total = len(frame)
    inspected = int(frame["detector_ran"].sum()) if total else 0
    regions = int(frame["predicted_regions"].sum()) if total else 0
    flagged = int((frame["predicted_regions"] > 0).sum()) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saved inspections", f"{total:,}")
    c2.metric("YOLO runs", f"{inspected:,}")
    c3.metric("Predicted regions", f"{regions:,}")
    c4.metric("Images with predictions", f"{flagged:,}")

    st.caption("Totals are calculated from persistent inspection history, including repeat analyses of an image.")

    if not total:
        st.markdown('<div class="tl-empty">No inspections saved yet. Begin with a single-image inspection or a small batch.</div>', unsafe_allow_html=True)
        st.button("Start first inspection", type="primary", on_click=_navigate, args=("Single Inspection",))
        return

    st.markdown("### Inspection analytics")
    left, right = st.columns([1, 1], gap="large")
    defect_counts = pd.DataFrame({
        "Class": ["Edge chipping", "Hole", "Line"],
        "Predicted regions": [
            int(frame["edge_chipping"].sum()),
            int(frame["hole"].sum()),
            int(frame["line"].sum()),
        ],
    })

    with left:
        st.markdown("**Predicted region distribution**")
        if regions:
            fig = px.bar(
                defect_counts,
                x="Class", y="Predicted regions", color="Class",
                color_discrete_map=COLORS,
            )
            fig.update_layout(showlegend=False)
            fig.update_yaxes(gridcolor="#26364B", zeroline=False)
            fig.update_xaxes(title_text="")
            st.plotly_chart(_chart_style(fig), use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No defect-region predictions have been recorded yet.")

    with right:
        st.markdown("**Inspection activity (UTC)**")
        activity = frame.copy()
        activity["date"] = pd.to_datetime(activity["inspected_at"], utc=True, errors="coerce").dt.date
        activity = activity.dropna(subset=["date"])
        if not activity.empty:
            daily = activity.groupby("date").size().reset_index(name="Inspections")
            daily["date"] = daily["date"].astype(str)
            fig = px.bar(daily, x="date", y="Inspections", color_discrete_sequence=["#4ADBC6"])
            fig.update_xaxes(title_text="Day (UTC)", type="category")
            fig.update_yaxes(gridcolor="#26364B", zeroline=False)
            st.plotly_chart(_chart_style(fig), use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No valid timestamps in inspection history.")

    st.markdown("### Recent inspections")
    recent = frame.head(8)[["inspected_at", "filename", "mode", "tile_prediction", "status", "predicted_regions"]]
    recent = recent.rename(columns={
        "inspected_at": "Time (UTC)", "filename": "Image", "mode": "Mode",
        "tile_prediction": "Tile prediction", "status": "Result", "predicted_regions": "Regions",
    })
    st.dataframe(recent, use_container_width=True, hide_index=True)
    st.button("Open full inspection history", on_click=_navigate, args=("Inspection History",))

    st.markdown("### Deployed research models")
    a, b = st.columns(2)
    with a:
        st.markdown("**MobileNetV3-Small · Tile identification**")
        st.caption("Trained on a curated tile / not-tile dataset. Strong internal results, but a false positive was seen on unfamiliar imagery.")
    with b:
        st.markdown("**YOLO11n · Surface defect localization**")
        st.caption("Internal benchmark mAP50: 57.78%; mAP50–95: 29.40%. Predictions on complex textures require careful review.")
    st.info("A model detecting zero regions does not certify that a tile is defect-free. Confidence scores are uncalibrated.")
