"""Local TileLens AI inspection history and PDF export.

Stores metadata in SQLite, with optional resized local image previews.
"""
from datetime import datetime, timezone
from io import BytesIO
from PIL import Image as PILImage, ImageOps
from pathlib import Path
import json
import sqlite3
from uuid import uuid4
from xml.sax.saxutils import escape

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "app" / "inspection_history.sqlite3"
IMAGE_DIR = ROOT / "data" / "app" / "inspection_images"
PREVIEW_MAX_SIZE = (1400, 1400)


def _image_path(inspection_id, kind):
    # UUID-generated ID and fixed kind avoid path traversal.
    if kind not in ("original", "annotated"):
        raise ValueError("Unexpected image kind")
    return IMAGE_DIR / f"{inspection_id}_{kind}.jpg"


def _write_preview(image, inspection_id, kind):
    """Store a modest-size, EXIF-corrected RGB preview, not full-resolution data."""
    if image is None:
        return None
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    preview = ImageOps.exif_transpose(image).convert("RGB")
    preview.thumbnail(PREVIEW_MAX_SIZE, PILImage.Resampling.LANCZOS)
    path = _image_path(inspection_id, kind)
    preview.save(path, format="JPEG", quality=86, optimize=True)
    return path




def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    # Keep manual-review records linked to their inspection on deletion.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id TEXT PRIMARY KEY,
            inspected_at TEXT NOT NULL,
            mode TEXT NOT NULL,
            filename TEXT NOT NULL,
            tile_prediction TEXT NOT NULL,
            tile_score REAL NOT NULL,
            detector_ran INTEGER NOT NULL,
            threshold REAL NOT NULL,
            status TEXT NOT NULL,
            predicted_regions INTEGER NOT NULL,
            edge_chipping INTEGER NOT NULL,
            hole INTEGER NOT NULL,
            line INTEGER NOT NULL,
            detections_json TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inspection_reviews (
            inspection_id TEXT PRIMARY KEY,
            reviewed_at TEXT NOT NULL,
            overall_review TEXT NOT NULL,
            tile_review TEXT NOT NULL,
            defect_review TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT '',
            FOREIGN KEY (inspection_id) REFERENCES inspections(id) ON DELETE CASCADE
        )
    """)
    return conn


def save_inspection(*, filename, mode, tile_prediction, tile_score,
                    detector_ran, threshold, status, detections,
                    original_image=None, annotated_image=None):
    """Save one actual model run; returns its local inspection ID."""
    counts = {"edge-chipping": 0, "hole": 0, "line": 0}
    for detection in detections:
        label = detection.get("Defect")
        if label in counts:
            counts[label] += 1

    inspection_id = uuid4().hex
    written = []
    try:
        for kind, image in (("original", original_image), ("annotated", annotated_image)):
            saved_path = _write_preview(image, inspection_id, kind)
            if saved_path is not None:
                written.append(saved_path)
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise

    try:
        with _connect() as conn:
            conn.execute("""
                INSERT INTO inspections (
                    id, inspected_at, mode, filename, tile_prediction,
                    tile_score, detector_ran, threshold, status,
                    predicted_regions, edge_chipping, hole, line, detections_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                inspection_id,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                mode,
                filename,
                tile_prediction,
                float(tile_score),
                int(detector_ran),
                float(threshold),
                status,
                len(detections),
                counts["edge-chipping"],
                counts["hole"],
                counts["line"],
                json.dumps(detections, ensure_ascii=False),
            ))
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return inspection_id


def _get_inspections():
    """Include any human review separately from immutable model predictions."""
    with _connect() as conn:
        rows = conn.execute("""
            SELECT i.*,
                   COALESCE(r.overall_review, 'unreviewed') AS review_outcome,
                   r.tile_review AS reviewed_tile,
                   r.defect_review AS reviewed_defects,
                   r.notes AS review_notes,
                   r.reviewed_at
            FROM inspections AS i
            LEFT JOIN inspection_reviews AS r ON r.inspection_id = i.id
            ORDER BY i.inspected_at DESC, i.rowid DESC
        """).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


OVERALL_OPTIONS = {
    "correct": "Appears correct",
    "partly_correct": "Partly correct",
    "incorrect": "Incorrect prediction",
    "needs_review": "Needs further review",
}
TILE_OPTIONS = {
    "correct": "Tile identification correct",
    "incorrect": "Tile identification incorrect",
    "uncertain": "Cannot determine",
}
DEFECT_OPTIONS = {
    "correct": "Detected regions appear correct",
    "false_positives": "False-positive regions",
    "missed_defects": "Missed visible defects",
    "both": "Both false positives and missed defects",
    "uncertain": "Cannot determine",
    "not_applicable": "Not applicable / detector skipped",
}


def save_review(inspection_id, overall_review, tile_review, defect_review, notes):
    """Create or replace a human judgement, without altering model predictions."""
    if overall_review not in OVERALL_OPTIONS:
        raise ValueError("Invalid overall assessment")
    if tile_review not in TILE_OPTIONS:
        raise ValueError("Invalid tile assessment")
    if defect_review not in DEFECT_OPTIONS:
        raise ValueError("Invalid defect assessment")
    if len(notes) > 2000:
        raise ValueError("Review notes must be at most 2000 characters")
    with _connect() as conn:
        if conn.execute("SELECT 1 FROM inspections WHERE id=?", (inspection_id,)).fetchone() is None:
            raise ValueError("Inspection not found")
        conn.execute("""
            INSERT INTO inspection_reviews (
                inspection_id, reviewed_at, overall_review, tile_review,
                defect_review, notes
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(inspection_id) DO UPDATE SET
                reviewed_at=excluded.reviewed_at,
                overall_review=excluded.overall_review,
                tile_review=excluded.tile_review,
                defect_review=excluded.defect_review,
                notes=excluded.notes
        """, (
            inspection_id,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            overall_review, tile_review, defect_review, notes.strip(),
        ))


def clear_review(inspection_id):
    """Return one record to unreviewed without deleting its model predictions."""
    with _connect() as conn:
        conn.execute("DELETE FROM inspection_reviews WHERE inspection_id=?", (inspection_id,))


def render_review_panel(inspection_id, selected):
    """Review one stored result; keep subjective judgements separate from AI outputs."""
    st.markdown("#### Manual validation")
    st.caption(
        "Review the actual photo and YOLO boxes before recording a judgement. "
        "Human assessments are not automatically verified ground truth."
    )
    overall = str(selected.get("review_outcome") or "unreviewed")
    tile = selected.get("reviewed_tile")
    defect = selected.get("reviewed_defects")
    note = selected.get("review_notes")
    if overall != "unreviewed":
        st.info("Saved review: " + OVERALL_OPTIONS.get(overall, overall))

    overall_choices = list(OVERALL_OPTIONS)
    tile_choices = list(TILE_OPTIONS)
    defect_choices = list(DEFECT_OPTIONS)
    if not bool(selected["detector_ran"]):
        st.caption("YOLO was skipped for this inspection: 'Not applicable' is appropriate.")
    with st.form(key=f"manual_review_{inspection_id}"):
        a, b = st.columns(2)
        with a:
            review_overall = st.selectbox(
                "Overall assessment", overall_choices,
                index=overall_choices.index(overall) if overall in overall_choices else 3,
                format_func=lambda x: OVERALL_OPTIONS[x],
            )
            review_tile = st.selectbox(
                "Tile identification", tile_choices,
                index=tile_choices.index(tile) if tile in tile_choices else 2,
                format_func=lambda x: TILE_OPTIONS[x],
            )
        with b:
            review_defect = st.selectbox(
                "Defect localization", defect_choices,
                index=defect_choices.index(defect) if defect in defect_choices else 5,
                format_func=lambda x: DEFECT_OPTIONS[x],
            )
        review_notes = st.text_area(
            "Reviewer notes (optional)",
            value=str(note) if isinstance(note, str) else "",
            max_chars=2000,
            placeholder="Example: decorative grout lines mistaken for defects",
        )
        submitted = st.form_submit_button("Save manual review", type="primary")
    if submitted:
        try:
            save_review(inspection_id, review_overall, review_tile, review_defect, review_notes)
        except Exception as exc:
            st.error(f"Couldn't save review: {exc}")
        else:
            st.success("Manual review saved.")
            st.rerun()

    if overall != "unreviewed":
        if st.button("Mark as unreviewed", key=f"unreview_{inspection_id}"):
            clear_review(inspection_id)
            st.rerun()


def _csv_bytes(frame):
    """Prevent spreadsheet formula execution in text fields on CSV open."""
    safe = frame.copy()
    for column in safe.select_dtypes(include=["object"]).columns:
        safe[column] = safe[column].map(
            lambda v: "'" + v if isinstance(v, str) and
            v.lstrip().startswith(("=", "+", "-", "@")) else v
        )
    return safe.to_csv(index=False).encode("utf-8-sig")


def _pdf_report(row):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, LongTable, TableStyle, Image as PDFImage, KeepTogether
    )

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    content = [
        Paragraph("TileLens AI - Inspection Report", styles["Title"]),
        Spacer(1, 8),
        Paragraph("Experimental computer-vision predictions - manual review required.", styles["BodyText"]),
        Spacer(1, 14),
    ]
    details = [
        ("Inspection ID", row["id"]),
        ("Date/time (UTC)", row["inspected_at"]),
        ("Image filename", row["filename"]),
        ("Inspection mode", row["mode"]),
        ("Tile prediction", row["tile_prediction"]),
        ("Tile model score", f"{float(row['tile_score']):.3f}"),
        ("YOLO detector executed", "Yes" if row["detector_ran"] else "No"),
        ("YOLO threshold", f"{float(row['threshold']):.2f}"),
        ("Status", row["status"]),
        ("Predicted regions", str(row["predicted_regions"])),
        ("Edge chipping", str(row["edge_chipping"])),
        ("Hole", str(row["hole"])),
        ("Line", str(row["line"])),
        ("Manual review", OVERALL_OPTIONS.get(str(row.get("review_outcome", "unreviewed")), "Unreviewed")),
        ("Reviewed at (UTC)", row.get("reviewed_at") or "Not reviewed"),
    ]
    details_table = LongTable([
        [Paragraph(f"<b>{escape(str(name))}</b>", styles["BodyText"]),
         Paragraph(escape(str(value)), styles["BodyText"])]
        for name, value in details
    ], colWidths=[54 * mm, 118 * mm], hAlign="LEFT")
    details_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    content.extend([details_table, Spacer(1, 14)])
    if row.get("review_outcome", "unreviewed") != "unreviewed":
        content.append(Paragraph("Manual reviewer assessment", styles["Heading3"]))
        content.append(Paragraph(
            "Tile classification: " + escape(TILE_OPTIONS.get(str(row.get("reviewed_tile")), "Not assessed")),
            styles["BodyText"],
        ))
        content.append(Paragraph(
            "Defect localization: " + escape(DEFECT_OPTIONS.get(str(row.get("reviewed_defects")), "Not assessed")),
            styles["BodyText"],
        ))
        if isinstance(row.get("review_notes"), str) and row.get("review_notes").strip():
            content.append(Paragraph("Reviewer note: " + escape(row["review_notes"]).replace("\n", "<br/>"), styles["BodyText"]))
        content.append(Spacer(1, 12))

    inspection_id = str(row["id"])
    found_images = 0
    for kind, title in (("original", "Uploaded image"), ("annotated", "Annotated YOLO result")):
        image_path = _image_path(inspection_id, kind)
        if not image_path.is_file():
            continue
        with PILImage.open(image_path) as img:
            w, h = img.size
        scale = min((165 * mm) / w, (83 * mm) / h)
        content.append(KeepTogether([
            Paragraph(title, styles["Heading3"]),
            PDFImage(str(image_path), width=w * scale, height=h * scale),
            Spacer(1, 10),
        ]))
        found_images += 1
    if found_images == 0:
        content.extend([
            Paragraph("Image previews", styles["Heading3"]),
            Paragraph("Not stored for this inspection. Older records contain metadata only.", styles["BodyText"]),
            Spacer(1, 10),
        ])
    elif found_images == 1 and not bool(row["detector_ran"]):
        content.extend([
            Paragraph("No annotated output: YOLO detection was skipped for this image.", styles["BodyText"]),
            Spacer(1, 10),
        ])

    predictions = json.loads(row["detections_json"])
    content.append(Paragraph("Predicted Defect Regions", styles["Heading2"]))
    if predictions:
        table_rows = [["#", "Class", "Score", "Bounding box (x1, y1, x2, y2)"]]
        for index, item in enumerate(predictions, 1):
            box = ", ".join(str(v) for v in item.get("Bounding box", []))
            table_rows.append([
                str(index),
                Paragraph(escape(str(item.get("Defect", "?"))), styles["BodyText"]),
                f"{float(item.get('Score', 0)):.3f}",
                Paragraph(escape(box), styles["BodyText"]),
            ])
        prediction_table = LongTable(
            table_rows, colWidths=[10 * mm, 40 * mm, 20 * mm, 102 * mm],
            repeatRows=1, hAlign="LEFT"
        )
        prediction_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        content.append(prediction_table)
    else:
        content.append(Paragraph("No regions were reported at the selected threshold, or the detector was skipped.", styles["BodyText"]))

    content.extend([
        Spacer(1, 16),
        Paragraph(
            "Important: A missing detection does not prove a tile is defect-free. "
            "A high model score does not guarantee a correct prediction. "
            "Any embedded images are resized local previews; manual verification is required.",
            styles["BodyText"],
        ),
    ])
    document.build(content)
    buffer.seek(0)
    return buffer.getvalue()


def delete_inspection(inspection_id):
    """Delete one saved result and its local image previews."""
    with _connect() as conn:
        result = conn.execute(
            "DELETE FROM inspections WHERE id = ?", (inspection_id,)
        )
        deleted = result.rowcount > 0
    if deleted:
        for kind in ("original", "annotated"):
            _image_path(inspection_id, kind).unlink(missing_ok=True)
    return deleted


def clear_inspection_history():
    """Delete all saved results and their associated image previews."""
    with _connect() as conn:
        ids = [row["id"] for row in conn.execute("SELECT id FROM inspections")]
        conn.execute("DELETE FROM inspections")
    for inspection_id in ids:
        for kind in ("original", "annotated"):
            _image_path(inspection_id, kind).unlink(missing_ok=True)
    return len(ids)


def render_history():
    st.subheader("Inspection History")
    st.caption("Metadata is stored in SQLite; new inspection image previews are saved locally.")

    frame = _get_inspections()
    if frame.empty:
        st.info("No saved inspections yet. Run single or batch inspection first.")
        return

    a, b, c, d = st.columns([1, 1.2, 1.2, 2])
    with a:
        mode = st.selectbox("Inspection type", ["All", "single", "batch"], key="history_mode")
    with b:
        statuses = st.multiselect("Status", sorted(frame["status"].unique()), key="history_status")
    with c:
        review_filter = st.selectbox(
            "Manual review", ["All", "unreviewed", *OVERALL_OPTIONS],
            format_func=lambda x: "All" if x == "All" else
                "Not reviewed" if x == "unreviewed" else OVERALL_OPTIONS[x],
            key="history_review_filter",
        )
    with d:
        text = st.text_input("Search filename", key="history_search")

    filtered = frame.copy()
    if mode != "All":
        filtered = filtered[filtered["mode"] == mode]
    if statuses:
        filtered = filtered[filtered["status"].isin(statuses)]
    if review_filter != "All":
        filtered = filtered[filtered["review_outcome"] == review_filter]
    if text:
        filtered = filtered[filtered["filename"].str.contains(text, case=False, regex=False, na=False)]

    if filtered.empty:
        st.info("No inspections match these filters.")
        return

    a, b, c, d = st.columns(4)
    a.metric("Records", len(filtered))
    b.metric("YOLO inspections", int(filtered["detector_ran"].sum()))
    c.metric("Predicted regions", int(filtered["predicted_regions"].sum()))
    d.metric("Manually reviewed", int((filtered["review_outcome"] != "unreviewed").sum()))

    visible = filtered[[
        "inspected_at", "mode", "filename", "tile_prediction",
        "tile_score", "status", "predicted_regions", "edge_chipping", "hole", "line",
        "review_outcome"
    ]].rename(columns={
        "inspected_at": "Time (UTC)",
        "mode": "Mode",
        "filename": "Filename",
        "tile_prediction": "Classifier",
        "tile_score": "Tile score",
        "status": "Result",
        "predicted_regions": "Regions",
        "review_outcome": "Manual review",
    })
    st.dataframe(visible, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Filtered History (CSV)",
        data=_csv_bytes(visible),
        file_name="tilelens_inspection_history.csv",
        mime="text/csv",
    )

    reviewed_rows = filtered.loc[filtered["review_outcome"] != "unreviewed"]
    if not reviewed_rows.empty:
        st.download_button(
            "Download Manual Validation (CSV)",
            data=_csv_bytes(reviewed_rows[[
                "id", "inspected_at", "filename", "mode", "tile_prediction",
                "tile_score", "detector_ran", "predicted_regions", "review_outcome",
                "reviewed_tile", "reviewed_defects", "review_notes", "reviewed_at",
            ]]),
            file_name="tilelens_manual_validation.csv",
            mime="text/csv",
            help="Save this file in your validation/review_reports directory.",
        )

    st.markdown("#### Inspection report")
    choices = filtered["id"].tolist()
    selected_id = st.selectbox(
        "Select an inspection",
        options=choices,
        format_func=lambda item_id: (
            f"{item_id[:8]} - " +
            str(filtered.loc[filtered["id"] == item_id, "filename"].iloc[0])
        ),
        key="history_inspection_id",
    )
    selected = filtered.loc[filtered["id"] == selected_id].iloc[0]
    st.write(
        f"**Result:** {selected['status']}  |  "
        f"**Predicted regions:** {selected['predicted_regions']}"
    )

    original_path = _image_path(selected_id, "original")
    annotated_path = _image_path(selected_id, "annotated")
    if original_path.is_file() or annotated_path.is_file():
        left, right = st.columns(2)
        with left:
            if original_path.is_file():
                st.image(str(original_path), caption="Uploaded image", use_container_width=True)
            else:
                st.info("Original preview was not saved for this inspection.")
        with right:
            if annotated_path.is_file():
                st.image(str(annotated_path), caption="YOLO annotated output", use_container_width=True)
            elif not bool(selected["detector_ran"]):
                st.info("YOLO detection was skipped, so no annotated image exists.")
            else:
                st.info("No annotated preview saved for this inspection.")
    else:
        st.info("This is an older metadata-only record. Run a new inspection to save image previews.")
    render_review_panel(selected_id, selected)

    st.download_button(
        "Download Selected Inspection (PDF)",
        data=_pdf_report(selected),
        file_name=f"tilelens_inspection_{selected_id[:8]}.pdf",
        mime="application/pdf",
    )

    st.divider()
    st.markdown("#### Delete inspection records")
    st.caption("Deletion is permanent and also removes saved previews for the deleted records.")

    confirm_one = st.checkbox(
        "I understand this selected inspection will be permanently deleted",
        key=f"history_confirm_one_{selected_id}",
    )
    if st.button(
        "Delete selected inspection",
        disabled=not confirm_one,
        key=f"history_delete_one_{selected_id}",
    ):
        delete_inspection(selected_id)
        st.rerun()

    with st.expander("Danger zone — clear all inspection history"):
        st.warning(
            "This permanently removes every saved inspection and its image previews. "
            "Download a CSV or PDFs first if you need copies."
        )
        confirmation = st.text_input(
            'Type DELETE ALL to confirm',
            key="history_delete_all_confirmation",
        )
        if st.button(
            "Clear all history",
            type="primary",
            disabled=confirmation != "DELETE ALL",
            key="history_clear_all_button",
        ):
            clear_inspection_history()
            st.rerun()
