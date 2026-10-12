from components.history import save_inspection
import hashlib
import io
from collections import Counter

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from inference import classify_tile, detect_damage


MAX_BATCH = 10


def render_batch_inspection(get_models, confidence):

    st.markdown(
        '<div class="tl-hero"><strong>Inspect tiles in batches</strong>'
        '<p>Upload up to ten individual tile photographs. Each inspection will be saved '
        'to local history with predicted defect regions by class.</p></div>',
        unsafe_allow_html=True,
    )

    files = st.file_uploader(
        "Select up to 10 tile images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="batch_images",
    )

    force_inspection = st.checkbox(
        "Inspect all images regardless of tile classification",
        value=False,
        key="batch_override",
    )

    files = files or []

    signature = (
        tuple(
            (
                file.name,
                hashlib.sha256(file.getvalue()).hexdigest(),
            )
            for file in files
        ),
        confidence,
        force_inspection,
    )

    if len(files) > MAX_BATCH:
        st.error(f"Please upload no more than {MAX_BATCH} images.")

    run = st.button(
        "Run Batch Inspection",
        type="primary",
        disabled=not files or len(files) > MAX_BATCH,
        use_container_width=True,
    )

    if run:

        rows = []
        previews = []

        try:
            with st.spinner("Loading AI models..."):
                (
                    classifier,
                    preprocess,
                    class_names,
                    detector,
                    detector_lock,
                ) = get_models()

            progress = st.progress(0)

            for index, file in enumerate(files):

                try:
                    if file.size > 20 * 1024 * 1024:
                        raise ValueError(
                            "Image exceeds the 20 MB limit"
                        )

                    image = ImageOps.exif_transpose(
                        Image.open(io.BytesIO(file.getvalue()))
                    ).convert("RGB")

                    if image.width * image.height > 25_000_000:
                        raise ValueError(
                            "Image exceeds the 25-megapixel limit"
                        )

                    classification = classify_tile(
                        classifier,
                        preprocess,
                        class_names,
                        image,
                    )

                    tile_score = classification["tile_score"]

                    detections = []
                    annotated = None

                    should_inspect = (
                        tile_score >= 0.80 or force_inspection
                    )

                    if should_inspect:

                        annotated, detections = detect_damage(
                            detector,
                            detector_lock,
                            image,
                            confidence=confidence,
                        )

                        if detections:
                            status = "Possible defects detected"
                        else:
                            status = "No detections above threshold"

                        if tile_score < 0.80:
                            status = "Manual override: " + status

                    else:
                        status = "Skipped: tile identification review"


                    # Save this image's inspection record
                    save_inspection(
                        filename=file.name,
                        mode="batch",
                        tile_prediction=classification["prediction"],
                        tile_score=tile_score,
                        detector_ran=should_inspect,
                        threshold=confidence,
                        status=status,
                        detections=detections,
                        original_image=image,
                        annotated_image=annotated,
                    )

                    counts = Counter(
                        d["Defect"] for d in detections
                    )

                    rows.append({
                        "Filename": file.name,
                        "Tile prediction": classification["prediction"],
                        "Tile score": round(tile_score, 4),
                        "Status": status,
                        "Predicted regions": len(detections),
                        "Edge chipping": counts["edge-chipping"],
                        "Hole": counts["hole"],
                        "Line": counts["line"],
                    })

                    # Always retain a thumbnail of the uploaded image.
                    original_preview = image.copy()
                    original_preview.thumbnail((900, 900))
                    annotated_preview = annotated.copy() if annotated is not None else None
                    if annotated_preview is not None:
                        annotated_preview.thumbnail((900, 900))
                    previews.append((file.name, original_preview, annotated_preview))

                except Exception as error:

                    rows.append({
                        "Filename": file.name,
                        "Tile prediction": "Error",
                        "Tile score": None,
                        "Status": f"Processing error: {error}",
                        "Predicted regions": 0,
                        "Edge chipping": 0,
                        "Hole": 0,
                        "Line": 0,
                    })

                progress.progress((index + 1) / len(files))

            progress.empty()

            st.session_state["batch_results"] = {
                "signature": signature,
                "rows": rows,
                "previews": previews,
                "threshold": confidence,
            }

        except Exception as error:
            st.error(f"Batch inspection failed: {error}")
            return

    saved = st.session_state.get("batch_results")

    if saved is None:
        return

    if saved["signature"] != signature:
        st.info(
            "Images or settings changed. "
            "Run Batch Inspection again to update the results."
        )
        return

    df = pd.DataFrame(saved["rows"])

    inspected = df["Status"].str.contains(
        "detected|detections",
        case=False,
        na=False,
    )

    total_regions = int(df["Predicted regions"].sum())

    st.divider()
    st.subheader("Batch Inspection Results")

    c1, c2, c3 = st.columns(3)

    c1.metric("Images uploaded", len(df))
    c2.metric("Images inspected", int(inspected.sum()))
    c3.metric("Predicted defect regions", total_regions)

    st.caption(
        "These are model-generated predictions. "
        "They are not verified defects."
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Predicted Defect Distribution")

    totals = pd.DataFrame({
        "Defect": ["Edge chipping", "Hole", "Line"],
        "Predicted regions": [
            int(df["Edge chipping"].sum()),
            int(df["Hole"].sum()),
            int(df["Line"].sum()),
        ],
    })

    if total_regions:
        st.bar_chart(
            totals.set_index("Defect"),
            y="Predicted regions",
        )
    else:
        st.info("No defect-region predictions in this batch at the selected threshold.")

    with st.expander("View uploaded and annotated images", expanded=True):

        if saved["previews"]:
            for record in saved["previews"]:
                if len(record) == 3:
                    filename, original_preview, annotated_preview = record
                    st.markdown(f"**{filename}**")
                    left, right = st.columns(2)
                    with left:
                        st.image(original_preview, caption="Uploaded image", use_container_width=True)
                    with right:
                        if annotated_preview is not None:
                            st.image(annotated_preview, caption="YOLO output", use_container_width=True)
                        else:
                            st.info("No YOLO output: inspection was skipped by tile identification.")
                else:
                    # Backward compatibility with already cached batch results.
                    filename, annotated_preview = record
                    st.image(annotated_preview, caption=filename, use_container_width=True)
        else:
            st.info("Run a batch inspection to see image previews.")

    st.download_button(
        label="Download Batch Results (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="tilelens_batch_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.warning(
        "A zero-detection result does not prove "
        "that a tile is defect-free. "
        "Review all predictions before making "
        "quality decisions."
    )
