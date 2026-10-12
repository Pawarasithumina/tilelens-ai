"""Persistent, two-stage single tile inspection with original/annotated comparison."""

import hashlib
import io
from collections import Counter

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from components.history import save_inspection
from inference import classify_tile, detect_damage

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 25_000_000


def _read_upload(upload):
    data = upload.getvalue()
    if len(data) > MAX_BYTES:
        raise ValueError("Image is larger than 20 MB")
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    if image.width * image.height > MAX_PIXELS:
        raise ValueError("Image exceeds 25 megapixels")
    return data, image


def _png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _navigate(page):
    st.session_state["tl_nav"] = page


def render_single_inspection(get_models, confidence, force_inspection):
    st.markdown(
        '<div class="tl-hero"><strong>Inspect a ceramic tile</strong>'
        '<p>Upload one tile photo, review the identification score, then inspect possible defect regions '
        'with YOLO. The same confidence threshold is shared with batch inspection.</p></div>',
        unsafe_allow_html=True,
    )

    upload = st.file_uploader(
        "Choose a tile photograph",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        key="single_image_upload",
        help="JPG, PNG or WEBP, up to 20 MB",
    )
    if upload is None:
        st.info("Upload an image to preview it and start a new inspection.")
        return

    try:
        data, image = _read_upload(upload)
    except Exception as exc:
        st.error(f"Couldn't open image: {exc}")
        return

    # A changed image or changed setting does not silently show an old result.
    signature = (hashlib.sha256(data).hexdigest(), float(confidence), bool(force_inspection))

    img_col, info_col = st.columns([1.65, 1], gap="large")
    with img_col:
        st.image(image, caption="Original upload", use_container_width=True)
    with info_col:
        st.markdown("#### Inspection setup")
        st.write("**Filename:**", upload.name)
        st.write("**Resolution:**", f"{image.width:,} × {image.height:,}")
        st.write("**File size:**", f"{len(data)/1024/1024:.2f} MB")
        st.write("**YOLO threshold:**", f"{confidence:.2f}")
        st.caption("The classifier's 0.80 routing cutoff is provisional and not a validated acceptance standard.")
        st.caption("Manual override: " + ("enabled" if force_inspection else "disabled"))

    run = st.button("Run AI inspection", type="primary", use_container_width=True, key="run_single")
    if run:
        try:
            with st.spinner("Loading AI models and identifying tile…"):
                classifier, preprocess, class_names, detector, detector_lock = get_models()
                classification = classify_tile(classifier, preprocess, class_names, image)

            tile_score = float(classification["tile_score"])
            proceed = tile_score >= 0.80 or bool(force_inspection)
            detections = []
            annotated = None
            if proceed:
                with st.spinner("Detecting possible defect regions…"):
                    annotated, detections = detect_damage(
                        detector, detector_lock, image, confidence=confidence
                    )
                status = "Possible defects detected" if detections else "No detections above threshold"
                if tile_score < 0.80:
                    status = "Manual override: " + status
            else:
                status = "Skipped: tile classification review"

            inspection_id = save_inspection(
                filename=upload.name,
                mode="single",
                tile_prediction=classification["prediction"],
                tile_score=tile_score,
                detector_ran=proceed,
                threshold=confidence,
                status=status,
                detections=detections,
                original_image=image,
                annotated_image=annotated,
            )
            st.session_state["single_last_result"] = {
                "signature": signature,
                "classification": classification,
                "proceed": proceed,
                "detections": detections,
                "status": status,
                "annotated": annotated,
                "original": image.copy(),
                "inspection_id": inspection_id,
            }
        except Exception as exc:
            st.error(f"Inspection failed: {exc}")
            return

    result = st.session_state.get("single_last_result")
    if result is None:
        return
    if result["signature"] != signature:
        st.info("The image or inspection settings changed. Click Run AI inspection to refresh the predictions.")
        return

    st.divider()
    st.markdown("### Inspection results")
    classification = result["classification"]
    detections = result["detections"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Tile prediction", classification["prediction"].replace("_", " ").title())
    c2.metric("Tile class score", f"{classification['tile_score']:.2%}")
    c3.metric("Predicted regions", len(detections) if result["proceed"] else "—")
    st.caption("Classifier scores are not calibrated probabilities. Counts refer to model predictions, not confirmed damage.")

    if not result["proceed"]:
        st.warning("YOLO was skipped because tile identification requires review. The original image is shown above; no annotated result exists.")
    else:
        if detections:
            st.warning("Potential defect regions found. Verify the boxed areas visually.")
        else:
            st.info("No regions met the selected threshold. This does NOT confirm that the tile is defect-free.")

        left, right = st.columns(2, gap="large")
        with left:
            st.image(result["original"], caption="Original image", use_container_width=True)
        with right:
            st.image(result["annotated"], caption="YOLO annotated image", use_container_width=True)

        if detections:
            st.markdown("#### Detected regions")
            st.dataframe(pd.DataFrame(detections), use_container_width=True, hide_index=True)
            counts = Counter(d["Defect"] for d in detections)
            st.bar_chart(pd.DataFrame({
                "Class": ["edge-chipping", "hole", "line"],
                "Predicted regions": [counts["edge-chipping"], counts["hole"], counts["line"]],
            }).set_index("Class"))
        st.download_button(
            "Download annotated PNG",
            data=_png_bytes(result["annotated"]),
            file_name="tilelens_annotated.png",
            mime="image/png",
            use_container_width=True,
        )

    st.success(f"Saved to local history · Inspection ID: {result['inspection_id'][:8]}")
    st.button("View saved inspection report", key="single_to_history", on_click=_navigate, args=("Inspection History",))
