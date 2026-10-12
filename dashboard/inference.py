
from pathlib import Path
from threading import Lock

import torch
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image


# Project paths
ROOT = Path(__file__).resolve().parents[1]

CLASSIFIER_PATH = (
    ROOT
    / "models/tile_classifier/mobilenetv3_small_tile_v2.pt"
)

DETECTOR_PATH = (
    ROOT
    / "models/tile_damage_detector/yolo11n_tile_damage_v1_best.pt"
)


def load_models():
    """
    Load both trained deep learning models.
    Inference runs on the CPU.
    """

    if not CLASSIFIER_PATH.is_file():
        raise FileNotFoundError(CLASSIFIER_PATH)

    if not DETECTOR_PATH.is_file():
        raise FileNotFoundError(DETECTOR_PATH)

    # Model 1: MobileNetV3
    checkpoint = torch.load(
        CLASSIFIER_PATH,
        map_location="cpu",
        weights_only=True,
    )

    class_names = checkpoint["class_names"]

    if class_names != ["not_tile", "tile"]:
        raise ValueError(
            f"Unexpected classifier classes: {class_names}"
        )

    classifier = models.mobilenet_v3_small(
        weights=None
    )

    classifier.classifier[3] = nn.Linear(
        classifier.classifier[3].in_features,
        len(class_names),
    )

    classifier.load_state_dict(
        checkpoint["model_state_dict"]
    )

    classifier.eval()

    preprocess = transforms.Compose([
        transforms.Resize((
            checkpoint["img_size"],
            checkpoint["img_size"],
        )),
        transforms.ToTensor(),
        transforms.Normalize(
            checkpoint["mean"],
            checkpoint["std"],
        ),
    ])

    # Model 2: YOLO11n
    detector = YOLO(str(DETECTOR_PATH))

    # Protect shared YOLO inference in Streamlit
    detector_lock = Lock()

    return (
        classifier,
        preprocess,
        class_names,
        detector,
        detector_lock,
    )


def classify_tile(
    classifier,
    preprocess,
    class_names,
    image,
):
    """
    Return the MobileNetV3 prediction.
    """

    image = image.convert("RGB")

    tensor = preprocess(image).unsqueeze(0)

    with torch.inference_mode():
        logits = classifier(tensor)
        probabilities = torch.softmax(
            logits, dim=1
        )[0]

    tile_index = class_names.index("tile")
    tile_score = float(probabilities[tile_index])

    predicted_index = int(
        probabilities.argmax().item()
    )

    return {
        "prediction": class_names[predicted_index],
        "tile_score": tile_score,
        "not_tile_score": 1.0 - tile_score,
    }


def detect_damage(
    detector,
    detector_lock,
    image,
    confidence=0.25,
):
    """
    Perform YOLO damage detection and return
    an annotated PIL image and detections.
    """

    with detector_lock:
        results = detector.predict(
            source=image.convert("RGB"),
            imgsz=640,
            conf=confidence,
            device="cpu",
            verbose=False,
        )

        result = results[0]

        annotated = Image.fromarray(
            result.plot()[..., ::-1].copy()
        )

        detections = []

        for box in result.boxes:
            class_id = int(box.cls.item())

            detections.append({
                "Defect": detector.names[class_id],
                "Score": round(
                    float(box.conf.item()), 3
                ),
                "Bounding box": [
                    round(float(v), 1)
                    for v in box.xyxy[0].tolist()
                ],
            })

    return annotated, detections
