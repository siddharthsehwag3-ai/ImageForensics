
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).parent
EXPERIMENTS = ROOT / "experiments"
OUTPUT = EXPERIMENTS / "robustness_results.csv"

THRESHOLD = 10


def load_rgb(path):
    return np.asarray(
        Image.open(path).convert("RGB")
    )


def load_mask(path):
    return np.asarray(
        Image.open(path).convert("L")
    ) > 127


def jpeg_transform(array, quality):
    image = Image.fromarray(array.astype(np.uint8))
    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=quality
    )

    buffer.seek(0)

    return np.asarray(
        Image.open(buffer).convert("RGB")
    )


def resize_transform(array, scale):
    height, width = array.shape[:2]

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    smaller = cv2.resize(
        array,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    restored = cv2.resize(
        smaller,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    return restored


def calculate_metrics(truth, prediction):
    tp = int(np.logical_and(truth, prediction).sum())
    fp = int(np.logical_and(~truth, prediction).sum())
    fn = int(np.logical_and(truth, ~prediction).sum())
    tn = int(np.logical_and(~truth, ~prediction).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn else 0.0
    )

    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
        if tp + tn + fp + fn else 0.0
    )

    return {
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "f1_score": round(f1, 5),
        "iou": round(iou, 5),
        "pixel_accuracy": round(accuracy, 5)
    }


def predict(original, manipulated):
    difference = np.max(
        np.abs(
            original.astype(np.int16)
            - manipulated.astype(np.int16)
        ),
        axis=2
    )

    return difference > THRESHOLD


def evaluate_condition(original, manipulated, truth, condition):
    prediction = predict(original, manipulated)

    metrics = calculate_metrics(truth, prediction)

    return {
        "condition": condition,
        "threshold": THRESHOLD,
        **metrics
    }


def main():
    results = []

    folders = [
        folder for folder in EXPERIMENTS.iterdir()
        if folder.is_dir()
        and (folder / "original.png").exists()
        and (folder / "manipulated.png").exists()
        and (folder / "ground_truth_mask.png").exists()
    ]

    if not folders:
        print("No saved experiments found.")
        print("Generate experiments in the Streamlit application first.")
        return

    for folder in sorted(folders):

        original = load_rgb(folder / "original.png")
        manipulated = load_rgb(folder / "manipulated.png")
        truth = load_mask(folder / "ground_truth_mask.png")

        experiment_id = folder.name

        if (
            original.shape != manipulated.shape
            or original.shape[:2] != truth.shape
        ):
            print("Skipping incompatible experiment:", experiment_id)
            continue

        # Baseline: no additional transformation
        conditions = [
            (
                "Original PNG",
                original,
                manipulated
            )
        ]

        # JPEG compression at three quality levels
        for quality in [95, 75, 50]:
            conditions.append((
                f"JPEG quality {quality}",
                jpeg_transform(original, quality),
                jpeg_transform(manipulated, quality)
            ))

        # Resize down and restore to original dimensions
        for scale in [0.75, 0.50]:
            conditions.append((
                f"Resize and restore {int(scale * 100)}%",
                resize_transform(original, scale),
                resize_transform(manipulated, scale)
            ))

        for name, transformed_original, transformed_manipulated in conditions:

            metrics = evaluate_condition(
                transformed_original,
                transformed_manipulated,
                truth,
                name
            )

            results.append({
                "experiment_id": experiment_id,
                "condition": name,
                **metrics
            })

            print(
                experiment_id,
                "|",
                name,
                "| F1:",
                metrics["f1_score"],
                "| IoU:",
                metrics["iou"]
            )

    if not results:
        print("No valid experiment results were generated.")
        return

    dataframe = pd.DataFrame(results)
    dataframe.to_csv(OUTPUT, index=False)

    print("\n--------------------------------")
    print("ROBUSTNESS TEST COMPLETE")
    print("--------------------------------")
    print("Experiments evaluated:", len(folders))
    print("Evaluation rows:", len(dataframe))
    print("CSV saved to:", OUTPUT)

    print("\nAverage results by condition:")

    summary = dataframe.groupby("condition")[
        ["precision", "recall", "f1_score", "iou"]
    ].mean().round(4)

    print(summary.to_string())


if __name__ == "__main__":
    main()
