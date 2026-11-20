from pathlib import Path
from datetime import datetime, timezone
from PIL import Image, ImageOps
import numpy as np
import csv
import json
import uuid

ROOT = Path(__file__).parent
BASE_DIR = ROOT / "dataset" / "base"
DONOR_DIR = ROOT / "dataset" / "donor"
EXP_DIR = ROOT / "experiments"
RECORD_FILE = EXP_DIR / "experiment_records.csv"

EXP_DIR.mkdir(exist_ok=True)

FIELDS = [
    "experiment_id", "method", "width", "height",
    "precision", "recall", "f1_score", "iou",
    "pixel_accuracy", "threshold", "timestamp_utc",
    "experiment_folder"
]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def images_in(folder):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def load_rgb(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def fit_donor(donor, width, height):
    # Center-crop donor to the base image's aspect ratio.
    dw, dh = donor.size
    target_ratio = width / height
    donor_ratio = dw / dh

    if donor_ratio > target_ratio:
        new_width = int(dh * target_ratio)
        left = (dw - new_width) // 2
        donor = donor.crop((left, 0, left + new_width, dh))
    else:
        new_height = int(dw / target_ratio)
        top = (dh - new_height) // 2
        donor = donor.crop((0, top, dw, top + new_height))

    return donor.resize((width, height), Image.Resampling.LANCZOS)


def new_folder(method):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    exp_id = f"{stamp}_{uuid.uuid4().hex[:6]}"
    folder = EXP_DIR / exp_id
    folder.mkdir(parents=True)
    return exp_id, folder


def save_experiment(method, original, manipulated, mask, details):
    exp_id, folder = new_folder(method)

    original.save(folder / "original.png")
    manipulated.save(folder / "manipulated.png")
    mask.save(folder / "ground_truth_mask.png")

    metadata = {
        "experiment_id": exp_id,
        "method": method,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "details": details,
        "ground_truth": "White pixels identify the inserted/duplicated region.",
        "prediction_note": "Evaluation uses the paired-image pixel-difference baseline."
    }

    (folder / "metadata.json").write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8"
    )

    record = {
        "experiment_id": exp_id,
        "method": method,
        "width": original.width,
        "height": original.height,
        "precision": "",
        "recall": "",
        "f1_score": "",
        "iou": "",
        "pixel_accuracy": "",
        "threshold": 10,
        "timestamp_utc": metadata["created_at_utc"],
        "experiment_folder": str(folder.relative_to(ROOT))
    }

    file_exists = RECORD_FILE.exists() and RECORD_FILE.stat().st_size > 0

    with open(RECORD_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

    print(f"Created {method}: {exp_id}")


def make_copy_move(image, index):
    arr = np.asarray(image).copy()
    width, height = image.size

    box = max(1, min(width, height) // 5)

    if box < 20:
        raise ValueError("Image is too small for a useful copy-move patch.")

    # Source is near the upper-left; destination near the lower-right.
    sx = int(width * 0.05)
    sy = int(height * 0.05)
    tx = width - box - int(width * 0.05)
    ty = height - box - int(height * 0.05)

    if tx < 0 or ty < 0:
        raise ValueError("Image dimensions are too small.")

    source = arr[sy:sy + box, sx:sx + box].copy()
    if source.shape[:2] != (box, box):
        raise ValueError("Could not extract the source patch.")

    arr[ty:ty + box, tx:tx + box] = source

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[ty:ty + box, tx:tx + box] = 255

    return Image.fromarray(arr), Image.fromarray(mask), {
        "source_x": sx, "source_y": sy,
        "destination_x": tx, "destination_y": ty,
        "patch_width": box, "patch_height": box,
        "source_image_index": index
    }


def make_splicing(base, donor, index):
    width, height = base.size
    donor = fit_donor(donor, width, height)

    patch_w = max(1, width // 4)
    patch_h = max(1, height // 4)

    # Donor crop near top-left, destination near bottom-right.
    cx = int(width * 0.05)
    cy = int(height * 0.05)
    tx = width - patch_w - int(width * 0.05)
    ty = height - patch_h - int(height * 0.05)

    base_arr = np.asarray(base).copy()
    donor_arr = np.asarray(donor)

    patch = donor_arr[cy:cy + patch_h, cx:cx + patch_w].copy()

    if patch.shape[:2] != (patch_h, patch_w):
        raise ValueError("Could not extract donor patch.")

    base_arr[ty:ty + patch_h, tx:tx + patch_w] = patch

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[ty:ty + patch_h, tx:tx + patch_w] = 255

    return Image.fromarray(base_arr), Image.fromarray(mask), {
        "donor_crop_x": cx, "donor_crop_y": cy,
        "destination_x": tx, "destination_y": ty,
        "patch_width": patch_w, "patch_height": patch_h,
        "donor_image_index": index
    }


def main():
    if not BASE_DIR.exists() or not DONOR_DIR.exists():
        raise SystemExit("Create dataset/base and dataset/donor folders first.")

    bases = images_in(BASE_DIR)
    donors = images_in(DONOR_DIR)

    print("Base images found:", len(bases))
    print("Donor images found:", len(donors))

    if len(bases) < 5 or len(donors) < 5:
        raise SystemExit(
            "Please place at least 5 images in each dataset folder."
        )

    bases = bases[:5]
    donors = donors[:5]

    created = 0

    for index, path in enumerate(bases, start=1):
        try:
            base = load_rgb(path)
            manipulated, mask, details = make_copy_move(base, index)
            details["source_file"] = path.name

            save_experiment(
                "Copy-Move", base, manipulated, mask, details
            )
            created += 1
        except Exception as error:
            print("Copy-move failed for", path.name, ":", error)

    for index, (base_path, donor_path) in enumerate(
        zip(bases, donors), start=1
    ):
        try:
            base = load_rgb(base_path)
            donor = load_rgb(donor_path)

            manipulated, mask, details = make_splicing(
                base, donor, index
            )

            details["base_file"] = base_path.name
            details["donor_file"] = donor_path.name

            save_experiment(
                "Image Splicing", base, manipulated, mask, details
            )
            created += 1
        except Exception as error:
            print("Splicing failed for", base_path.name, ":", error)

    print("\n--------------------------------")
    print("DATASET GENERATION FINISHED")
    print("--------------------------------")
    print("New experiments created:", created)
    print("Experiment records:", RECORD_FILE)

    if created != 10:
        print("WARNING: Expected 10. Review any errors printed above.")
    else:
        print("SUCCESS: 5 copy-move + 5 splicing experiments created.")


if __name__ == "__main__":
    main()