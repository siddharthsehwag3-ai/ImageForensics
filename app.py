import zipfile
import io
import csv
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps


# =========================================================
# CONFIGURATION AND STORAGE
# =========================================================

st.set_page_config(
    page_title="ImageForensics | Research Lab",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

ROOT = Path(__file__).parent
EXPERIMENT_DIR = ROOT / "experiments"
RECORD_FILE = EXPERIMENT_DIR / "experiment_records.csv"
BATCH_FILE = EXPERIMENT_DIR / "batch_results.csv"

EXPERIMENT_DIR.mkdir(exist_ok=True)

RECORD_FIELDS = [
    "experiment_id", "method", "width", "height",
    "precision", "recall", "f1_score", "iou",
    "pixel_accuracy", "threshold", "timestamp_utc",
    "experiment_folder"
]


# =========================================================
# VISUAL DESIGN
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

.stApp {
    background:
      radial-gradient(ellipse at 5% 0%, rgba(75,145,117,.10), transparent 35%),
      radial-gradient(ellipse at 95% 10%, rgba(64,151,139,.08), transparent 30%),
      #101714;
    color: #edf4ef;
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #eaf4ec !important;
}

h1 { font-size: clamp(2rem, 5vw, 3.3rem) !important; }

p, label { color: #c4d1c8; }

[data-testid="stHeader"] { background: transparent; }

[data-testid="stMetric"] {
    background: #1a2821;
    border: 1px solid #344c3e;
    border-radius: 14px;
    padding: 15px;
}

[data-testid="stMetricValue"] { color: #8ed1b0; }

.stButton > button,
.stDownloadButton > button {
    background: #315b49;
    color: #f3fff6;
    border: 1px solid #527d67;
    border-radius: 11px;
    min-height: 45px;
    font-weight: 700;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #40765d;
    color: white;
    border-color: #8ed1b0;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 5px;
    background: #18231d;
    padding: 7px;
    border: 1px solid #304338;
    border-radius: 13px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    padding: 9px 13px;
}

.stTabs [aria-selected="true"] {
    background: #294b3b !important;
    color: #eaf5ee !important;
}

div[data-testid="stExpander"] {
    background: #19241e;
    border: 1px solid #304338;
    border-radius: 12px;
}

[data-testid="stImage"] img { border-radius: 12px; }

.hero {
    background: linear-gradient(120deg, #1b2c23, #1a2924);
    border: 1px solid #3a5948;
    border-radius: 20px;
    padding: clamp(20px, 4vw, 34px);
    margin: 12px 0 24px 0;
}

.kicker {
    color: #8ed1b0;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero-title {
    color: #f0f8f1;
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(28px, 5vw, 46px);
    font-weight: 700;
    line-height: 1.12;
    margin: 12px 0;
}

.hero-text {
    color: #b9cabe;
    max-width: 720px;
    line-height: 1.7;
}

@media(max-width:640px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .stButton > button,
    .stDownloadButton > button {
        width: 100%;
    }
}
</style>

<div class="hero">
    <div class="kicker">DIGITAL IMAGE FORENSICS · RESEARCH LAB</div>
    <div class="hero-title">See beyond the pixels.</div>
    <div class="hero-text">
        Analyse image integrity, generate controlled manipulations,
        evaluate localization methods and collect reproducible
        experimental results.
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# IMAGE FUNCTIONS
# =========================================================

def load_image(upload):
    image = Image.open(upload)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def save_png(image, path):
    image.save(path, format="PNG")


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def create_ela(image, quality=90, scale=10):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    compressed = Image.open(buffer).convert("RGB")

    a = np.asarray(image).astype(np.int16)
    b = np.asarray(compressed).astype(np.int16)

    difference = np.abs(a - b)
    result = np.clip(difference * scale, 0, 255).astype(np.uint8)

    return Image.fromarray(result)


def create_noise(image):
    array = np.asarray(image)
    bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    blurred = cv2.GaussianBlur(bgr, (5, 5), 0)
    residual = cv2.absdiff(bgr, blurred)

    residual = cv2.normalize(
        residual, None, 0, 255, cv2.NORM_MINMAX
    )

    return Image.fromarray(
        cv2.cvtColor(residual, cv2.COLOR_BGR2RGB)
    )


def create_edges(image):
    gray = cv2.cvtColor(
        np.asarray(image), cv2.COLOR_RGB2GRAY
    )
    return Image.fromarray(cv2.Canny(gray, 100, 200))


def create_dct(image):
    gray = cv2.cvtColor(
        np.asarray(image), cv2.COLOR_RGB2GRAY
    )
    gray = cv2.resize(gray, (256, 256))
    gray = np.float32(gray) / 255.0

    coefficients = cv2.dct(gray)
    magnitude = np.log1p(np.abs(coefficients))

    visual = cv2.normalize(
        magnitude, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    return Image.fromarray(visual)


# =========================================================
# CONTROLLED MANIPULATION GENERATORS
# =========================================================

def copy_move(image, sx, sy, tx, ty, box):
    array = np.asarray(image).copy()
    height, width = array.shape[:2]

    source = array[sy:sy+box, sx:sx+box].copy()

    array[ty:ty+box, tx:tx+box] = source

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[ty:ty+box, tx:tx+box] = 255

    return Image.fromarray(array), Image.fromarray(mask)


def splice_image(base, donor, cx, cy, tx, ty, patch_w, patch_h):
    base_array = np.asarray(base).copy()
    donor_array = np.asarray(donor)

    patch = donor_array[
        cy:cy+patch_h, cx:cx+patch_w
    ].copy()

    base_array[ty:ty+patch_h, tx:tx+patch_w] = patch

    mask = np.zeros(base_array.shape[:2], dtype=np.uint8)
    mask[ty:ty+patch_h, tx:tx+patch_w] = 255

    return Image.fromarray(base_array), Image.fromarray(mask)


# =========================================================
# QUANTITATIVE EVALUATION
# =========================================================

def evaluate_masks(ground_truth, prediction):
    truth = np.asarray(ground_truth).astype(bool)
    predicted = np.asarray(prediction).astype(bool)

    if truth.shape != predicted.shape:
        raise ValueError("Ground-truth and prediction sizes differ.")

    tp = int(np.logical_and(truth, predicted).sum())
    fp = int(np.logical_and(~truth, predicted).sum())
    fn = int(np.logical_and(truth, ~predicted).sum())
    tn = int(np.logical_and(~truth, ~predicted).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "f1_score": round(f1, 5),
        "iou": round(iou, 5),
        "pixel_accuracy": round(accuracy, 5)
    }


def difference_baseline(original, manipulated, threshold):
    a = np.asarray(original).astype(np.int16)
    b = np.asarray(manipulated).astype(np.int16)

    difference = np.max(np.abs(a - b), axis=2)

    prediction = (difference > threshold).astype(np.uint8) * 255

    difference_visual = np.clip(
        difference * 4, 0, 255
    ).astype(np.uint8)

    return (
        Image.fromarray(prediction),
        Image.fromarray(difference_visual)
    )


# =========================================================
# PERSISTENT EXPERIMENT STORAGE
# =========================================================

def read_records():
    if not RECORD_FILE.exists():
        return []

    with open(RECORD_FILE, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save_experiment(method, original, manipulated, mask, details, threshold=10):
    experiment_id = (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + "_" + uuid.uuid4().hex[:6]
    )

    folder = EXPERIMENT_DIR / experiment_id
    folder.mkdir(parents=True, exist_ok=True)

    prediction, difference = difference_baseline(
        original, manipulated, threshold
    )

    metrics = evaluate_masks(mask, prediction)

    save_png(original, folder / "original.png")
    save_png(manipulated, folder / "manipulated.png")
    save_png(mask, folder / "ground_truth_mask.png")
    save_png(prediction, folder / "predicted_mask.png")
    save_png(difference, folder / "difference_map.png")

    metadata = {
        "experiment_id": experiment_id,
        "method": method,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "details": details,
        "threshold": threshold,
        "evaluation": metrics,
        "evaluation_note": (
            "Paired-image pixel-difference baseline. "
            "Original and manipulated images are both available."
        )
    }

    with open(folder / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    record = {
        "experiment_id": experiment_id,
        "method": method,
        "width": original.width,
        "height": original.height,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "iou": metrics["iou"],
        "pixel_accuracy": metrics["pixel_accuracy"],
        "threshold": threshold,
        "timestamp_utc": metadata["created_at_utc"],
        "experiment_folder": str(folder.relative_to(ROOT))
    }

    exists = RECORD_FILE.exists()

    with open(RECORD_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RECORD_FIELDS)

        if not exists or RECORD_FILE.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(record)

    return experiment_id, folder, metrics, prediction, difference


def run_batch(threshold):
    records = read_records()
    results = []

    for record in records:
        folder = ROOT / record["experiment_folder"]

        try:
            original = Image.open(folder / "original.png").convert("RGB")
            manipulated = Image.open(folder / "manipulated.png").convert("RGB")
            mask = Image.open(folder / "ground_truth_mask.png").convert("L")

            prediction, _ = difference_baseline(
                original, manipulated, threshold
            )

            metrics = evaluate_masks(mask, prediction)

            results.append({
                "experiment_id": record["experiment_id"],
                "method": record["method"],
                "threshold": threshold,
                **metrics
            })

        except (OSError, ValueError) as error:
            st.warning(
                f"Could not evaluate {record['experiment_id']}: {error}"
            )

    if results:
        pd.DataFrame(results).to_csv(BATCH_FILE, index=False)

    return results


# =========================================================
# SESSION STATE
# =========================================================

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


# =========================================================
# APPLICATION NAVIGATION
# =========================================================

tabs = st.tabs([
    "Overview",
    "Image Analysis",
    "Copy-Move Lab",
    "Splicing Lab",
    "Evaluation",
    "Experiment Records"
])


# =========================================================
# TAB 1: OVERVIEW
# =========================================================

with tabs[0]:

    st.header("Research Workspace")

    records = read_records()

    total = len(records)
    copy_count = sum(r["method"] == "Copy-Move" for r in records)
    splice_count = sum(r["method"] == "Image Splicing" for r in records)

    c1, c2, c3 = st.columns(3)

    c1.metric("Saved experiments", total)
    c2.metric("Copy-move tests", copy_count)
    c3.metric("Splicing tests", splice_count)

    st.subheader("Research modules")

    a, b = st.columns(2)

    with a:
        st.markdown("### 🔍 Forensic analysis")
        st.write(
            "Explore ELA, noise residuals, edge maps and frequency features."
        )

        st.markdown("### 🧩 Copy-move experiments")
        st.write(
            "Create controlled duplicated regions and ground-truth masks."
        )

    with b:
        st.markdown("### 🖼️ Image splicing")
        st.write(
            "Insert a patch from another image and record its exact location."
        )

        st.markdown("### 📊 Quantitative evaluation")
        st.write(
            "Compare pixel-difference predictions against ground-truth masks."
        )

    st.info(
        "The pixel-difference method is a paired-image baseline. "
        "It requires the original image and is not a general-purpose "
        "detector for unknown photographs."
    )


# =========================================================
# TAB 2: IMAGE ANALYSIS
# =========================================================

with tabs[1]:

    st.header("Image Forensic Analysis")

    upload = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="analysis_upload"
    )

    if upload:

        image = load_image(upload)
        image.thumbnail((1600, 1600))

        with st.spinner("Generating forensic views..."):
            ela = create_ela(image)
            noise = create_noise(image)
            edges = create_edges(image)
            dct = create_dct(image)

        c1, c2 = st.columns(2)

        with c1:
            st.image(image, caption="Original image", use_container_width=True)

        with c2:
            st.image(ela, caption="Error Level Analysis", use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            st.image(noise, caption="Noise residual", use_container_width=True)

        with c4:
            st.image(edges, caption="Canny edge map", use_container_width=True)

        st.image(dct, caption="DCT frequency visualization", use_container_width=True)

        analysis_zip = io.BytesIO()

        with zipfile.ZipFile(analysis_zip, "w") as archive:
            for name, picture in {
                "original.png": image,
                "ela.png": ela,
                "noise.png": noise,
                "edges.png": edges,
                "dct.png": dct
            }.items():
                archive.writestr(name, png_bytes(picture))

        st.download_button(
            "Download forensic visualizations",
            analysis_zip.getvalue(),
            "forensic_analysis.zip",
            "application/zip",
            use_container_width=True
        )


# =========================================================
# TAB 3: COPY-MOVE LAB
# =========================================================

with tabs[2]:

    st.header("Copy-Move Forgery Generator")

    st.write(
        "Select a square source region and duplicate it at another location."
    )

    upload = st.file_uploader(
        "Upload source photograph",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="copy_upload"
    )

    if upload:

        image = load_image(upload)
        image.thumbnail((1000, 1000))

        width, height = image.size

        box = max(1, min(width // 5, height // 5, 120))

        st.image(image, caption="Source photograph", use_container_width=True)

        st.caption(
            f"Image size: {width} × {height} pixels. "
            f"Duplicate region: {box} × {box} pixels."
        )

        col1, col2 = st.columns(2)

        with col1:
            sx = st.slider("Source X", 0, width-box, 0, key="sx")
            sy = st.slider("Source Y", 0, height-box, 0, key="sy")

        with col2:
            tx = st.slider("Destination X", 0, width-box, width-box, key="tx")
            ty = st.slider("Destination Y", 0, height-box, height-box, key="ty")

        if st.button("Generate copy-move manipulation", use_container_width=True):

            overlap = (
                sx < tx + box and tx < sx + box
                and sy < ty + box and ty < sy + box
            )

            if overlap:
                st.warning(
                    "Source and destination overlap. Move the destination "
                    "so that it does not overlap the source."
                )
            else:
                manipulated, mask = copy_move(
                    image, sx, sy, tx, ty, box
                )

                details = {
                    "source_x": sx,
                    "source_y": sy,
                    "destination_x": tx,
                    "destination_y": ty,
                    "region_width": box,
                    "region_height": box
                }

                result = save_experiment(
                    "Copy-Move", image, manipulated, mask, details
                )

                st.session_state.latest_result = {
                    "method": "Copy-Move",
                    "original": image,
                    "manipulated": manipulated,
                    "mask": mask,
                    "prediction": result[3],
                    "difference": result[4],
                    "metrics": result[2],
                    "folder": result[1]
                }

                st.success(
                    f"Experiment saved: {result[0]}"
                )

        result = st.session_state.latest_result

        if result and result["method"] == "Copy-Move":

            c1, c2 = st.columns(2)

            with c1:
                st.image(result["original"], caption="Original")

            with c2:
                st.image(result["manipulated"], caption="Manipulated")

            st.image(result["mask"], caption="Ground-truth mask: changed destination")

            st.subheader("Baseline evaluation")
            st.json(result["metrics"])

            st.image(
                result["prediction"],
                caption="Pixel-difference predicted mask"
            )

            st.image(
                result["difference"],
                caption="Pixel difference visualization"
            )


# =========================================================
# TAB 4: SPLICING LAB
# =========================================================

with tabs[3]:

    st.header("Image Splicing Generator")

    st.write(
        "Insert a rectangular patch from a donor photograph into a base image."
    )

    base_upload = st.file_uploader(
        "Upload base image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="base_upload"
    )

    donor_upload = st.file_uploader(
        "Upload donor image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="donor_upload"
    )

    if base_upload and donor_upload:

        base = load_image(base_upload)
        donor = load_image(donor_upload)

        base.thumbnail((1000, 1000))
        donor.thumbnail((1000, 1000))

        bw, bh = base.size
        dw, dh = donor.size

        patch_w = max(1, min(bw // 4, dw // 3, 150))
        patch_h = max(1, min(bh // 4, dh // 3, 150))

        c1, c2 = st.columns(2)

        with c1:
            st.image(base, caption="Base image")

        with c2:
            st.image(donor, caption="Donor image")

        c3, c4 = st.columns(2)

        with c3:
            cx = st.slider("Donor crop X", 0, dw-patch_w, 0, key="cx")
            cy = st.slider("Donor crop Y", 0, dh-patch_h, 0, key="cy")

        with c4:
            tx = st.slider("Paste location X", 0, bw-patch_w, bw-patch_w, key="spx")
            ty = st.slider("Paste location Y", 0, bh-patch_h, bh-patch_h, key="spy")

        if st.button("Generate splicing manipulation", use_container_width=True):

            manipulated, mask = splice_image(
                base, donor, cx, cy, tx, ty, patch_w, patch_h
            )

            details = {
                "donor_crop_x": cx,
                "donor_crop_y": cy,
                "destination_x": tx,
                "destination_y": ty,
                "patch_width": patch_w,
                "patch_height": patch_h
            }

            result = save_experiment(
                "Image Splicing", base, manipulated, mask, details
            )

            st.session_state.latest_result = {
                "method": "Image Splicing",
                "original": base,
                "manipulated": manipulated,
                "mask": mask,
                "prediction": result[3],
                "difference": result[4],
                "metrics": result[2],
                "folder": result[1]
            }

            st.success(f"Experiment saved: {result[0]}")

        result = st.session_state.latest_result

        if result and result["method"] == "Image Splicing":

            c1, c2 = st.columns(2)

            with c1:
                st.image(result["original"], caption="Original base")

            with c2:
                st.image(result["manipulated"], caption="Spliced image")

            st.image(result["mask"], caption="Ground-truth mask: inserted region")

            st.subheader("Baseline evaluation")
            st.json(result["metrics"])

            st.image(
                result["prediction"],
                caption="Pixel-difference predicted mask"
            )

            st.image(
                result["difference"],
                caption="Pixel difference visualization"
            )


# =========================================================
# TAB 5: QUANTITATIVE EVALUATION
# =========================================================

with tabs[4]:

    st.header("Quantitative Evaluation")

    st.write(
        "Evaluate the paired-image pixel-difference baseline against "
        "the known ground-truth masks of saved experiments."
    )

    threshold = st.slider(
        "Pixel difference threshold",
        min_value=0,
        max_value=100,
        value=10,
        help=(
            "A pixel is predicted as changed when the largest absolute "
            "RGB channel difference exceeds this threshold."
        )
    )

    st.markdown("### Evaluation metrics")

    st.write(
        "**Precision:** proportion of predicted changed pixels that "
        "are truly changed."
    )

    st.write(
        "**Recall:** proportion of truly changed pixels detected."
    )

    st.write(
        "**F1-score:** harmonic mean of precision and recall."
    )

    st.write(
        "**IoU:** intersection of predicted and true changed regions "
        "divided by their union."
    )

    st.write(
        "**Pixel accuracy:** proportion of all pixels classified correctly."
    )

    if st.button("Run batch evaluation", use_container_width=True):

        with st.spinner("Evaluating saved experiments..."):
            results = run_batch(threshold)

        if results:
            st.session_state.batch_results = results
            st.success(
                f"Evaluated {len(results)} saved experiments."
            )
        else:
            st.warning("No saved experiments were found.")

    results = st.session_state.get("batch_results", [])

    if results:

        df = pd.DataFrame(results)

        st.subheader("Batch results")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Metric comparison")

        chart_data = df.set_index("experiment_id")[
            ["precision", "recall", "f1_score", "iou"]
        ]

        st.bar_chart(chart_data)

        average = df[
            ["precision", "recall", "f1_score", "iou", "pixel_accuracy"]
        ].mean()

        st.subheader("Average results")

        a, b, c, d = st.columns(4)

        a.metric("Mean precision", f"{average['precision']:.4f}")
        b.metric("Mean recall", f"{average['recall']:.4f}")
        c.metric("Mean F1", f"{average['f1_score']:.4f}")
        d.metric("Mean IoU", f"{average['iou']:.4f}")

        st.download_button(
            "Download batch evaluation CSV",
            df.to_csv(index=False),
            "batch_evaluation_results.csv",
            "text/csv",
            use_container_width=True
        )


# =========================================================
# TAB 6: EXPERIMENT RECORDS
# =========================================================

with tabs[5]:

    st.header("Saved Experiment Records")

    records = read_records()

    if records:

        df = pd.DataFrame(records)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download all experiment records",
            df.to_csv(index=False),
            "experiment_records.csv",
            "text/csv",
            use_container_width=True
        )

        st.markdown("### Stored experiment files")

        st.write(
            f"Total saved experiments: **{len(records)}**"
        )

        st.write(
            "Each experiment folder contains original.png, "
            "manipulated.png, ground_truth_mask.png, "
            "predicted_mask.png, difference_map.png and metadata.json."
        )

    else:
        st.info(
            "No experiments saved yet. Generate a copy-move or "
            "splicing example to begin."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "ImageForensics | Student Research Prototype | "
    "Controlled image manipulation and paired-image evaluation"
)