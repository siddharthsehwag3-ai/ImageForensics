# Research Methodology

## 1. Study Design
This project is a controlled experimental prototype for studying image manipulation and pixel-level localization.

## 2. Manipulation Categories
Two manipulation types are generated:
- Copy-move: a rectangular region is copied from one location and placed at another location in the same image.
- Image splicing: a rectangular patch from a donor image is inserted into a base image.

## 3. Dataset
The demonstration dataset contains five base photographs and five donor photographs. The generator creates five copy-move and five splicing examples.

The generated masks identify the destination region modified by each operation. These are synthetic, controlled examples and are not a representative collection of real-world forged images.

## 4. Baseline Method
The paired-image baseline compares an original image with its manipulated counterpart.

For each pixel (x,y), the difference score is:

D(x,y) = max over RGB channels of |I_original(x,y,c) - I_manipulated(x,y,c)|

A pixel is predicted as changed when D(x,y) exceeds a fixed threshold of 10.

## 5. Evaluation
Predicted masks are compared with the generated ground-truth masks using:
- Precision
- Recall
- F1-score
- Intersection over Union (IoU)
- Pixel accuracy

## 6. Robustness Conditions
The paired images are evaluated under:
- Original PNG condition
- JPEG quality 95
- JPEG quality 75
- JPEG quality 50
- Resize to 75% followed by restoration
- Resize to 50% followed by restoration

The same transformation is applied to both members of each image pair. The ground-truth mask remains at the original dimensions.

## 7. Reproducibility
The scripts generate experiment folders, metadata, CSV results and charts. The threshold is kept fixed at 10 for all conditions.

## 8. Scope and Limitations
This baseline requires access to both the original and manipulated image. It is not a standalone detector for an unknown photograph.

The rectangular manipulations are controlled and relatively simple. The study does not establish performance on arbitrary real-world image forgeries.

ELA, noise residuals, edge maps and DCT visualizations are exploratory forensic clues. They do not independently prove manipulation.

## 9. Future Work
Future work may evaluate public benchmark datasets, unknown-image detection, more varied manipulation shapes, neural-network baselines, cross-dataset performance and calibrated confidence estimates.
