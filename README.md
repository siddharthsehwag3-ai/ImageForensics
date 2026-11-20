# ImageForensics
### Experimental Image Manipulation Analysis and Localization Prototype

ImageForensics is a student research and demonstration project exploring image-integrity clues, controlled image manipulations and paired-image localization.

## Features

- Image upload and analysis
- Error Level Analysis (ELA) visualization
- Noise residual visualization
- Edge-map visualization
- DCT frequency visualization
- Controlled copy-move generation
- Controlled image-splicing generation
- Ground-truth mask generation
- Paired-image pixel-difference baseline
- Precision, recall, F1-score, IoU and pixel accuracy
- JPEG compression and resizing experiments
- CSV experiment records and research charts

## Technology

- Python
- Streamlit
- OpenCV
- NumPy
- Pillow
- Pandas
- Matplotlib

## Run locally

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Generate the controlled dataset

Place five base images in `dataset/base/` and five donor images in `dataset/donor/`.

Run:

```bash
python generate_dataset.py
python robustness_test.py
python summarize_results.py
```

Results are written to the `experiments/` and `research_figures/` folders.

## Research scope

The evaluation baseline compares an original image with its corresponding manipulated image. It requires both images and should not be interpreted as standalone detection of an unknown forged image.

The generated examples are controlled synthetic manipulations. Results should not be generalised to all real-world image forgeries.

ELA, noise, edge and frequency visualizations are exploratory clues, not proof of image manipulation.

## Project structure

```text
ImageForensics/
├── app.py
├── generate_dataset.py
├── robustness_test.py
├── summarize_results.py
├── requirements.txt
├── METHODOLOGY.md
├── ARCHITECTURE.md
├── DATASET_DETAILS.md
├── dataset/
├── experiments/
└── research_figures/
```

## Author

Siddharth Sehwag  
B.Tech Computer Science and Engineering  
SGT University

## Academic use

This project is intended for educational and experimental use. Cite the original sources of all external photographs and research methods used.
