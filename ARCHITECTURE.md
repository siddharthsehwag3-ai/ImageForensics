# ImageForensics — System Architecture

## Overall workflow

```mermaid
flowchart TD
    A[Input Images] --> B[Image Validation]
    B --> C[Image Analysis Dashboard]

    C --> D[ELA Visualization]
    C --> E[Noise Residual]
    C --> F[Edge Detection]
    C --> G[DCT Frequency Analysis]

    H[Base Images] --> I[Dataset Generator]
    J[Donor Images] --> I

    I --> K[Copy-Move Manipulation]
    I --> L[Image Splicing]

    K --> M[Ground Truth Masks]
    L --> M

    N[Original and Manipulated Pairs] --> O[Pixel Difference Baseline]
    O --> P[Predicted Masks]
    M --> Q[Evaluation]
    P --> Q

    Q --> R[Precision, Recall, F1, IoU]
    R --> S[CSV Results and Charts]
```

## Main modules

1. `app.py` — interactive Streamlit application.
2. `generate_dataset.py` — creates controlled manipulated image examples and masks.
3. `robustness_test.py` — evaluates image pairs under selected transformations.
4. `summarize_results.py` — calculates averages and creates research charts.

## Important distinction

The interactive visualizations are forensic analysis aids. The paired-image pixel-difference baseline requires an original reference image. This prototype is not a validated standalone detector for unknown forged images.
