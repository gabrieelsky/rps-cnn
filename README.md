# Rock-Paper-Scissors CNN Classification

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-red.svg)

## What the project does
This project trains and evaluates convolutional neural networks (CNNs) to classify Rock, Paper, and Scissors hand gestures from images. The workflow emphasizes reproducibility, careful train/val/test splitting, and data-leakage prevention while providing multiple model architectures and evaluation utilities.

- Provides a complete, reproducible pipeline for a small image classification task.
- Includes multiple model architectures (baseline CNN, deeper CNN, and a tiny ResNet-style model).
- Implements clean train/validation/test splits and normalization based only on training data.
- Logs evaluation artifacts and misclassifications for error analysis.

## Project layout
```
.
├── main.ipynb
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── paper/
│   │   ├── rock/
│   │   └── scissors/
│   └── custom_test/
│       ├── paper/
│       ├── rock/
│       └── scissors/
├── report/
│   └── report.pdf
├── saved_models/
│   ├── baseline_cnn.pth
│   ├── grid_search_results.csv
│   ├── medium_cnn.pth
│   └── micro_resnet.pth
└── src/
    ├── config.py
    ├── data_loader.py
    ├── evaluate.py
    ├── models.py
    ├── train.py
    └── utils.py
```

## How to get started

### 1) Set up the environment
The provided [requirements.txt](requirements.txt) is a Conda export, not a pip-style file.

```bash
conda create --name rps-cnn --file requirements.txt
conda activate rps-cnn
```

### 2) Prepare data
Place images under [data/raw](data/raw) using the class subfolders shown above. For an out-of-sample check, place images under [data/custom_test](data/custom_test) with the same class folder structure.

### 3) Run the notebook
Open and run [main.ipynb](main.ipynb). It drives the end-to-end workflow: data loading, training, evaluation, and saving results.
