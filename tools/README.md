# AIAMS Tools Package

A comprehensive Python package for AI-Assisted Assembly Manual Sheet (AIAMS) containing tools for data processing, model training, and visualization.

## Overview

This package provides modular utilities for:
- Converting and processing PDF and Excel files
- Extracting images from documents
- Generating synthetic data for YOLO model training
- Training and evaluating YOLO object detection models
- Visualizing training results

## Package Structure

```
tools/
├── __init__.py
├── utils/                 # Common utility functions
│   ├── __init__.py
│   ├── file_utils.py      # File operations utilities
│   └── image_utils.py     # Image processing utilities
├── converters/            # File conversion utilities
│   ├── __init__.py
│   ├── excel_utils.py     # Excel conversion and extraction
│   └── pdf_utils.py       # PDF conversion and extraction
├── data/                  # Data generation utilities
│   ├── __init__.py
│   └── synthetic_data.py  # Synthetic dataset generation
├── unused/                # Legacy and archive code
│   ├── case_study.py
│   ├── csv_to_tensorboard.py
│   ├── excel_to_jpg.py
│   ├── excel_to_pdf.py
│   ├── extract_excel_images.py
│   ├── extract_pdf_images.py
│   ├── generate_synthetic_data.py
│   ├── pdf_to_jpg.py
│   ├── plot_results.py
│   ├── test_validation.py
│   └── train_yolo.py
├── yolo/                  # YOLO model utilities
│   ├── __init__.py
│   ├── case_study.py      # Model analysis and comparison
│   ├── test_validation.py # Model validation
│   └── train.py           # Model training
└── visualization/         # Visualization tools
    ├── __init__.py
    ├── csv_to_tensorboard.py  # Convert CSV to TensorBoard
    └── plot_results.py        # Plot training metrics
```

## Installation and Prerequisites

### Dependencies

- Python 3.7+
- PIL/Pillow for image processing
- PyMuPDF for PDF processing
- pdf2image for PDF to image conversion
- ultralytics for YOLO training
- matplotlib for visualization
- TensorBoard for training visualization (optional)

### Setup

Clone the repository and install required packages:

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
pip install -e .
```

This installs the package in development mode, making it importable from any directory.

## Module Documentation

### Utils Module

Common utility functions used across the package.

#### file_utils.py

```python
from tools.utils.file_utils import ensure_directory_exists, find_executable

# Create directory if it doesn't exist
output_dir = ensure_directory_exists("/path/to/output")

# Find executable in system path or predefined locations
libreoffice_path = find_executable("soffice", [
    "/Applications/LibreOffice.app/Contents/MacOS/soffice"
])
```

#### image_utils.py

Image processing utilities for resizing, converting, and manipulating images.

### Converters Module

Utilities for converting between file formats.

#### pdf_utils.py

PDF processing utilities for converting PDFs to images and extracting embedded images.

**Command-line usage:**

```bash
# Convert PDF to JPG images
python -m tools.converters.pdf_utils convert input.pdf --output output_dir --dpi 300

# Extract images from PDF
python -m tools.converters.pdf_utils extract input.pdf --output output_dir --min-width 100 --min-height 100

# Batch conversion
python -m tools.converters.pdf_utils batch-convert pdf_dir --output output_dir --dpi 300

# Batch extraction
python -m tools.converters.pdf_utils batch-extract pdf_dir --output output_dir --min-width 100 --min-height 100
```

**Python API usage:**

```python
from tools.converters.pdf_utils import pdf_to_jpg, extract_images_from_pdf

# Convert PDF to JPG
images = pdf_to_jpg("input.pdf", "output_dir", dpi=300)

# Extract images from PDF
count, skipped, files = extract_images_from_pdf(
    "input.pdf", 
    "output_dir",
    min_width=100,
    min_height=100
)
```

#### excel_utils.py

Excel processing utilities for converting Excel files to PDF/JPG and extracting embedded images.

**Command-line usage:**

```bash
# Convert Excel to PDF
python -m tools.converters.excel_utils to-pdf input.xlsx --output output_dir

# Convert Excel to JPG (via PDF)
python -m tools.converters.excel_utils to-jpg input.xlsx --output output_dir --dpi 300

# Extract images from Excel
python -m tools.converters.excel_utils extract input.xlsx --output output_dir --min-width 100 --min-height 100

# Batch operations
python -m tools.converters.excel_utils batch-to-pdf excel_dir --output output_dir
python -m tools.converters.excel_utils batch-to-jpg excel_dir --output output_dir --dpi 300
python -m tools.converters.excel_utils batch-extract excel_dir --output output_dir --min-width 100
```

**Python API usage:**

```python
from tools.converters.excel_utils import excel_to_pdf_libreoffice, excel_to_jpg, extract_images_from_excel

# Convert Excel to PDF
pdf_path = excel_to_pdf_libreoffice("input.xlsx", "output_dir")

# Convert Excel to JPG images (via PDF)
image_paths = excel_to_jpg("input.xlsx", "output_dir", dpi=300)

# Extract images from Excel
extracted_count, skipped_count, files = extract_images_from_excel(
    "input.xlsx",
    "output_dir",
    min_width=100,
    min_height=100
)
```

### Data Module

Tools for generating synthetic data for model training.

#### synthetic_data.py

Generates synthetic data for training YOLO models by placing icons on backgrounds with various transformations.

**Command-line usage:**

```bash
python -m tools.data.synthetic_data \
  --icons icons_dir \
  --backgrounds backgrounds_dir \
  --output output_dataset_dir \
  --num-images 100 \
  --min-icons 1 \
  --max-icons 5 \
  --min-scale 0.05 \
  --max-scale 0.25 \
  --train-split 0.7 \
  --val-split 0.15 \
  --test-split 0.15 \
  --seed 42
```

**Python API usage:**

```python
from tools.data.synthetic_data import SyntheticDataGenerator

generator = SyntheticDataGenerator(seed=42)
generator.generate_synthetic_data(
    icons_folder="icons_dir",
    backgrounds_folder="backgrounds_dir",
    output_folder="output_dir",
    num_images=100,
    min_icons_per_image=1,
    max_icons_per_image=5,
    min_icon_scale=0.05,
    max_icon_scale=0.25,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15
)
```

### YOLO Module

Tools for training, validating, and analyzing YOLO object detection models.

#### train.py

Comprehensive YOLO model training with support for custom datasets, hyperparameter tuning, and various training configurations.

**Command-line usage:**

```bash
# Basic training
python -m tools.yolo.train --data dataset.yaml --model-size m --pretrained --epochs 100

# With custom model directory
python -m tools.yolo.train --data dataset.yaml --model-size m --model-dir /path/to/models --pretrained

# Creating dataset from paths
python -m tools.yolo.train --train-path train_images --val-path val_images --classes car person bicycle

# Advanced options
python -m tools.yolo.train \
  --data dataset.yaml \
  --model-size l \
  --pretrained \
  --epochs 300 \
  --batch-size 32 \
  --imgsz 640 \
  --device 0 \
  --optimizer Adam \
  --lr0 0.001 \
  --project runs/my_experiment \
  --name run1 \
  --export \
  --export-format onnx
```

**Python API usage:**

```python
from tools.yolo.train import YOLOTrainer

trainer = YOLOTrainer()

# Create dataset configuration
dataset_yaml = trainer.create_dataset_yaml(
    train_path="train_images",
    val_path="val_images",
    class_names=["car", "person", "bicycle"],
    output_path="my_dataset.yaml"
)

# Train model
model_path = trainer.train(
    data_yaml=dataset_yaml,
    model_size="m",
    epochs=100,
    batch_size=16,
    imgsz=640,
    device="0",  # Use first GPU
    pretrained=True,
    model_dir="/path/to/models"  # Custom model directory
)

# Validate model
metrics = trainer.validate(
    model_path=model_path,
    data_yaml=dataset_yaml,
    batch_size=16,
    imgsz=640
)

# Export model
exported_path = trainer.export(
    model_path=model_path,
    format="onnx",
    imgsz=640,
    half=True  # Use FP16 precision
)
```

#### test_validation.py

Validate YOLO models with detailed metrics.

**Command-line usage:**

```bash
python -m tools.yolo.test_validation --model runs/train/exp/weights/best.pt --data dataset.yaml
```

**Python API usage:**

```python
from tools.yolo.test_validation import YOLOValidator

validator = YOLOValidator()
results = validator.run_validation(
    model_path="runs/train/exp/weights/best.pt",
    data_yaml="dataset.yaml",
    batch=16,
    imgsz=640
)
```

#### case_study.py

Analyze and compare YOLO models on specific test images or datasets.

**Command-line usage:**

```bash
# Run on a single image
python -m tools.yolo.case_study --single --model runs/train/exp/weights/best.pt --image test.jpg

# Compare multiple models
python -m tools.yolo.case_study --compare --image test.jpg --models model1.pt model2.pt model3.pt --model-names "YOLOv12n" "YOLOv12m" "YOLOv12l"

# Batch analysis
python -m tools.yolo.case_study --batch --model runs/train/exp/weights/best.pt --data dataset.yaml --num-samples 10
```

**Python API usage:**

```python
from tools.yolo.case_study import run_case_study, compare_models, batch_analysis

# Analyze single image
run_case_study(
    model_path="runs/train/exp/weights/best.pt",
    image_path="test.jpg",
    output_dir="results/case_study",
    conf=0.3
)

# Compare multiple models
compare_models(
    image_path="test.jpg",
    model_paths=["model1.pt", "model2.pt", "model3.pt"],
    model_names=["YOLOv12n", "YOLOv12m", "YOLOv12l"],
    output_dir="results/comparison",
    export_individual=True
)
```

### Visualization Module

Tools for visualizing training results and metrics.

#### plot_results.py

Plot training metrics from CSV files generated during YOLO training.

**Command-line usage:**

```bash
# Basic plotting
python -m tools.visualization.plot_results runs/train/exp/results.csv

# Custom output with smoothing options
python -m tools.visualization.plot_results runs/train/exp/results.csv --output-dir plots --smooth-factor 0.8
```

**Python API usage:**

```python
from tools.visualization.plot_results import plot_results

plot_results(
    csv_file="runs/train/exp/results.csv",
    output_dir="plots",
    smooth=True,
    smooth_factor=0.6
)
```

#### csv_to_tensorboard.py

Convert CSV training metrics to TensorBoard format for interactive visualization.

**Command-line usage:**

```bash
python -m tools.visualization.csv_to_tensorboard runs/train/exp/results.csv --log_dir tensorboard_logs
```

**Python API usage:**

```python
from tools.visualization.csv_to_tensorboard import csv_to_tensorboard

log_dir = csv_to_tensorboard("runs/train/exp/results.csv", "tensorboard_logs")
```

## Scripts

The `scripts/` directory contains example shell scripts demonstrating how to use the tools for common tasks.

### PDF Processing Scripts

- `pdf_to_jpg.sh`: Convert PDF files to JPG images
- `extract_pdf_images.sh`: Extract embedded images from PDF files

### Excel Processing Scripts

- `excel_to_pdf.sh`: Convert Excel files to PDF
- `extract_excel_images.sh`: Extract embedded images from Excel files

### YOLO Training Scripts

- `train_yolo.sh`: Train YOLO models
- `train_yolo_with_model_dir.sh`: Train YOLO models with custom model directory
- `visualize_results.sh`: Visualize training results

### Model Management Scripts

- `setup_model_directory.sh`: Set up a custom model directory and copy existing model files

### Data Generation Scripts

- `generate_synthetic_data.sh`: Generate synthetic data for YOLO training

## Practical Examples

### Example 1: Complete Workflow for Training on Custom Dataset

```bash
# Step 1: Generate synthetic data
./scripts/generate_synthetic_data.sh

# Step 2: Set up model directory
./scripts/setup_model_directory.sh

# Step 3: Train YOLO model with custom model directory
./scripts/train_yolo_with_model_dir.sh

# Step 4: Visualize training results
./scripts/visualize_results.sh
```

### Example 2: Converting Documents to Images

```bash
# Convert PDF documents to images
python -m tools.converters.pdf_utils batch-convert pdfs/ --output images/ --dpi 300

# Convert Excel spreadsheets to images
python -m tools.converters.excel_utils batch-to-jpg excel/ --output images/ --dpi 300
```

### Example 3: Extract Images from Documents

```bash
# Extract images from PDFs
python -m tools.converters.pdf_utils batch-extract pdfs/ --output extracted_images/ --min-width 100 --min-height 100

# Extract images from Excel files
python -m tools.converters.excel_utils batch-extract excel/ --output extracted_images/ --min-width 100 --min-height 100
```
