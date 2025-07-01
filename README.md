# AIAMS - AI-Assisted Assembly Manual Sheet

A comprehensive toolkit for processing industrial documents, training computer vision models, and implementing AI-assisted systems for Assembly Manual Sheets (AMS). AMS documents are used in manufacturing to specify assembly procedures for components.

## Project Overview

This repository contains tools and utilities for:

- Converting and extracting data from industrial documents (PDF, Excel)
- Generating synthetic training data for machine learning models
- Training and evaluating YOLO object detection models
- Analyzing and visualizing model performance

## Repository Structure

```
AIAMS/
├── data/                # Training and testing data
│   ├── icons/           # Icon images for object detection
│   ├── main_picture/    # Background images
│   └── synthetic/       # Generated synthetic datasets
├── docs/                # Documentation files
├── runs/                # Training outputs and results
├── scripts/             # Executable shell scripts
│   ├── excel_to_pdf.sh
│   ├── extract_excel_images.sh
│   ├── extract_pdf_images.sh
│   ├── generate_synthetic_data.sh
│   ├── pdf_to_jpg.sh
│   ├── setup_model_directory.sh
│   ├── train_yolo.sh
│   ├── train_yolo_with_model_dir.sh
│   └── visualize_results.sh
└── tools/               # Python package with all functionality
    ├── converters/      # File conversion utilities
    ├── data/            # Data generation utilities
    ├── utils/           # Common utility functions
    ├── visualization/   # Results visualization
    ├── yolo/            # YOLO model training and analysis
    └── README.md        # Detailed documentation for tools package
```

## Quick Start

### Prerequisites

- Required Python packages (see requirements.txt)
- LibreOffice (for Excel conversion)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AIAMS.git
   cd AIAMS
   ```

2. Use Docker container(Recommend):
   ```bash
   # create and build the container
   make dev-up
   # Run the container
   make dev-run
   ```

3. Install dependencies and package in development mode(w/o Docker):
   ```bash
   sudo apt-get install libreoffice
   ```
   and
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. Set up your model directory (optional):
   ```bash
   ./scripts/setup_model_directory.sh
   ```

### Example Workflows

#### Document Processing

Convert Excel documents to PDF documents:
```bash
./scripts/excel_to_pdf.sh
```

Extract images from PDF documents:
```bash
./scripts/pdf_extract_jpg.sh
```

#### YOLO Model Training

1. Generate synthetic training data:
   ```bash
   ./scripts/generate_synthetic_data.sh
   ```

2. Train a YOLO model:
   ```bash
   ./scripts/train_yolo.sh
   ```

3. Visualize training results:
   ```bash
   ./scripts/visualize_results.sh
   ```

## Detailed Documentation

For detailed documentation on each module and its functionality, see the [tools README](tools/README.md).