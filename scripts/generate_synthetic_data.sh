#!/bin/bash

# script.sh - Synthetic data generation for YOLO object detection
# ===============================================================

# Exit on any error
set -e

##########################################################
# Configuration
ICONS_DIR=./data/icons/Cv4-removed
BACKGROUNDS_DIR=./data/main_picture
OUTPUT_DIR="./data/synthetic/icon=Cv4R_50k-medium-is"
NUM_IMAGES=50000
MIN_ICONS=0
MAX_ICONS=15
AUGMENTATION=medium
SEED=42

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Display setup information
echo "===== Synthetic Data Generator Setup ====="
echo "Icons directory: $ICONS_DIR"
echo "Backgrounds directory: $BACKGROUNDS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Number of images: $NUM_IMAGES"
echo "Icons per image: $MIN_ICONS-$MAX_ICONS"
echo "Augmentation strength: $AUGMENTATION"
echo "Random seed: $SEED"
echo "========================================"

# Run the synthetic data generator
python -m tools.data.synthetic_data \
  --icons "$ICONS_DIR" \
  --backgrounds "$BACKGROUNDS_DIR" \
  --output "$OUTPUT_DIR" \
  --num-images "$NUM_IMAGES" \
  --min-icons "$MIN_ICONS" \
  --max-icons "$MAX_ICONS" \
  --min-scale 0.07 \
  --max-scale 0.08 \
  --augmentation "$AUGMENTATION" \
  --train-split 0.8 \
  --val-split 0.2 \
  --test-split 0.0 \
  --seed "$SEED" \
  --min-icon-size 100 \
  --max-icon-size 120
  # --augment-backgrounds

# Provide a summary of the generated dataset
echo ""
echo "===== Dataset Generation Complete ====="
echo "Dataset saved to: $OUTPUT_DIR"
echo "Train images: $(ls "$OUTPUT_DIR/train/images" | wc -l)"
echo "Validation images: $(ls "$OUTPUT_DIR/val/images" | wc -l)"
echo "Test images: $(ls "$OUTPUT_DIR/test/images" | wc -l)"
echo "Total images: $(find "$OUTPUT_DIR" -path "*/images/*.jpg" | wc -l)"
echo "Number of classes: $(grep "nc:" "$OUTPUT_DIR/data.yaml" | cut -d' ' -f2)"
echo "Class names: $(grep "names:" "$OUTPUT_DIR/data.yaml" | cut -d':' -f2-)"
echo "======================================="
echo ""
echo "To train with YOLOv12:"
echo "yolo train model=yolov8s.pt data=$OUTPUT_DIR/data.yaml epochs=100"
echo ""

