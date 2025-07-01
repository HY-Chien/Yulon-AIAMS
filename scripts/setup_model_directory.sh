#!/bin/bash
# Script to set up a model directory with pre-downloaded YOLO models
# This is useful to avoid re-downloading large model files

# Define the model directory where you want to store all YOLO models
MODEL_DIR="./yolo_models"

# Create the PyTorch hub directory structure
# PyTorch uses a specific folder structure for its cache
PYTORCH_HUB_DIR="$MODEL_DIR/hub/ultralytics_yolo_master"

echo "Creating model directory structure at $MODEL_DIR"
mkdir -p "$PYTORCH_HUB_DIR"

# Check if source model files exist in the current directory
if [ -f "yolo12n.pt" ] || [ -f "yolo12s.pt" ] || [ -f "yolo12m.pt" ] || [ -f "yolo12l.pt" ] || [ -f "yolo12x.pt" ]; then
    echo "Found YOLO model files in current directory"
    
    # Copy any existing YOLO model files to the model directory
    for model in yolo12n.pt yolo12s.pt yolo12m.pt yolo12l.pt yolo12x.pt; do
        if [ -f "$model" ]; then
            echo "Copying $model to $PYTORCH_HUB_DIR"
            cp "$model" "$PYTORCH_HUB_DIR/"
        else
            echo "Model file $model not found in current directory"
        fi
    done
    
    echo "Model files copied successfully"
else
    echo "No YOLO model files found in current directory"
    echo "You will need to manually copy or download model files"
    echo "Example models: yolo12n.pt, yolo12s.pt, yolo12m.pt, yolo12l.pt"
fi

echo ""
echo "To use this model directory with the training script:"
echo "python -m tools.yolo.train --data path/to/data.yaml --model-dir \"$MODEL_DIR\" [other options]"
echo ""
echo "This will tell PyTorch to look for/download models in this directory"
echo "instead of using the default ~/.cache/torch location"
