#!/bin/bash
# Script for running Ultralytics model ensemble

# Define model paths
MODELS=(
  "./runs/train/Cv4/icons=Cv4_ds=20k(8,1,1)_epoch=60_yolo12m/weights/best.pt"
  # "./runs/train/Cv4/icons=Cv4_ds=50k(8,1,1)_epoch=60_yolo12m/weights/best.pt"

  # "./runs/train/Cv4R/icons=Cv4R_20k-medium(8,2)_epoch=30_yolo12x/weights/best.pt"
  # "./runs/train/Cv4R/icons=Cv4R_20k-medium(8,2)_epoch=40_yolo12x/weights/best.pt"
  # "./runs/train/Cv4R/icons=Cv4R_20k-medium(8,2)_epoch=40_yolo12x-is/weights/best.pt"
  "./runs/train/Cv4R/icons=Cv4R_30k-medium(8,2)_epoch=40_yolo12x-is/weights/best.pt"

  # "./models/yolov8l.pt"
  # "./models/yolov8x.pt"
  # "./runs/train/exp1/weights/best.pt"
  # "./runs/train/exp2/weights/best.pt"
)

# Define model weights (optional - comment out for equal weights)
WEIGHTS=(
  0.5
  # 0.5
  # 0.0
  # 0.0
  # 0.0
  0.8
  # 0.8
  # 0.25
  # 0.5
  # 0.5
)

# Ensemble parameters
ENSEMBLE_METHOD="nms"  # Options: nms, wbf, avg
CONF_THRESHOLD=0.25
IOU_THRESHOLD=0.25
MAX_DET=300
DEVICE="cuda:0"  # Options: auto, cpu, cuda, cuda:0, etc.

# Input source (can be image, video, directory, etc.)
# SOURCE="./data/test_image.jpg"           # Single image
# SOURCE="./data/test_video.mp4"           # Single video
SOURCE="./data/case_study/D"               # Folder of images (batch processing)
# SOURCE="./data/validation/images/"       # Validation dataset folder

# Output directory
OUTPUT_DIR="./results/ensemble/"
SAVE_VISUALIZATION=true  # Set to false to disable saving visualizations
BATCH_PROCESSING=true    # Set to true when processing folders

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "🚀 Starting ensemble inference..."
echo "Models: ${#MODELS[@]}"
echo "Source: $SOURCE"
echo "Method: $ENSEMBLE_METHOD"
echo "Confidence: $CONF_THRESHOLD"
echo "IoU: $IOU_THRESHOLD"
echo "Output: $OUTPUT_DIR"

# Check if source is a directory for batch processing
if [[ -d "$SOURCE" ]]; then
  echo "📁 Batch processing mode: Processing folder of images"
  # Count images in folder
  IMAGE_COUNT=$(find "$SOURCE" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) | wc -l)
  echo "📸 Found $IMAGE_COUNT images to process"
elif [[ -f "$SOURCE" ]]; then
  echo "🖼️  Single file processing mode"
else
  echo "❌ Error: Source not found: $SOURCE"
  exit 1
fi

# Build model arguments
MODEL_ARGS=""
for model in "${MODELS[@]}"; do
  MODEL_ARGS+="$model "
done

# Build weights arguments (if defined)
WEIGHT_ARGS=""
if [ ${#WEIGHTS[@]} -gt 0 ]; then
  for weight in "${WEIGHTS[@]}"; do
    WEIGHT_ARGS+="$weight "
  done
fi

# Build save path for visualization
SAVE_ARG=""
if [ "$SAVE_VISUALIZATION" = true ]; then
  if [[ -d "$SOURCE" ]] && [ "$BATCH_PROCESSING" = true ]; then
    # For batch processing, create subdirectory for ensemble results
    BATCH_OUTPUT="${OUTPUT_DIR}ensemble_results/"
    mkdir -p "$BATCH_OUTPUT"
    echo "📁 Batch output directory: $BATCH_OUTPUT"
    # Pass the batch output directory to Python script
    SAVE_ARG="--save-dir $BATCH_OUTPUT"
  elif [[ -f "$SOURCE" ]]; then
    # For single image, specify exact save path
    FILENAME=$(basename "$SOURCE")
    NAME="${FILENAME%.*}"
    EXT="${FILENAME##*.}"
    SAVE_ARG="--save ${OUTPUT_DIR}${NAME}_ensemble.${EXT}"
  fi
fi

# Execute ensemble script
echo "🔄 Executing ensemble command..."
if [ ${#WEIGHTS[@]} -gt 0 ]; then
  # With custom weights
  echo "Using custom weights: ${WEIGHTS[*]}"
  python tools/yolo/ensemble_inference.py \
    --models $MODEL_ARGS \
    --source "$SOURCE" \
    --weights $WEIGHT_ARGS \
    --method "$ENSEMBLE_METHOD" \
    --conf "$CONF_THRESHOLD" \
    --iou "$IOU_THRESHOLD" \
    --device "$DEVICE" \
    $SAVE_ARG
else
  # With equal weights
  echo "Using equal weights for all models"
  python tools/yolo/ensemble_inference.py \
    --models $MODEL_ARGS \
    --source "$SOURCE" \
    --method "$ENSEMBLE_METHOD" \
    --conf "$CONF_THRESHOLD" \
    --iou "$IOU_THRESHOLD" \
    --device "$DEVICE" \
    $SAVE_ARG
fi

# Check if Python script executed successfully
if [ $? -eq 0 ]; then
  echo "✅ Python script executed successfully"
else
  echo "❌ Python script failed with exit code: $?"
  exit 1
fi

echo "✅ Ensemble complete. Results processed."
if [ "$SAVE_VISUALIZATION" = true ]; then
  if [[ -d "$SOURCE" ]] && [ "$BATCH_PROCESSING" = true ]; then
    echo "📁 Batch processing complete. Check results in: $OUTPUT_DIR"
    echo "📊 Individual results and visualizations saved for each image"
  elif [[ -f "$SOURCE" ]]; then
    echo "📸 Visualization saved in: $OUTPUT_DIR"
  fi
fi