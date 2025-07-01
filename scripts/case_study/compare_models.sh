#!/bin/bash
# Script for comparing YOLO model performance using the Case Study tool

# Define model paths (base directories for each model)
MODELS=(
  # Cv4
  "./runs/train/Cv4/icons=Cv4_20k-none(8,2)_epoch=40_yolo12x"

  # Cv4-p12
  # "./runs/train/icons=Cv4-p1_10k(8,1,1)_epoch=40_yolo12m"
  # "./runs/train/icons=Cv4-p2_10k(8,1,1)_epoch=40_yolo12m"

  # Cv4
  # "./runs/train/Cv4/icons=Cv4_20k-none(8,2)_epoch=40_yolo12m"
  # "./runs/train/Cv4/icons=Cv4_20k-none(8,2)_epoch=40_yolo12x"
  # "./runs/train/Cv4/icons=Cv4_20k-none(8,2)_epoch=60_yolo12m"
  # "./runs/train/Cv4/icons=Cv4_20k-none(8,2)_epoch=60_yolo12x"

  # Cv4R
  # "./runs/train/Cv4R/icons=Cv4R_20k-medium(8,2)_epoch=10_yolo12x"
  # "./runs/train/Cv4R/icons=Cv4R_20k-medium(8,2)_epoch=20_yolo12x"
  # "./runs/train/Cv4R/icons=Cv4R_20k-medium(8,2)_epoch=30_yolo12x"
  # "./runs/train/Cv4R/icons=Cv4R_20k-medium(8,2)_epoch=40_yolo12x"
  # "./runs/train/Cv4R/icons=Cv4R_20k-medium(8,2)_epoch=40_yolo12x-is"
  "./runs/train/Cv4R/icons=Cv4R_20k-medium(8,2)_epoch=60_yolo12x"
  "./runs/train/Cv4R/icons=Cv4R_20k-medium(8,2)_epoch=80_yolo12x"
  "./runs/train/Cv4R/icons=Cv4R_30k-medium-is(8,2)_epoch=40_yolo12x-is"
  "./runs/train/Cv4R/icons=Cv4R_50k-medium(8,2)_epoch=40_yolo12x-is"
)

# Define model names for better readability (ensure this matches the MODELS array)
MODEL_NAMES=(
  "e60_20k_m"
  # "80_20k_m"

  # "60_50k_m"
  # "80_50k_m"

  # "p1_10k_e40_m"
  # "p2_10k_e40_m"

  # "e40_20k-none_m"   # Example: Cv4_20k-none(8,2)_epoch=40_yolo12m
  # "e40_20k-none_x"   # Example: Cv4_20k-none(8,2)_epoch=40_yolo12m
  # "e60_20k-none_m"   # Example: Cv4_20k-none(8,2)_epoch=60_yolo12m
  # "e60_20k-none_x"   # Example: Cv4_20k-none(8,2)_epoch=60_yolo12x

  # "e10_20k-medium_x"   # Example: Cv4_20k-none(8,2)_epoch=60_yolo12m
  # "e20_20k-medium_x"   # Example: Cv4_20k-none(8,2)_epoch=60_yolo12x
  # "e30_20k-medium_x"   # Example: Cv4_20k-none(8,2)_epoch=60_yolo12m
  # "e40_20k-medium_x"   # Example: Cv4_20k-none(8,2)_epoch=60_yolo12x
  "e60_20k-medium_x"   # Example: Cv4_20k-none(8,2)_epoch=60_yolo12x
  "e80_20k-medium_x"   # Example: Cv4_20k-none(8,2)_epoch=60_yolo12x
  "e40_30k-medium_x-is"   # Example: Cv4_20k-none(8,2)_epoch=60_yolo12x
  "e40_50k-medium_x-is"   # Example: Cv4_20k-none(8,2)_epoch=60_yolo12x
)

# Detection parameters
CONF_THRESHOLD=0.30
IOU_THRESHOLD=0.2 # Note: An IoU of 0.0 might be specific for certain NMS behavior

# --- Image Path Configuration ---
# Collect all image paths into an array
declare -a IMAGE_PATHS_LIST
for i in {1..7}; do
  IMAGE_PATHS_LIST+=("./data/case_study/D/$i.png")
done

# --- Output Directory Configuration ---
# Define a single main output directory for the combined comparison.
# The Python script will save individual image comparisons within this directory.
MAIN_OUTPUT_DIR="./results/case_study/D/Cv4/"

# Create the output directory if it doesn't exist
mkdir -p "$MAIN_OUTPUT_DIR"

echo "🚀 Starting model comparison..."
echo "Models to compare: ${#MODELS[@]}"
echo "Images to process: ${IMAGE_PATHS_LIST[*]}"
echo "Output directory: $MAIN_OUTPUT_DIR"
echo "Confidence threshold: $CONF_THRESHOLD"
echo "IoU threshold: $IOU_THRESHOLD"

# --- Argument Building ---
# Build the --models arguments (list of .pt file paths)
MODEL_PT_FILES_ARGS=""
for model_base_path in "${MODELS[@]}"; do
  # IMPORTANT: Assuming all your relevant model weights are named 'epoch30.pt'
  # If your weight files have different names or are not consistently 'epoch30.pt',
  # you'll need to adjust this logic or ensure your MODELS array points directly to .pt files.
  MODEL_PT_FILES_ARGS+="${model_base_path}/weights/best.pt "
done

# Build the --model-names arguments
MODEL_NAME_ARGS=""
for name in "${MODEL_NAMES[@]}"; do
  MODEL_NAME_ARGS+="${name} "
done

# Convert IMAGE_PATHS_LIST array to a space-separated string for the command line
IMAGE_ARGS="${IMAGE_PATHS_LIST[*]}"

# --- Python Script Execution ---
# Call the Python script ONCE with all images and models
python -m tools.yolo.case_study --compare \
  --image $IMAGE_ARGS \
  --models $MODEL_PT_FILES_ARGS \
  --model-names $MODEL_NAME_ARGS \
  --conf "$CONF_THRESHOLD" \
  --iou "$IOU_THRESHOLD" \
  --output-dir "$MAIN_OUTPUT_DIR" \
  --export-individual \
  --no-show # Added --no-show as plots for many images might be overwhelming

echo "✅ Comparison complete. Results saved in $MAIN_OUTPUT_DIR"