#!/bin/bash
# Simple script to plot training results comparison for multiple YOLO models

# Define model paths and names as lists
MODELS=(
  # "/home/W20862/AIAMS/runs/train/icons=Cv4_bg=all_ds=50k(8,1,1)_epoch=10_yolo12s"
  # "/home/W20862/AIAMS/runs/train/icons=Cv4_bg=all_ds=5k(8,1,1)_epoch=60_yolo12s"
  # "/home/W20862/AIAMS/runs/train/icons=Cv4_bg=all_ds=5k(8,1,1)_epoch=80_yolo12s"
  # "/home/W20862/AIAMS/runs/train/icons=Cv4_bg=all_ds=30k(8,1,1)_epoch=60_yolo12s"
  # "/home/W20862/AIAMS/runs/train/icons=Cv4_bg=all_ds=30k(8,1,1)_epoch=80_yolo12s"
  # "/home/W20862/AIAMS/runs/train/icons=Cv4_bg=all_ds=50k(8,1,1)_epoch=60_yolo12s"
  # "/home/W20862/AIAMS/runs/train/icons=Cv4_bg=all_ds=50k(8,1,1)_epoch=80_yolo12s"
  "./runs/train/Cv4/icons=Cv4_ds=20k(8,1,1)_epoch=60_yolo12m"
  # "./runs/train/Cv4/icons=Cv4_ds=20k(8,1,1)_epoch=80_yolo12m"
  # "./runs/train/Cv4/icons=Cv4_ds=50k(8,1,1)_epoch=60_yolo12m"
  # "./runs/train/Cv4/icons=Cv4_ds=50k(8,1,1)_epoch=80_yolo12m"
  # "./runs/train/icons=Cv4-p1_10k(8,1,1)_epoch=40_yolo12m"
  # "./runs/train/icons=Cv4-p2_10k(8,1,1)_epoch=40_yolo12m"
  "./runs/train/icons=Cv4_20k-none(8,2)_epoch=40_yolo12m"
  "./runs/train/icons=Cv4_20k-none(8,2)_epoch=40_yolo12x"
  "./runs/train/icons=Cv4_20k-none(8,2)_epoch=60_yolo12m"
  "./runs/train/icons=Cv4_20k-none(8,2)_epoch=60_yolo12x"

  "./runs/train/Cv4R/icons=Cv4R_20k-medium(8,2)_epoch=10_yolo12x"
  "./runs/train/Cv4R/icons=Cv4R_20k-medium(8,2)_epoch=20_yolo12x"
)

# Define model names for better readability
MODEL_NAMES=(
  # "50k_e10_12s"
  # "5k_e60_12s"
  # "5k_e80_12s"
  # "30k_e60_12s"
  # "30k_e80_12s"
  # "50k_e60_12s"
  # "50k_e80_12s"
  "60_20k"
  # "80_20k"
  # "60_50k"
  # "80_50k"
  # "p1-10_10k"
  # "p2-10_10k"
  "m40_20k-none"
  "x40_20k-none"
  "m60_20k-none"
  "x60_20k-none"

  "x10_20k-medium"
  "x20_20k-medium"
)

# Build CSV file paths from model directories
CSV_FILES=()
VALID_MODEL_NAMES=()

echo "Looking for results.csv files..."
for i in "${!MODELS[@]}"; do
  csv_file="${MODELS[$i]}/results.csv"
  if [[ -f "$csv_file" ]]; then
    CSV_FILES+=("$csv_file")
    VALID_MODEL_NAMES+=("${MODEL_NAMES[$i]}")
    echo "✓ Found: ${MODEL_NAMES[$i]}"
  else
    echo "✗ Missing: ${MODEL_NAMES[$i]} (${csv_file})"
  fi
done

# Check if we have any valid files
if [[ ${#CSV_FILES[@]} -eq 0 ]]; then
  echo "Error: No results.csv files found!"
  exit 1
fi

echo ""
echo "Generating plots for ${#CSV_FILES[@]} models..."

# Call the Python plotting script
python -m tools.visualization.plot_results \
  "${CSV_FILES[@]}" \
  --model-names "${VALID_MODEL_NAMES[@]}" \
  --output-dir "./results/training_comparison" \
  --figsize 16 10

echo "Done! Check ./results/training_comparison/ for plots."