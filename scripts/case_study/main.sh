#!/bin/bash
# Main entry script for YOLO case study functionality
# This script routes to the appropriate sub-script based on the mode

# Source common utilities
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/common.sh"

# Print help information
print_help() {
  echo "YOLO Case Study - Analysis and visualization tools for YOLO models"
  echo ""
  echo "Usage: $0 [mode] [options]"
  echo ""
  echo "Modes:"
  echo "  --single         Analyze a single image or directory of images"
  echo "  --batch          Analyze random samples from a dataset"
  echo "  --compare        Compare multiple models on the same image"
  echo "  --help           Show this help message"
  echo ""
  echo "For mode-specific options, run:"
  echo "  $0 [mode] --help"
  echo ""
  echo "Examples:"
  echo "  $0 --single --model runs/train/experiment_1/weights/best.pt --image test/sample.jpg"
  echo "  $0 --batch --model runs/train/experiment_1/weights/best.pt --data data/dataset.yaml --num-samples 10"
  echo "  $0 --compare --image test/sample.jpg --models model1.pt model2.pt --model-names \"Model A\" \"Model B\""
  echo ""
  echo "Notes:"
  echo "- All modes support --conf, --iou, --device, --output-dir, and --no-show parameters"
  echo "- Visualizations are saved to the output directory (default: runs/detect/case_study/[mode])"
}

# Check if no arguments provided
if [ $# -eq 0 ]; then
  print_error "No arguments provided"
  print_help
  exit 1
fi

# Parse the first argument to determine the mode
MODE="$1"
shift  # Remove the mode argument

case "$MODE" in
  --single)
    # Run the single image analysis script
    "$SCRIPT_DIR/single_image.sh" "$@"
    exit $?
    ;;
  --batch)
    # Run the batch analysis script
    "$SCRIPT_DIR/batch_analysis.sh" "$@"
    exit $?
    ;;
  --compare)
    # Run the model comparison script
    "$SCRIPT_DIR/compare_models.sh" "$@"
    exit $?
    ;;
  --help)
    print_help
    exit 0
    ;;
  *)
    print_error "Unknown mode: $MODE"
    print_help
    exit 1
    ;;
esac
