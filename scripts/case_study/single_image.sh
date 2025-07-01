#!/bin/bash
# Script for analyzing a single image with YOLO

# Source common utilities
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/common.sh"

# Function to analyze a single image with a trained model
analyze_single_image() {
  local model="$1"
  local image="$2"
  local conf="${3:-0.25}"
  local iou="${4:-0.45}"
  local output_dir="${5:-$OUTPUT_DIR/single}"
  local device="${6:-}"
  local show="${7:-true}"
  
  # Verify model file exists
  if ! check_file_exists "$model" "Model not found: $model"; then
    print_message "Please specify a valid model path with --model"
    return 1
  fi
  
  # Verify image file exists
  if ! check_file_exists "$image" "Image not found: $image"; then
    print_message "Please specify a valid image path with --image"
    return 1
  fi
  
  # Set up show/no-show flag
  local show_flag=""
  if [ "$show" = "false" ]; then
    show_flag="--no-show"
  fi
  
  # Run the prediction
  print_message "Running case study on image: $image"
  print_message "Using model: $model"
  print_message "Output directory: $output_dir"
  
  python -m tools.yolo.case_study --single \
    --model "$model" \
    --image "$image" \
    --conf "$conf" \
    --iou "$iou" \
    --output-dir "$output_dir" \
    ${device:+--device "$device"} \
    $show_flag
    
  return $?
}

# If this script is run directly, not sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  # Parse command line arguments
  MODEL="$MODEL_PATH"
  IMAGE=""
  CONF="0.25"
  IOU="0.45"
  OUTPUT="$OUTPUT_DIR/single"
  DEVICE=""
  SHOW="true"
  
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)
        MODEL="$2"
        shift 2
        ;;
      --image)
        IMAGE="$2"
        shift 2
        ;;
      --conf)
        CONF="$2"
        shift 2
        ;;
      --iou)
        IOU="$2"
        shift 2
        ;;
      --output-dir)
        OUTPUT="$2"
        shift 2
        ;;
      --device)
        DEVICE="$2"
        shift 2
        ;;
      --no-show)
        SHOW="false"
        shift
        ;;
      --help)
        echo "Usage: $0 --model MODEL_PATH --image IMAGE_PATH [options]"
        echo "Options:"
        echo "  --model MODEL_PATH     Path to the trained model (required)"
        echo "  --image IMAGE_PATH     Path to the image (required)"
        echo "  --conf FLOAT           Confidence threshold (default: 0.25)"
        echo "  --iou FLOAT            IoU threshold for NMS (default: 0.45)"
        echo "  --output-dir DIR       Directory to save results (default: $OUTPUT_DIR/single)"
        echo "  --device DEVICE        Device to use (cuda device or cpu)"
        echo "  --no-show              Don't display plots"
        exit 0
        ;;
      *)
        echo "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
    esac
  done
  
  # Check required arguments
  if [ -z "$IMAGE" ]; then
    # Try to find a sample image
    IMAGE=$(find_sample_image "data/samples" "data/samples/test_image.jpg")
    if [ $? -ne 0 ]; then
      print_error "No image specified and no sample image found"
      echo "Please specify an image with --image"
      exit 1
    fi
  fi
  
  # Run analysis
  analyze_single_image "$MODEL" "$IMAGE" "$CONF" "$IOU" "$OUTPUT" "$DEVICE" "$SHOW"
fi
