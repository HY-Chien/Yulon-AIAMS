#!/bin/bash
# Script for analyzing a batch of images from a YOLO dataset

# Source common utilities
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/common.sh"

# Function to analyze a batch of images from a dataset
analyze_batch() {
  local model="$1"
  local data_yaml="$2"
  local num_samples="${3:-5}"
  local conf="${4:-0.25}"
  local iou="${5:-0.45}"
  local output_dir="${6:-$OUTPUT_DIR/batch}"
  local device="${7:-}"
  local show="${8:-true}"
  
  # Verify model file exists
  if ! check_file_exists "$model" "Model not found: $model"; then
    print_message "Please specify a valid model path with --model"
    return 1
  fi
  
  # Verify data YAML file exists
  if ! check_file_exists "$data_yaml" "Dataset YAML not found: $data_yaml"; then
    print_message "Please specify a valid dataset YAML path with --data"
    return 1
  fi
  
  # Set up show/no-show flag
  local show_flag=""
  if [ "$show" = "false" ]; then
    show_flag="--no-show"
  fi
  
  # Run the batch analysis
  print_message "Running batch analysis on dataset: $data_yaml"
  print_message "Using model: $model"
  print_message "Number of samples: $num_samples"
  print_message "Output directory: $output_dir"
  
  python -m tools.yolo.case_study --batch \
    --model "$model" \
    --data "$data_yaml" \
    --num-samples "$num_samples" \
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
  DATA_YAML=""
  NUM_SAMPLES=5
  CONF="0.25"
  IOU="0.45"
  OUTPUT="$OUTPUT_DIR/batch"
  DEVICE=""
  SHOW="true"
  
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)
        MODEL="$2"
        shift 2
        ;;
      --data)
        DATA_YAML="$2"
        shift 2
        ;;
      --num-samples)
        NUM_SAMPLES="$2"
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
        echo "Usage: $0 --model MODEL_PATH --data DATA_YAML [options]"
        echo "Options:"
        echo "  --model MODEL_PATH     Path to the trained model (required)"
        echo "  --data DATA_YAML       Path to the dataset YAML file (required)"
        echo "  --num-samples NUM      Number of random samples to analyze (default: 5)"
        echo "  --conf FLOAT           Confidence threshold (default: 0.25)"
        echo "  --iou FLOAT            IoU threshold for NMS (default: 0.45)"
        echo "  --output-dir DIR       Directory to save results (default: $OUTPUT_DIR/batch)"
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
  if [ -z "$DATA_YAML" ]; then
    # Try to find a data YAML file
    if [ -d "data" ]; then
      DATA_YAML=$(find data -name "*.yaml" | head -n 1)
      if [ -z "$DATA_YAML" ]; then
        print_error "No dataset YAML specified and none found in data directory"
        echo "Please specify a dataset YAML file with --data"
        exit 1
      else
        print_message "Found dataset YAML: $DATA_YAML"
      fi
    else
      print_error "No dataset YAML specified and data directory not found"
      echo "Please specify a dataset YAML file with --data"
      exit 1
    fi
  fi
  
  # Run batch analysis
  analyze_batch "$MODEL" "$DATA_YAML" "$NUM_SAMPLES" "$CONF" "$IOU" "$OUTPUT" "$DEVICE" "$SHOW"
fi
