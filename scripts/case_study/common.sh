#!/bin/bash
# Common utilities and variables for YOLO case study scripts

# Default path to the trained YOLO model
MODEL_PATH="runs/train/experiment_1/weights/best.pt"

# Set default output directory for case study results
OUTPUT_DIR="runs/detect/case_study"

# Function to print formatted messages
print_message() {
  echo -e "\033[1;34m[INFO]\033[0m $1"
}

# Function to print warnings
print_warning() {
  echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# Function to print errors
print_error() {
  echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# Function to check if a file exists, with error message if not
check_file_exists() {
  local file="$1"
  local message="$2"
  
  if [ ! -f "$file" ]; then
    print_error "$message"
    return 1
  fi
  return 0
}

# Function to find a sample image
find_sample_image() {
  local search_dir="$1"
  local default_path="$2"
  
  if [ -d "$search_dir" ]; then
    # Try to find a sample image
    SAMPLE_IMG=$(find "$search_dir" -type f -name "*.jpg" -o -name "*.png" | head -n 1)
    if [ -n "$SAMPLE_IMG" ]; then
      echo "$SAMPLE_IMG"
      return 0
    else
      print_warning "No sample images found in $search_dir"
      echo "$default_path"
      return 1
    fi
  else
    print_warning "Directory not found: $search_dir"
    echo "$default_path"
    return 1
  fi
}
