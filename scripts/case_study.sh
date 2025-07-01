#!/bin/bash
# YOLO Case Study Tool
# This script is a wrapper for the modular case study scripts
# It forwards all arguments to the main script

# Get the directory of this script
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Path to the modular case study scripts
MODULES_DIR="$SCRIPT_DIR/case_study"

# Check if modules directory exists
if [ ! -d "$MODULES_DIR" ]; then
  echo "Error: Case study modules directory not found at $MODULES_DIR"
  exit 1
fi

# Check if the main script exists
if [ ! -f "$MODULES_DIR/main.sh" ]; then
  echo "Error: Main case study script not found at $MODULES_DIR/main.sh"
  exit 1
fi

# Forward all arguments to the main script
"$MODULES_DIR/main.sh" "$@"
exit $?
