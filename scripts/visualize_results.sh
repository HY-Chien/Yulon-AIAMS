#!/bin/bash
# Example script for visualizing and analyzing YOLO training results

# Set path to the CSV results file (generated during YOLO training)
RESULTS_CSV="runs/train/experiment_1/results.csv"

# Set output directory for plots
OUTPUT_DIR="runs/train/experiment_1/plots"

# Generate training result plots
python -m tools.visualization.plot_results "$RESULTS_CSV" --output-dir "$OUTPUT_DIR"

# Convert results to TensorBoard format (alternative visualization)
# python -m tools.visualization.csv_to_tensorboard "$RESULTS_CSV" --log_dir "$OUTPUT_DIR/tensorboard"

# Open TensorBoard to view results (if installed)
# tensorboard --logdir="$OUTPUT_DIR/tensorboard"

# To run case study on images using your trained model:
# MODEL_PATH="runs/train/experiment_1/weights/best.pt"
# TEST_IMAGE="path/to/test/image.jpg"
# python -m tools.yolo.case_study --single --model "$MODEL_PATH" --image "$TEST_IMAGE"
