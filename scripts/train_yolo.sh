#!/bin/bash
################################################################################
# Default values
MODEL_DIR="./yolo_models"
AUGMENTATION="medium"
DATA_PATH="./data/synthetic/icon=Cv4R_50k-${AUGMENTATION}-is/data.yaml"
MODEL_SIZE=x
BATCH_SIZE=8
IMAGE_SIZE=769
DEVICE=0
PROJECT="./runs/train"
EPOCHS=60
#修改這部分看路徑會不會正常
#NAME="Cv4R/icons=Cv4R_50k-${AUGMENTATION}-is(8,2)_epoch=${EPOCHS}_yolo12${MODEL_SIZE}-is${IMAGE_SIZE}"
NAME="Cv4R_50k-${AUGMENTATION}_yolo12${MODEL_SIZE}"
# EXPORT_FORMAT="onnx"
# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Run the training script with the specified model directory
python tools/yolo/train.py \
  --model "$MODEL_DIR/yolo12$MODEL_SIZE.pt" \
  --data "$DATA_PATH" \
  --model-size "$MODEL_SIZE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --imgsz "$IMAGE_SIZE" \
  --device "$DEVICE" \
  --project "$PROJECT" \
  --name "$NAME" \
  --model-dir "$MODEL_DIR" \
  $PRETRAINED \
  $EXPORT


# Default values
EPOCHS=80
NAME="Cv4R/icons=Cv4R_50k-${AUGMENTATION}(8,2)_epoch=${EPOCHS}_yolo12${MODEL_SIZE}-is${IMAGE_SIZE}"
# EXPORT_FORMAT="onnx"
# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Run the training script with the specified model directory
python tools/yolo/train.py \
  --model "$MODEL_DIR/yolo12$MODEL_SIZE.pt" \
  --data "$DATA_PATH" \
  --model-size "$MODEL_SIZE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --imgsz "$IMAGE_SIZE" \
  --device "$DEVICE" \
  --project "$PROJECT" \
  --name "$NAME" \
  --model-dir "$MODEL_DIR" \
  $PRETRAINED \
  $EXPORT