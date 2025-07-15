#!/bin/bash
set -e # 如果任何指令執行失敗，立即中斷腳本

################################################################################
# 設定區 (Configuration Section)
################################################################################

# --- 步驟 1：請填寫你要驗證的模型權重路徑 (最重要！) ---
# 從 `runs/train/` 目錄下找到你訓練完成的實驗資料夾，並指向 best.pt
# 範例： MODEL_PATH="runs/train/Cv4R_50k-medium_e60_yolo12x2/weights/best.pt"
MODEL_PATH="runs/train/Cv4R/icons=Cv4R_50k-medium-is(8,2)_epoch=60_yolo12x-is7697/weights/best.pt"


# --- 步驟 2：確認驗證所需的其他參數 (通常與訓練時相同) ---
DATA_PATH="./data/synthetic/icon=Cv4R_50k-medium-is/data.yaml"
IMAGE_SIZE=769
BATCH_SIZE=16
DEVICE=0


################################################################################
# 執行區 (Execution Section)
################################################################################

echo "--- Starting Validation Only ---"
echo "  - Model: ${MODEL_PATH}"
echo "  - Dataset: ${DATA_PATH}"
echo "--------------------------------"

# 檢查模型檔案是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model file not found at '${MODEL_PATH}'"
    echo "👉 Please update the MODEL_PATH variable in the script."
    exit 1
fi

# 使用 python -c 執行一小段 Python 程式碼，直接呼叫 validate 方法
python -c "
import sys
import os
from pathlib import Path

# 假設此腳本是從 /workspace/aiams 根目錄執行
# 將 tools/yolo 目錄加入 Python 的搜尋路徑，這樣才能 import train
sys.path.append('tools/yolo')

try:
    from train import YOLOTrainer
except ImportError:
    print('❌ Error: Could not import YOLOTrainer from tools/yolo/train.py')
    print('👉 Please ensure you are running this script from the project root directory.')
    sys.exit(1)

# 建立 YOLOTrainer 的實例
trainer = YOLOTrainer()

print('--- Calling YOLOTrainer.validate() ---')

# 直接呼叫 validate 方法，並傳入 Shell 腳本中定義的變數
metrics = trainer.validate(
    model_path='${MODEL_PATH}',
    data_yaml='${DATA_PATH}',
    imgsz=${IMAGE_SIZE},
    batch_size=${BATCH_SIZE},
    device='${DEVICE}'
)

print('\n✅ Validation Complete. Metrics:')
if hasattr(metrics, 'box'):
    print(f'  - Box mAP50-95: {metrics.box.map:.4f}')
    print(f'  - Box mAP50:    {metrics.box.map50:.4f}')
    print(f'  - Box mAP75:    {metrics.box.map75:.4f}')

if hasattr(metrics, 'speed'):
    print('\n  - Speed Metrics:')
    for k, v in metrics.speed.items():
        print(f'    - {k}: {v:.2f}ms')

print(f'\nResults saved to Ultralytics default validation directory.')
"

echo "✅ Script finished."