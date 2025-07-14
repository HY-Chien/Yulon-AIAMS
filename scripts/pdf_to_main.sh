#!/bin/bash

# --- 1. 設定要處理的單一 PDF 檔案路徑 ---
PDF_FILE="data/original/D/D/pdf/C23XXX_D31裝前煞車軟管.pdf"

# --- 2. 設定圖片輸出的目標資料夾 ---
# 為了清楚起見，我們為單檔處理指定一個新的輸出資料夾
OUTPUT_DIR="path/main_picture/D/D/single_output"

# --- 3. 建立輸出資料夾 ---
# -p 參數可以確保在資料夾已存在時不會顯示錯誤
mkdir -p "$OUTPUT_DIR"

echo "準備處理單一檔案..."
echo "來源檔案: $PDF_FILE"
echo "輸出目錄: $OUTPUT_DIR"

# --- 4. 執行單檔提取指令 (extract) ---
# - 同樣需要設定 PYTHONPATH 才能找到 lib/ 中的套件
# - 呼叫 extract 指令
# - 傳入檔案路徑、輸出路徑和尺寸過濾參數
PYTHONPATH=/workspace/aiams/lib python -m tools.converters.pdf_utils extract "$PDF_FILE" \
    --output "$OUTPUT_DIR" \
    --min-width 100 \
    --min-height 100

# 檢查上一個指令的執行結果
if [ $? -eq 0 ]; then
    echo "處理成功！"
else
    echo "處理失敗，請檢查上面的錯誤訊息。"
fi