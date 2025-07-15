#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
一個提供圖片裁剪功能的工具腳本。
此版本將輸入/輸出路徑和尺寸直接寫在程式碼中。
"""

import os
from PIL import Image

def crop_image_from_center(input_path: str, output_path: str, target_width: int, target_height: int):
    """
    從圖片中心裁剪出指定寬度和高度的區域，並儲存為新圖片。
    (這個核心函式保持不變)
    """
    try:
        if not os.path.exists(input_path):
            print(f"錯誤：找不到輸入檔案 -> {input_path}")
            return False

        img = Image.open(input_path)
        original_width, original_height = img.size

        if target_width > original_width or target_height > original_height:
            print(f"錯誤：目標尺寸 ({target_width}x{target_height}) 不能大於原始圖片尺寸 ({original_width}x{original_height})。")
            return False

        left = (original_width - target_width) / 2
        top = (original_height - target_height) / 2
        right = (original_width + target_width) / 2
        bottom = (original_height + target_height) / 2
        
        crop_box = (left, top, right, bottom)
        cropped_img = img.crop(crop_box)
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cropped_img.save(output_path)
        print(f"圖片成功裁剪並儲存至 -> {output_path}")
        return True

    except Exception as e:
        print(f"處理圖片時發生錯誤: {e}")
        return False

def main():
    """
    主執行函式。
    這裡我們不再從命令列讀取參數，而是直接定義它們。
    """
    # --- 您可以在這裡修改預設路徑和尺寸 ---
    INPUT_IMAGE_PATH = "data/original/D/D/jpg/C23XXX_D31裝前煞車軟管_page_1.jpg"
    OUTPUT_IMAGE_PATH = "path/main_picture/D/D/"
    TARGET_WIDTH = 500
    TARGET_HEIGHT = 300
    # -----------------------------------------

    print(f"開始處理固定任務...")
    print(f"  - 輸入檔案: {INPUT_IMAGE_PATH}")
    print(f"  - 輸出檔案: {OUTPUT_IMAGE_PATH}")
    print(f"  - 目標尺寸: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    
    # 直接使用上面定義的變數來呼叫裁剪函式
    crop_image_from_center(
        input_path=INPUT_IMAGE_PATH,
        output_path=OUTPUT_IMAGE_PATH,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT
    )

if __name__ == "__main__":
    main()