import os
from datetime import datetime as dt

import openpyxl as pyxl
import pymupdf
from openpyxl.drawing.image import Image as PyxlImage


def to_int_list(data: str, delimeter: str = "/") -> list[int]:
    """Convert a string to a list of integers. <br>
    For example, "1/2/3" with delimeter '/' will be converted to [1, 2, 3].

    Args:
        data (str): string to convert

    Returns:
        tuple: List of integers
    """
    return list(map(int, data.split(delimeter)))


def extract_main_workspace_from_pdf(
    pdf_file_path: str, output_folder: str, pos: list[int], size=None
):
    """Extract main workspace using pymupdf. Note that the extracted image may lose some quality

    Args:
        pdf_file_path (str): Path to the PDF file
        output_folder (str): Directory to save images
        pos (list[int]): The area to extract. Left-top is identified as (0, 0). Format is [x0,y0,x1,y1]
        size (list[int], optional): Expected size for the output images (width, height). If not provided, the original size will be used.

    Returns:
        list: Paths to the created PNG images
    """
    if len(pos) != 4:
        print(
            "Error: pos of main workspace should be a list of 4 integers (x0, y0, x1, y1)"
        )
        return

    if (size is not None) and (len(size) != 2):
        print(
            "Error: expected size of main workspace should be a list of 2 integers (width, height)"
        )
        return

    # file_utils.ensure_directory_exists(output_folder)

    image_paths = []
    doc = pymupdf.open(pdf_file_path)

    for i, page in enumerate(doc):
        output_file_path = os.path.join(output_folder, "page_" + str(i) + ".png")
        rect = pymupdf.Rect(pos[0], pos[1], pos[2], pos[3])
        width, height = pos[2] - pos[0], pos[3] - pos[1]

        mult = 1
        if size is not None:
            ratio_x, ratio_y = size[0] / width, size[1] / height
            mult = min(ratio_x, ratio_y)
            # print(f"{size[0], size[1]}\n{width, height}\n{ratio_x, ratio_y}\n{mult}")

        mat = pymupdf.Matrix(mult, mult)
        pix = page.get_pixmap(matrix=mat, clip=rect)
        pix.save(output_file_path)
        pix = None
        image_paths.append(output_file_path)

    return image_paths


def extract_table_from_pdf(pdf_file_path: str, ams_info: str) -> dict[str, any]:
    """Extract table from whole pdf, save the result in "result.xlsx" under output_folder.

    Args:
        pdf_file_path (str): Path to the PDF file
        ams_info (dict[str, str]): AMS info from info.csv that specifies how to process the PDF
        language (str): language used for OCR, e.g. "chinese_cht", "japan". Check Paddle OCR for detail

    Returns:
        table (dict[str, Any]): A dictionary containing the extracted table data.
    """
    # file_utils.ensure_directory_exists(output_folder)

    headers = ["process_name_en_pos", "refer_graph_pos"]
    tables = []

    doc = pymupdf.open(pdf_file_path)
    for i, page in enumerate(doc):
        table = {"process_name_en": None, "process_name_ch": None}
        for header in headers:
            rect = to_int_list(ams_info[header])
            if len(rect) != 4:
                continue

            table[header[:-4]] = page.get_text(
                "text", clip=to_int_list(ams_info[header])
            ).strip()

        table["page"] = str(i + 1)
        table["last_page"] = str(len(doc))
        table["revise_records"] = [
            {"date": dt.today().strftime("%Y/%m/%d"), "record": "初訂"}
        ]

        # Translate logic
        table["process"] = {
            "id": ams_info["uid"],
            "name": {"ch": table["process_name_ch"], "en": table["process_name_en"]},
        }
        tables.append(table)

        """Debug code."""
        # print(str.join('\n', map(str, page.get_text("blocks"))))
        # for box in page.find_tables()[0].cells:
        # print(page.get_text("text", clip=box))

    return tables

    """This is using OCR"""
    # output_file_path = os.path.join(output_folder, 'result.xlsx')
    # ocr = PaddleOCR(lang=ams_info["lang"])
    # doc = PDF(pdf_file_path)
    # doc.to_xlsx(output_file_path, ocr)

    # return output_file_path


def write_xlsx(
    xlsx_file_path: str,
    text_dict: dict[str, str] = None,
    image_dict: dict[str, str] = None,
):
    """Write xlsx file. Write in first sheet. Overwrite.

    Args:
        xlsx_file_path (str): Path to the XLSX file
        text_dict (dict[str, str], optional): Dictionary containing text to write. The format is {cell: text}, e.g. {"C2": "text_to_write"}
        image_dict (dict[str, str], optional): Dictionary containing images to add. The format is {cell: image_path}, e.g. {"C2": "path_to_image.png"}
    """
    try:
        wb = pyxl.load_workbook(xlsx_file_path)
    except Exception as e:
        print(f"Error processing {xlsx_file_path}: {str(e)}")
        return

    ws = wb.active
    if text_dict is not None:
        for cell, text in text_dict.items():
            ws[cell] = text

    if image_dict is not None:
        for cell, image_path in image_dict.items():
            ws.add_image(PyxlImage(image_path), cell)

    wb.save(xlsx_file_path)
