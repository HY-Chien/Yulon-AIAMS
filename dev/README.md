# AIAMS Dev Package

A Python package for converting other types of Assembly Manual Sheet (AMS) to Yulon AMS.

## Overview

This package provides utilities for:
- Converting other AMS to Yulon AMS
- Extracting main workspace image
- Extracting table from AMS
- Writing data into excel (.xlsx) file

## Package Structure

```
dev/
├── __init__.py
├── convert_ams.py         # AMS Converter (Not done yet)
├── data_utils.py          # Utility functions
├── yulon_template.py      # Yulon template info
└── data/                  # Data
    ├── info.csv           # Containing brand-dependent AMS info
    └── template.xlsx      # Yulon AMS template
```

## Installation and Prerequisites

### Dependencies

- Python 3.7+
- pandas for CSV processing
- PyMuPDF for PDF processing
- openpyxl for XLSX processing
- img2table for OCR
- img2table[paddle] for PaddleOCR

## Documentation

### convert_ams.py

**Python API usage:**
```python
from dev.convert_ams import AMSConverter

converter = AMSConverter()
converter.convert_ams(
    uid="uid"
    input_path="path/to/input/file.pdf"
    output_path="path/to/output/file.xlsx"
    intermediate_folder="path/to/tmp/folder"
)
```

### data_utils.py

**Python API usage:**
```python
from dev.data_utils import extract_main_workspace_from_pdf, extract_table_from_pdf, write_xlsx

# Extract main workspace from pdf.
# If size is specified, the size of the output image will be multiplied by a smaller factor that can reach either expected_width or expected_height.
extract_main_workspace_from_pdf(
    pdf_file_path="path/to/file.pdf"
    output_folder="path/to/output/folder"
    pos=[x0, y0, x1, y1]
    size=[expected_width, expected_height]
)

# Extract table data from pdf. Needs AMS info to specify how to extract table. 
# See `data/info.csv` for AMS info details.
extract_table_from_pdf(
    pdf_file_path="path/to/file.pdf"
    ams_info={"table_data_name": "extract_info", ... }
)

# Write data to xlsx file. (NOT CREATE)
write_xlsx(
    xlsx_file_path="path/to/file.xlsx"
    text_dict={"cell_pos": "data_to_write", ... }
    image_dict ={"cell_pos": "path_to_image.jpg", ... }
)
```