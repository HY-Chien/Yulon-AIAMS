#!/bin/bash
# Example script for converting Excel files to PDF

# Set path to input Excel file
EXCEL_FILE="./data/original/D/D/excel/C23XXX_D31裝前煞車軟管.xls"

# Set output directory
OUTPUT_DIR="path/to/output"

# Convert a single Excel file to PDF
python -m tools.converters.excel_utils to-pdf "$EXCEL_FILE" --output "$OUTPUT_DIR"

# To convert all Excel files in a directory:
# EXCEL_DIR="path/to/excel_directory"
# python -m tools.converters.excel_utils batch-to-pdf "$EXCEL_DIR" --output "$OUTPUT_DIR"
