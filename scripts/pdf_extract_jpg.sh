#!/bin/bash
# Example script for converting PDF files to JPG images

# Set path to the input PDF file
PDF_FILE="data/original/D/D/pdf/C23XXX_D31裝前煞車軟管.pdf"

# Set output directory 
OUTPUT_DIR="path/main_picture/D/D/"

# Run the conversion with higher DPI for better quality
python -m tools.converters.pdf_utils convert "$PDF_FILE" --output "$OUTPUT_DIR" --dpi 300

# To run batch conversion on a directory:
# PDF_DIR="path/to/pdf_directory"
# python -m tools.converters.pdf_utils batch-convert "$PDF_DIR" --output "$OUTPUT_DIR" --dpi 300 --workers 4
