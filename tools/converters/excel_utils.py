#!/usr/bin/env python3
"""
Excel processing utilities for converting and extracting content from Excel files.
"""

import glob
import io
import multiprocessing
import os
import shutil
import subprocess
import uuid
import zipfile
from pathlib import Path

from PIL import Image

from tools.converters.pdf_utils import pdf_to_jpg
from tools.utils.file_utils import ensure_directory_exists, find_executable


def excel_to_pdf_libreoffice(excel_file_path, output_folder=None):
    """Convert Excel file to PDF using LibreOffice.

    Args:
        excel_file_path (str): Path to the Excel file
        output_folder (str, optional): Directory to save the PDF file

    Returns:
        str: Path to the created PDF file, or None if conversion failed
    """
    # Get the absolute paths
    excel_file_path = os.path.abspath(excel_file_path)
    output_folder = ensure_directory_exists(
        output_folder or os.path.dirname(excel_file_path)
    )

    # On macOS, check common locations for the LibreOffice executable
    libreoffice_path = find_executable(
        "soffice",
        [
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
            "/Applications/LibreOffice.app/Contents/MacOS/soffice.bin",
        ],
    )

    # Build the command
    cmd = [
        libreoffice_path,
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        output_folder,
        excel_file_path,
    ]

    # Execute the command
    print(f"Running LibreOffice command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error converting {excel_file_path} to PDF")
        print(f"Error: {stderr.decode('utf-8')}")
        return None

    # Return the path to the created PDF
    pdf_name = os.path.splitext(os.path.basename(excel_file_path))[0] + ".pdf"
    pdf_path = os.path.join(output_folder, pdf_name)

    return pdf_path


def batch_convert_excel_to_pdf(excel_folder, output_folder=None):
    """Convert all Excel files in a folder to PDF using LibreOffice.

    Args:
        excel_folder (str): Directory containing Excel files
        output_folder (str, optional): Directory to save PDF files

    Returns:
        list: Paths to the created PDF files
    """
    # Ensure output directory exists
    output_folder = ensure_directory_exists(output_folder or excel_folder)

    # Get the list of Excel files
    excel_files = [
        f
        for f in os.listdir(excel_folder)
        if os.path.isfile(os.path.join(excel_folder, f))
        and f.lower().endswith((".xls", ".xlsx"))
    ]

    # Convert each Excel file to PDF
    pdfs = []
    for excel_file in excel_files:
        excel_file_path = os.path.join(excel_folder, excel_file)
        print(f"Converting: {excel_file}")
        pdf_path = excel_to_pdf_libreoffice(excel_file_path, output_folder)
        if pdf_path is not None:
            pdfs.append(pdf_path)

    return pdfs


def excel_to_jpg(excel_file_path, output_folder=None, dpi=300, keep_pdf=False):
    """Convert Excel file to JPG images by first converting to PDF and then to JPG.

    Args:
        excel_file_path (str): Path to the Excel file
        output_folder (str, optional): Directory to save the JPG images
        dpi (int): DPI for the output images
        keep_pdf (bool): Whether to keep the intermediate PDF file

    Returns:
        list: Paths to the created JPG images
    """
    # Get the absolute paths
    excel_file_path = os.path.abspath(excel_file_path)
    output_folder = ensure_directory_exists(
        output_folder or os.path.dirname(excel_file_path)
    )

    # Create a unique temporary folder for PDF if needed
    if not keep_pdf:
        # Use a unique identifier to avoid conflicts in parallel processing
        unique_id = str(uuid.uuid4())[:8]
        excel_basename = os.path.splitext(os.path.basename(excel_file_path))[0]
        temp_pdf_folder = os.path.join(
            output_folder, f"temp_pdf_{excel_basename}_{unique_id}"
        )
        os.makedirs(temp_pdf_folder, exist_ok=True)
    else:
        temp_pdf_folder = output_folder

    try:
        # Step 1: Convert Excel to PDF
        pdf_path = excel_to_pdf_libreoffice(excel_file_path, temp_pdf_folder)
        if pdf_path is None:
            print(f"Failed to convert Excel file to PDF: {excel_file_path}")
            return []

        # Step 2: Convert PDF to JPG
        image_paths = pdf_to_jpg(pdf_path, output_folder, dpi)

        return image_paths
    finally:
        # Clean up temporary PDF and folder if not keeping it
        if not keep_pdf and os.path.exists(temp_pdf_folder):
            try:
                # Remove the entire temporary directory and its contents
                shutil.rmtree(temp_pdf_folder, ignore_errors=True)
            except Exception as e:
                print(
                    f"Warning: Could not remove temporary directory {temp_pdf_folder}: {str(e)}"
                )


def batch_convert_excel_to_jpg(
    excel_folder, output_folder=None, dpi=300, keep_pdf=False, workers=None
):
    """Convert all Excel files in a folder to JPG images using multiprocessing.

    Args:
        excel_folder (str): Directory containing Excel files
        output_folder (str, optional): Directory to save JPG images
        dpi (int): DPI for the output images
        keep_pdf (bool): Whether to keep the intermediate PDF files
        workers (int, optional): Number of worker processes

    Returns:
        list: Paths to the created JPG images
    """
    # Get the list of Excel files
    excel_files = [
        f
        for f in os.listdir(excel_folder)
        if os.path.isfile(os.path.join(excel_folder, f))
        and f.lower().endswith((".xls", ".xlsx"))
    ]

    # Set the number of workers (processes)
    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)  # Use all CPUs except one

    # Use multiprocessing to convert Excel files in parallel
    all_image_paths = []
    if len(excel_files) > 0:
        print(f"Processing {len(excel_files)} Excel files using {workers} workers...")

        # Create a process pool and distribute the work
        excel_paths = [
            os.path.join(excel_folder, excel_file) for excel_file in excel_files
        ]

        def process_excel(excel_path, output_folder=None, dpi=300, keep_pdf=False):
            """Process a single Excel file and convert it to JPG images."""
            try:
                return excel_to_jpg(excel_path, output_folder, dpi, keep_pdf)
            except Exception as e:
                print(f"Error processing {excel_path}: {str(e)}")
                return []

        if len(excel_paths) > 1 and workers > 1:
            # Use multiprocessing for multiple Excel files
            with multiprocessing.Pool(processes=workers) as pool:
                # Use starmap to pass multiple arguments to the process_excel function
                results = pool.starmap(
                    process_excel,
                    [(path, output_folder, dpi, keep_pdf) for path in excel_paths],
                )

            # Collect all image paths from the results
            for result in results:
                if result:
                    all_image_paths.extend(result)
        else:
            # Process sequentially for a single Excel file or if only one worker
            for excel_path in excel_paths:
                result = process_excel(excel_path, output_folder, dpi, keep_pdf)
                if result:
                    all_image_paths.extend(result)

    return all_image_paths


def extract_images_from_excel(
    excel_file,
    output_folder,
    min_width=0,
    min_height=0,
    max_width=float("inf"),
    max_height=float("inf"),
    prefix_filename=False,
):
    """Extract images from an Excel file with optional size filtering.

    Args:
        excel_file (str): Path to the Excel file
        output_folder (str): Directory to save extracted images
        min_width (int): Minimum width in pixels (default: 0)
        min_height (int): Minimum height in pixels (default: 0)
        max_width (int/float): Maximum width in pixels (default: infinity)
        max_height (int/float): Maximum height in pixels (default: infinity)
        prefix_filename (bool): Whether to prefix image filenames with Excel filename

    Returns:
        tuple: (extracted_count, skipped_count, extracted_files)
    """
    # Ensure output directory exists
    output_folder = ensure_directory_exists(output_folder)

    # Get Excel filename for potential prefixing
    excel_basename = Path(excel_file).stem

    extracted_count = 0
    skipped_count = 0
    extracted_files = []

    # Open the Excel file as a zip archive
    with zipfile.ZipFile(excel_file, "r") as zip_ref:
        # Process only image files from "xl/media/"
        for file in zip_ref.namelist():
            if file.startswith("xl/media/"):
                # Read the image data
                image_data = zip_ref.read(file)

                try:
                    # Open the image to get its dimensions
                    img = Image.open(io.BytesIO(image_data))
                    width, height = img.size

                    # Check if image meets size criteria
                    if (
                        width >= min_width
                        and height >= min_height
                        and width <= max_width
                        and height <= max_height
                    ):
                        image_filename = os.path.basename(file)

                        if prefix_filename:
                            # Save with Excel filename as prefix
                            output_path = os.path.join(
                                output_folder, f"{excel_basename}_{image_filename}"
                            )
                        else:
                            # Save without prefix
                            output_path = os.path.join(output_folder, image_filename)

                        # Save the image
                        with open(output_path, "wb") as f:
                            f.write(image_data)

                        extracted_files.append(output_path)
                        print(f"Extracted: {file} (Size: {width}x{height})")
                        extracted_count += 1
                    else:
                        print(f"Skipped: {file} (Size: {width}x{height})")
                        skipped_count += 1

                except Exception as e:
                    # If we can't determine the image size, extract it anyway
                    print(
                        f"Warning: Could not determine size of {file}, extracting anyway. Error: {e}"
                    )

                    image_filename = os.path.basename(file)
                    if prefix_filename:
                        output_path = os.path.join(
                            output_folder, f"{excel_basename}_{image_filename}"
                        )
                    else:
                        output_path = os.path.join(output_folder, image_filename)

                    with open(output_path, "wb") as f:
                        f.write(image_data)

                    extracted_files.append(output_path)
                    extracted_count += 1

    print(f"Images saved in: {output_folder}")
    print(
        f"Summary: {extracted_count} images extracted, {skipped_count} images skipped due to size filters"
    )

    return extracted_count, skipped_count, extracted_files


def batch_extract_images_from_excel(
    excel_folder,
    output_folder,
    min_width=0,
    min_height=0,
    max_width=float("inf"),
    max_height=float("inf"),
    prefix_filename=True,
    num_processes=None,
):
    """Extract images from all Excel files in a folder with optional size filtering.
    Uses multiprocessing to process files in parallel.

    Args:
        excel_folder (str): Path to the folder containing Excel files
        output_folder (str): Directory to save extracted images
        min_width (int): Minimum width in pixels
        min_height (int): Minimum height in pixels
        max_width (int/float): Maximum width in pixels
        max_height (int/float): Maximum height in pixels
        prefix_filename (bool): Whether to prefix image filenames with Excel filename
        num_processes (int): Number of processes to use

    Returns:
        tuple: (processed_files, failed_files)
    """
    # Ensure output directory exists
    output_folder = ensure_directory_exists(output_folder)

    # Get the list of Excel files
    excel_files = []
    for ext in [".xlsx", ".xls"]:
        excel_files.extend(glob.glob(os.path.join(excel_folder, f"*{ext}")))

    if not excel_files:
        print(f"No Excel files found in {excel_folder}")
        return 0, 0

    # Determine number of processes to use
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    # Limit processes to the number of files or CPU cores, whichever is smaller
    num_processes = min(num_processes, len(excel_files), multiprocessing.cpu_count())

    print(f"Starting parallel processing with {num_processes} processes")
    print(f"Found {len(excel_files)} Excel files to process")

    def process_excel_file(
        excel_file,
        output_folder,
        min_width,
        min_height,
        max_width,
        max_height,
        prefix_filename,
    ):
        """Wrapper function for parallel processing of Excel files."""
        try:
            extract_images_from_excel(
                excel_file,
                output_folder,
                min_width=min_width,
                min_height=min_height,
                max_width=max_width,
                max_height=max_height,
                prefix_filename=prefix_filename,
            )
            return excel_file, True, None
        except Exception as e:
            return excel_file, False, str(e)

    # Prepare arguments for each worker
    args = []
    for excel_file in excel_files:
        args.append(
            (
                excel_file,
                output_folder,
                min_width,
                min_height,
                max_width,
                max_height,
                prefix_filename,
            )
        )

    # Process Excel files in parallel
    processed_files = 0
    failed_files = 0

    # Use multiprocessing.Pool for better process management
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map the function to the arguments
        results = pool.starmap(process_excel_file, args)

        # Process results
        for file_path, success, error in results:
            if success:
                print(f"\nSuccessfully processed: {os.path.basename(file_path)}")
                processed_files += 1
            else:
                print(f"\nError processing {os.path.basename(file_path)}: {error}")
                failed_files += 1

    print("\nBatch processing complete:")
    print(f"Successfully processed {processed_files} Excel files")
    if failed_files > 0:
        print(f"Failed to process {failed_files} Excel files")

    return processed_files, failed_files


def main():
    """Parse command line arguments and run Excel processing functions."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Excel conversion and extraction utilities"
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Convert Excel to PDF (single file)
    convert_pdf_parser = subparsers.add_parser("to-pdf", help="Convert Excel to PDF")
    convert_pdf_parser.add_argument("excel_file", help="Path to the Excel file")
    convert_pdf_parser.add_argument(
        "--output", "-o", dest="output_folder", help="Output folder for PDF file"
    )

    # Batch convert Excel to PDF
    batch_convert_pdf_parser = subparsers.add_parser(
        "batch-to-pdf", help="Convert multiple Excel files to PDF"
    )
    batch_convert_pdf_parser.add_argument(
        "excel_folder", help="Folder containing Excel files"
    )
    batch_convert_pdf_parser.add_argument(
        "--output", "-o", dest="output_folder", help="Output folder for PDF files"
    )

    # Convert Excel to JPG (single file)
    convert_jpg_parser = subparsers.add_parser(
        "to-jpg", help="Convert Excel to JPG images"
    )
    convert_jpg_parser.add_argument("excel_file", help="Path to the Excel file")
    convert_jpg_parser.add_argument(
        "--output", "-o", dest="output_folder", help="Output folder for JPG images"
    )
    convert_jpg_parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for JPG images (default: 300)"
    )
    convert_jpg_parser.add_argument(
        "--keep-pdf", "-k", action="store_true", help="Keep intermediate PDF files"
    )

    # Batch convert Excel to JPG
    batch_convert_jpg_parser = subparsers.add_parser(
        "batch-to-jpg", help="Convert multiple Excel files to JPG images"
    )
    batch_convert_jpg_parser.add_argument(
        "excel_folder", help="Folder containing Excel files"
    )
    batch_convert_jpg_parser.add_argument(
        "--output", "-o", dest="output_folder", help="Output folder for JPG images"
    )
    batch_convert_jpg_parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for JPG images (default: 300)"
    )
    batch_convert_jpg_parser.add_argument(
        "--keep-pdf", "-k", action="store_true", help="Keep intermediate PDF files"
    )
    batch_convert_jpg_parser.add_argument(
        "--workers",
        "-w",
        type=int,
        help="Number of worker processes (default: CPU count - 1)",
    )

    # Extract images from Excel (single file)
    extract_parser = subparsers.add_parser("extract", help="Extract images from Excel")
    extract_parser.add_argument("excel_file", help="Path to the Excel file")
    extract_parser.add_argument(
        "--output",
        "-o",
        dest="output_folder",
        required=True,
        help="Output folder for extracted images",
    )
    extract_parser.add_argument(
        "--min-width", type=int, default=0, help="Minimum width for images (default: 0)"
    )
    extract_parser.add_argument(
        "--min-height",
        type=int,
        default=0,
        help="Minimum height for images (default: 0)",
    )
    extract_parser.add_argument(
        "--max-width",
        type=int,
        default=0,
        help="Maximum width for images (default: no limit, use 0 for no limit)",
    )
    extract_parser.add_argument(
        "--max-height",
        type=int,
        default=0,
        help="Maximum height for images (default: no limit, use 0 for no limit)",
    )
    extract_parser.add_argument(
        "--prefix",
        "-p",
        action="store_true",
        dest="prefix_filename",
        help="Prefix image filenames with Excel filename",
    )

    # Batch extract images from Excel
    batch_extract_parser = subparsers.add_parser(
        "batch-extract", help="Extract images from multiple Excel files"
    )
    batch_extract_parser.add_argument(
        "excel_folder", help="Folder containing Excel files"
    )
    batch_extract_parser.add_argument(
        "--output",
        "-o",
        dest="output_folder",
        required=True,
        help="Output folder for extracted images",
    )
    batch_extract_parser.add_argument(
        "--min-width", type=int, default=0, help="Minimum width for images (default: 0)"
    )
    batch_extract_parser.add_argument(
        "--min-height",
        type=int,
        default=0,
        help="Minimum height for images (default: 0)",
    )
    batch_extract_parser.add_argument(
        "--max-width",
        type=int,
        default=0,
        help="Maximum width for images (default: no limit, use 0 for no limit)",
    )
    batch_extract_parser.add_argument(
        "--max-height",
        type=int,
        default=0,
        help="Maximum height for images (default: no limit, use 0 for no limit)",
    )
    batch_extract_parser.add_argument(
        "--prefix",
        "-p",
        action="store_true",
        dest="prefix_filename",
        help="Prefix image filenames with Excel filename",
    )
    batch_extract_parser.add_argument(
        "--workers",
        "-w",
        type=int,
        help="Number of worker processes (default: CPU count)",
    )

    args = parser.parse_args()

    # Handle max width/height of 0 (no limit)
    max_width = (
        float("inf")
        if hasattr(args, "max_width") and args.max_width == 0
        else getattr(args, "max_width", float("inf"))
    )
    max_height = (
        float("inf")
        if hasattr(args, "max_height") and args.max_height == 0
        else getattr(args, "max_height", float("inf"))
    )

    # Execute the appropriate command
    if args.command == "to-pdf":
        pdf_path = excel_to_pdf_libreoffice(args.excel_file, args.output_folder)
        if pdf_path:
            print(f"Successfully converted to PDF: {pdf_path}")
        else:
            print(f"Failed to convert {args.excel_file} to PDF")
            return 1

    elif args.command == "batch-to-pdf":
        pdfs = batch_convert_excel_to_pdf(args.excel_folder, args.output_folder)
        print(f"Converted {len(pdfs)} Excel files to PDF")

    elif args.command == "to-jpg":
        image_paths = excel_to_jpg(
            args.excel_file, args.output_folder, args.dpi, args.keep_pdf
        )
        print(f"Created {len(image_paths)} JPG images from {args.excel_file}")

    elif args.command == "batch-to-jpg":
        image_paths = batch_convert_excel_to_jpg(
            args.excel_folder, args.output_folder, args.dpi, args.keep_pdf, args.workers
        )
        print(f"Total JPG images created: {len(image_paths)}")

    elif args.command == "extract":
        extracted_count, skipped_count, _ = extract_images_from_excel(
            args.excel_file,
            args.output_folder,
            min_width=args.min_width,
            min_height=args.min_height,
            max_width=max_width,
            max_height=max_height,
            prefix_filename=args.prefix_filename,
        )
        print(f"Extracted {extracted_count} images, skipped {skipped_count} images")

    elif args.command == "batch-extract":
        processed, failed = batch_extract_images_from_excel(
            args.excel_folder,
            args.output_folder,
            min_width=args.min_width,
            min_height=args.min_height,
            max_width=max_width,
            max_height=max_height,
            prefix_filename=args.prefix_filename,
            num_processes=args.workers,
        )
        print(f"Processed {processed} Excel files, failed {failed} files")

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
