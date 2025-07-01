#!/usr/bin/env python3
"""
PDF processing utilities for converting and extracting content from PDF files.
"""

import multiprocessing
import os
import platform
import subprocess
from functools import partial

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

try:
    from nanoid import generate as nanoid_generate
except ImportError:
    # Simple fallback for nanoid if not installed
    import random
    import string

    def nanoid_generate(size=10):
        chars = string.ascii_letters + string.digits
        return "".join(random.choice(chars) for _ in range(size))


from tools.utils.file_utils import ensure_directory_exists


def pdf_to_jpg(pdf_file_path, output_folder=None, dpi=300):
    """Convert PDF file to JPG images using pdf2image.

    Args:
        pdf_file_path (str): Path to the PDF file
        output_folder (str, optional): Directory to save images
        dpi (int): DPI for the output images

    Returns:
        list: Paths to the created JPG images
    """
    if convert_from_path is None:
        raise ImportError(
            "pdf2image is required for this function. Install with: pip install pdf2image"
        )

    # Get the absolute paths
    pdf_file_path = os.path.abspath(pdf_file_path)
    output_folder = ensure_directory_exists(
        output_folder or os.path.dirname(pdf_file_path)
    )

    # Get the PDF filename without extension
    pdf_filename = os.path.splitext(os.path.basename(pdf_file_path))[0]

    # Check if poppler is installed on macOS
    if platform.system() == "Darwin":
        try:
            subprocess.run(["pdftoppm", "-v"], check=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("Poppler not found. Please install it using: brew install poppler")
            return []

    # Convert PDF to images with error handling
    print(f"Converting {pdf_file_path} to JPG images...")
    try:
        images = convert_from_path(pdf_file_path, dpi=dpi)
    except Exception as e:
        print(f"Error converting {pdf_file_path}: {str(e)}")
        return []

    # Save images
    image_paths = []
    for i, image in enumerate(images):
        # Create filename for the image
        image_path = os.path.join(output_folder, f"{pdf_filename}_page_{i + 1}.jpg")
        # Save the image
        image.save(image_path, "JPEG")
        image_paths.append(image_path)

    print(f"Converted {len(image_paths)} pages to JPG images")
    return image_paths


def extract_images_from_pdf(
    pdf_file,
    output_folder,
    min_width=0,
    min_height=0,
    max_width=float("inf"),
    max_height=float("inf"),
    verbose=True,
):
    """Extract images from a PDF file with optional size filtering.

    Args:
        pdf_file (str): Path to the PDF file
        output_folder (str): Directory to save extracted images
        min_width (int): Minimum width in pixels
        min_height (int): Minimum height in pixels
        max_width (int/float): Maximum width in pixels
        max_height (int/float): Maximum height in pixels
        verbose (bool): Whether to print progress messages

    Returns:
        tuple: (extracted_count, skipped_count, extracted_files)
    """
    if fitz is None:
        raise ImportError(
            "PyMuPDF is required for this function. Install with: pip install pymupdf"
        )

    # Ensure output directory exists
    output_folder = ensure_directory_exists(output_folder)

    # Open the PDF file
    doc = fitz.open(pdf_file)

    extracted_count = 0
    skipped_count = 0
    extracted_files = []

    for page_num in range(len(doc)):
        for img_index, img in enumerate(doc[page_num].get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)

            # Get image dimensions
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)

            # Check if image meets size criteria
            if (
                width >= min_width
                and height >= min_height
                and width <= max_width
                and height <= max_height
            ):
                img_bytes = base_image["image"]
                img_ext = base_image["ext"]

                # Generate a unique ID
                unique_id = nanoid_generate(size=10)

                img_filename = f"{output_folder}/{unique_id}_page_{page_num + 1}_img_{img_index + 1}.{img_ext}"

                with open(img_filename, "wb") as f:
                    f.write(img_bytes)

                extracted_files.append(img_filename)
                if verbose:
                    print(f"Extracted: {img_filename} (Size: {width}x{height})")
                extracted_count += 1
            else:
                if verbose:
                    print(
                        f"Skipped image on page {page_num + 1}, index {img_index + 1} (Size: {width}x{height})"
                    )
                skipped_count += 1

    if verbose:
        print(f"Images saved in: {output_folder}")
        print(
            f"Summary: {extracted_count} images extracted, {skipped_count} images skipped due to size filters"
        )

    return extracted_count, skipped_count, extracted_files


def batch_convert_pdf_to_jpg(pdf_folder, output_folder=None, dpi=300, workers=None):
    """Convert all PDF files in a folder to JPG images using multiprocessing.

    Args:
        pdf_folder (str): Directory containing PDF files
        output_folder (str, optional): Directory to save images
        dpi (int): DPI for the output images
        workers (int, optional): Number of worker processes

    Returns:
        list: Combined list of all image paths created
    """
    # Get the list of PDF files
    pdf_files = [
        os.path.join(pdf_folder, f)
        for f in os.listdir(pdf_folder)
        if os.path.isfile(os.path.join(pdf_folder, f)) and f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"No PDF files found in {pdf_folder}")
        return []

    # Process PDF files in parallel
    def process_pdf(pdf_path, out_folder, dpi=300):
        """Helper function to process a single PDF"""
        try:
            return pdf_to_jpg(pdf_path, out_folder, dpi)
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return []

    # Set number of workers
    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)
    workers = min(workers, len(pdf_files), multiprocessing.cpu_count())

    all_image_paths = []

    if len(pdf_files) > 1 and workers > 1:
        # Prepare arguments for multiprocessing
        args = []
        for pdf_file in pdf_files:
            args.append((pdf_file, output_folder, dpi))

        # Use multiprocessing for multiple files
        with multiprocessing.Pool(processes=workers) as pool:
            results = pool.starmap(process_pdf, args)

        # Collect all results
        for result in results:
            if result:
                all_image_paths.extend(result)
    else:
        # Process sequentially
        for pdf_file in pdf_files:
            result = process_pdf(pdf_file, output_folder, dpi)
            if result:
                all_image_paths.extend(result)

    return all_image_paths


def batch_extract_images_from_pdf(
    pdf_folder,
    output_folder,
    min_width=0,
    min_height=0,
    max_width=float("inf"),
    max_height=float("inf"),
    num_processes=None,
):
    """Extract images from all PDF files in a folder using multiprocessing.

    Args:
        pdf_folder (str): Directory containing PDF files
        output_folder (str): Directory to save extracted images
        min_width (int): Minimum width in pixels
        min_height (int): Minimum height in pixels
        max_width (int/float): Maximum width in pixels
        max_height (int/float): Maximum height in pixels
        num_processes (int): Number of processes to use

    Returns:
        tuple: (successful, failed, total_extracted, total_skipped)
    """
    # Ensure output directory exists
    output_folder = ensure_directory_exists(output_folder)

    # Get the list of PDF files
    pdf_files = [
        f
        for f in os.listdir(pdf_folder)
        if os.path.isfile(os.path.join(pdf_folder, f)) and f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"No PDF files found in {pdf_folder}")
        return 0, 0, 0, 0

    # Set number of processes
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    num_processes = min(num_processes, len(pdf_files), multiprocessing.cpu_count())

    print(
        f"Starting extraction with {num_processes} processes for {len(pdf_files)} PDF files"
    )

    # Create a helper function for processing a single PDF file
    def process_pdf_file_extract(
        pdf_file,
        pdf_folder,
        output_folder,
        min_width=0,
        min_height=0,
        max_width=float("inf"),
        max_height=float("inf"),
    ):
        """Process a single PDF file for image extraction"""
        try:
            pdf_file_path = os.path.join(pdf_folder, pdf_file)
            print(f"Processing: {pdf_file_path}")
            extracted_count, skipped_count, _ = extract_images_from_pdf(
                pdf_file_path,
                output_folder,
                min_width=min_width,
                min_height=min_height,
                max_width=max_width,
                max_height=max_height,
                verbose=False,  # Reduce output clutter in multiprocessing
            )
            print(
                f"Completed {pdf_file}: {extracted_count} extracted, {skipped_count} skipped"
            )
            return pdf_file, True, (extracted_count, skipped_count)
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            return pdf_file, False, (0, 0)

    # Create a partial function with fixed parameters
    process_func = partial(
        process_pdf_file_extract,
        pdf_folder=pdf_folder,
        output_folder=output_folder,
        min_width=min_width,
        min_height=min_height,
        max_width=max_width,
        max_height=max_height,
    )

    # Process files in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_func, pdf_files)

    # Report results
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    total_extracted = sum(
        extracted for _, success, (extracted, _) in results if success
    )
    total_skipped = sum(skipped for _, success, (_, skipped) in results if success)

    print("\nMultiprocessing summary:")
    print(f"- {successful} files processed successfully, {failed} files failed")
    print(
        f"- {total_extracted} images extracted, {total_skipped} images skipped due to size filters"
    )
    print(f"- All images saved in: {output_folder}")

    return successful, failed, total_extracted, total_skipped


def main():
    """Parse command line arguments and run PDF processing functions."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PDF conversion and extraction utilities"
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Convert PDF to JPG (single file)
    convert_parser = subparsers.add_parser("convert", help="Convert PDF to JPG images")
    convert_parser.add_argument("pdf_file", help="Path to the PDF file")
    convert_parser.add_argument(
        "--output", "-o", dest="output_folder", help="Output folder for JPG images"
    )
    convert_parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for JPG images (default: 300)"
    )

    # Batch convert PDFs to JPGs
    batch_convert_parser = subparsers.add_parser(
        "batch-convert", help="Convert multiple PDF files to JPG images"
    )
    batch_convert_parser.add_argument("pdf_folder", help="Folder containing PDF files")
    batch_convert_parser.add_argument(
        "--output", "-o", dest="output_folder", help="Output folder for JPG images"
    )
    batch_convert_parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for JPG images (default: 300)"
    )
    batch_convert_parser.add_argument(
        "--workers",
        "-w",
        type=int,
        help="Number of worker processes (default: CPU count - 1)",
    )

    # Extract images from PDF (single file)
    extract_parser = subparsers.add_parser("extract", help="Extract images from PDF")
    extract_parser.add_argument("pdf_file", help="Path to the PDF file")
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
        default=100000,
        help="Maximum width for images (default: no limit)",
    )
    extract_parser.add_argument(
        "--max-height",
        type=int,
        default=100000,
        help="Maximum height for images (default: no limit)",
    )
    extract_parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress messages"
    )

    # Batch extract images from PDFs
    batch_extract_parser = subparsers.add_parser(
        "batch-extract", help="Extract images from multiple PDF files"
    )
    batch_extract_parser.add_argument("pdf_folder", help="Folder containing PDF files")
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
        default=100000,
        help="Maximum width for images (default: no limit)",
    )
    batch_extract_parser.add_argument(
        "--max-height",
        type=int,
        default=100000,
        help="Maximum height for images (default: no limit)",
    )
    batch_extract_parser.add_argument(
        "--workers",
        "-w",
        type=int,
        help="Number of worker processes (default: CPU count)",
    )

    args = parser.parse_args()

    # Execute the appropriate command
    if args.command == "convert":
        image_paths = pdf_to_jpg(args.pdf_file, args.output_folder, args.dpi)
        print(f"Created {len(image_paths)} JPG images from {args.pdf_file}")

    elif args.command == "batch-convert":
        image_paths = batch_convert_pdf_to_jpg(
            args.pdf_folder, args.output_folder, args.dpi, args.workers
        )
        print(f"Total JPG images created: {len(image_paths)}")

    elif args.command == "extract":
        extracted_count, skipped_count, _ = extract_images_from_pdf(
            args.pdf_file,
            args.output_folder,
            min_width=args.min_width,
            min_height=args.min_height,
            max_width=args.max_width,
            max_height=args.max_height,
            verbose=not args.quiet,
        )
        print(f"Extracted {extracted_count} images, skipped {skipped_count} images")

    elif args.command == "batch-extract":
        successful, failed, total_extracted, total_skipped = (
            batch_extract_images_from_pdf(
                args.pdf_folder,
                args.output_folder,
                min_width=args.min_width,
                min_height=args.min_height,
                max_width=args.max_width,
                max_height=args.max_height,
                num_processes=args.workers,
            )
        )
        print(f"Processed {successful} files, failed {failed} files")
        print(f"Total images extracted: {total_extracted}, skipped: {total_skipped}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
