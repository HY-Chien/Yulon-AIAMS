#!/usr/bin/env python3
"""
CSV to TensorBoard converter

This module provides functionality to convert training metrics from CSV files
to TensorBoard event files for visualization.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from tools.utils.file_utils import ensure_directory_exists


def csv_to_tensorboard(csv_path, log_dir=None, auto_group=True):
    """Convert CSV file with training metrics to TensorBoard event file.

    Args:
        csv_path (str): Path to the CSV file containing training metrics
        log_dir (str, optional): Directory to save TensorBoard event files.
                                If None, will use the same directory as the CSV file.
        auto_group (bool): Whether to automatically group metrics in TensorBoard
                          by detecting common prefixes in column names.

    Returns:
        str: Path to the TensorBoard log directory
    """
    # Validate CSV file exists
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Create log directory if not provided
    if log_dir is None:
        log_dir = os.path.dirname(os.path.abspath(csv_path))

    log_dir = ensure_directory_exists(log_dir)

    # Read CSV file
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)

    # Get filename without extension for run name
    csv_name = Path(csv_path).stem

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir, comment=f"_{csv_name}")

    # Get column names excluding 'epoch' and 'time'
    metric_columns = [col for col in df.columns if col not in ["epoch", "time"]]

    # Add each metric to TensorBoard
    for _, row in df.iterrows():
        # Use 'epoch' column as step if available, otherwise use row index
        epoch = int(row["epoch"]) if "epoch" in df.columns else _

        # Add each metric as a scalar
        for metric in metric_columns:
            # Skip any NaN values
            if pd.notna(row[metric]):
                if auto_group:
                    # Replace any slashes with underscores to avoid TensorBoard issues
                    tag = metric.replace("/", "_")
                    # Try to extract group name (everything before first '/')
                    parts = metric.split("/")
                    if len(parts) > 1:
                        tag = "/".join([parts[0], tag])
                else:
                    tag = metric

                writer.add_scalar(tag, row[metric], epoch)

    writer.close()
    print(f"TensorBoard event files saved to: {log_dir}")
    print(f"To view the results, run: tensorboard --logdir={log_dir}")

    return log_dir


def convert_multiple_csv_files(csv_files, log_dir=None):
    """Convert multiple CSV files to TensorBoard event files.

    Args:
        csv_files (list): List of paths to CSV files
        log_dir (str, optional): Base directory to save TensorBoard event files

    Returns:
        list: Paths to the TensorBoard log directories
    """
    log_dirs = []

    for csv_file in csv_files:
        # Use subdirectory named after CSV file if multiple files
        if log_dir is not None and len(csv_files) > 1:
            csv_name = Path(csv_file).stem
            file_log_dir = os.path.join(log_dir, csv_name)
        else:
            file_log_dir = log_dir

        log_dir_path = csv_to_tensorboard(csv_file, file_log_dir)
        log_dirs.append(log_dir_path)

    return log_dirs


def main():
    """Parse command line arguments and convert CSV files to TensorBoard."""
    parser = argparse.ArgumentParser(
        description="Convert CSV file(s) to TensorBoard event file(s)"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        nargs="+",
        help="Path to the CSV file(s) containing training metrics",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory to save TensorBoard event files",
    )
    parser.add_argument(
        "--no-auto-group",
        action="store_false",
        dest="auto_group",
        help="Disable automatic grouping of metrics by common prefixes",
    )

    args = parser.parse_args()

    # Handle single or multiple CSV files
    if len(args.csv_path) == 1:
        csv_to_tensorboard(args.csv_path[0], args.log_dir, args.auto_group)
    else:
        convert_multiple_csv_files(args.csv_path, args.log_dir)


if __name__ == "__main__":
    main()
