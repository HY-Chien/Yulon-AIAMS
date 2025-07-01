#!/usr/bin/env python3
"""
Plot Results Module

Visualizes training metrics from results.csv files generated during YOLO training.
Creates plots for losses, metrics, and learning rates for performance analysis.
Supports multiple models for comparison.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import TABLEAU_COLORS

from tools.utils.file_utils import ensure_directory_exists


def load_multiple_csv_files(
    csv_files: List[str], model_names: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load multiple CSV files and return a dictionary of DataFrames.

    Args:
        csv_files: List of paths to CSV files
        model_names: Optional list of model names (defaults to filenames)

    Returns:
        Dictionary mapping model names to DataFrames
    """
    data_dict = {}

    if model_names is None:
        model_names = [Path(f).stem for f in csv_files]

    if len(model_names) != len(csv_files):
        raise ValueError("Number of model names must match number of CSV files")

    for csv_file, model_name in zip(csv_files, model_names):
        try:
            data = pd.read_csv(csv_file)
            data_dict[model_name] = data
            print(f"Loaded {model_name}: {len(data)} epochs")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue

    return data_dict


def get_common_columns(data_dict: Dict[str, pd.DataFrame], prefix: str) -> List[str]:
    """
    Get columns that are common across all models with a given prefix.

    Args:
        data_dict: Dictionary of model DataFrames
        prefix: Column prefix to filter by

    Returns:
        List of common columns
    """
    if not data_dict:
        return []

    # Get all columns with the prefix from all models
    all_columns = set()
    common_columns = None

    for model_name, data in data_dict.items():
        model_columns = {col for col in data.columns if prefix in col}
        all_columns.update(model_columns)

        if common_columns is None:
            common_columns = model_columns
        else:
            common_columns = common_columns.intersection(model_columns)

    return sorted(list(common_columns))


def apply_smoothing(
    data_dict: Dict[str, pd.DataFrame], smooth_factor: float = 0.6
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Apply exponential moving average smoothing to all numeric columns.

    Args:
        data_dict: Dictionary of model DataFrames
        smooth_factor: Smoothing factor (0-1, higher = more smoothing)

    Returns:
        Dictionary of smoothed data
    """
    smoothed_dict = {}

    for model_name, data in data_dict.items():
        smoothed_dict[model_name] = {}
        for column in data.columns:
            if column not in ["epoch", "time"] and pd.api.types.is_numeric_dtype(
                data[column]
            ):
                smoothed_dict[model_name][column] = (
                    data[column].ewm(alpha=(1 - smooth_factor)).mean()
                )

    return smoothed_dict


def plot_multiple_results(
    csv_files: Union[str, List[str]],
    model_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    show: bool = True,
    smooth: bool = True,
    smooth_factor: float = 0.6,
    figsize: tuple = (12, 8),
    dpi: int = 100,
):
    """
    Plot training results from multiple CSV files for model comparison.

    Args:
        csv_files: Path(s) to the results.csv file(s)
        model_names: Optional list of model names for legends
        output_dir: Directory to save the plots
        show: Whether to display the plots
        smooth: Whether to apply exponential moving average smoothing
        smooth_factor: Smoothing factor (0-1, higher = more smoothing)
        figsize: Figure size for individual plots
        dpi: DPI for saved figures
    """
    # Handle single file input
    if isinstance(csv_files, str):
        csv_files = [csv_files]

    # Load all CSV files
    data_dict = load_multiple_csv_files(csv_files, model_names)

    if not data_dict:
        print("No valid CSV files found!")
        return

    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(csv_files[0])), "results_comparison"
        )

    output_dir = ensure_directory_exists(output_dir)

    # Apply smoothing if requested
    smoothed_dict = None
    if smooth:
        smoothed_dict = apply_smoothing(data_dict, smooth_factor)

    # Set up plotting style
    plt.style.use("seaborn-v0_8-darkgrid")
    colors = list(TABLEAU_COLORS.values())

    # Plot 1: Training Losses Comparison
    train_loss_columns = get_common_columns(data_dict, "train/")
    train_loss_columns = [col for col in train_loss_columns if "loss" in col]

    if train_loss_columns:
        plot_comparison(
            data_dict,
            smoothed_dict,
            train_loss_columns,
            "Training Losses Comparison",
            "Loss",
            output_dir,
            "train_losses_comparison.png",
            colors,
            figsize,
            show,
            smooth,
        )

    # Plot 2: Validation Losses Comparison
    val_loss_columns = get_common_columns(data_dict, "val/")
    val_loss_columns = [col for col in val_loss_columns if "loss" in col]

    if val_loss_columns:
        plot_comparison(
            data_dict,
            smoothed_dict,
            val_loss_columns,
            "Validation Losses Comparison",
            "Loss",
            output_dir,
            "val_losses_comparison.png",
            colors,
            figsize,
            show,
            smooth,
        )

    # Plot 3: Metrics Comparison
    metric_columns = get_common_columns(data_dict, "metrics/")

    if metric_columns:
        plot_comparison(
            data_dict,
            smoothed_dict,
            metric_columns,
            "Performance Metrics Comparison",
            "Value",
            output_dir,
            "metrics_comparison.png",
            colors,
            figsize,
            show,
            smooth,
        )

    # Plot 4: Learning Rates Comparison
    lr_columns = get_common_columns(data_dict, "lr/")

    if lr_columns:
        plot_comparison(
            data_dict,
            smoothed_dict,
            lr_columns,
            "Learning Rate Comparison",
            "Learning Rate",
            output_dir,
            "learning_rates_comparison.png",
            colors,
            figsize,
            show,
            smooth,
            log_scale=True,
        )

    # Plot 5: Best Metrics Summary
    plot_best_metrics_summary(data_dict, output_dir, figsize, show)

    # Plot 6: Training Progress Dashboard
    plot_dashboard(data_dict, smoothed_dict, output_dir, colors, show, smooth)

    # Plot 7: Individual model plots (if multiple models)
    if len(data_dict) > 1:
        for model_name, data in data_dict.items():
            model_output_dir = os.path.join(output_dir, f"{model_name}_individual")
            model_output_dir = ensure_directory_exists(model_output_dir)

            # Use the original single-model function for individual plots
            plot_results(
                csv_files[list(data_dict.keys()).index(model_name)],
                model_output_dir,
                show=False,
                smooth=smooth,
                smooth_factor=smooth_factor,
            )

    if not show:
        plt.close("all")

    print(f"Comparison plots saved to {output_dir}")


def plot_comparison(
    data_dict: Dict[str, pd.DataFrame],
    smoothed_dict: Optional[Dict[str, Dict[str, pd.Series]]],
    columns: List[str],
    title: str,
    ylabel: str,
    output_dir: str,
    filename: str,
    colors: List[str],
    figsize: tuple,
    show: bool,
    smooth: bool,
    log_scale: bool = False,
):
    """Plot comparison of specific metrics across models."""
    fig, axes = plt.subplots(
        len(columns), 1, figsize=(figsize[0], figsize[1] * len(columns)), squeeze=False
    )
    if len(columns) == 1:
        axes = [axes[0]]
    else:
        axes = axes.flatten()

    for idx, column in enumerate(columns):
        ax = axes[idx]

        for model_idx, (model_name, data) in enumerate(data_dict.items()):
            if column in data.columns:
                color = colors[model_idx % len(colors)]

                if (
                    smooth
                    and smoothed_dict
                    and model_name in smoothed_dict
                    and column in smoothed_dict[model_name]
                ):
                    y_data = smoothed_dict[model_name][column]
                    label = f"{model_name} (smoothed)"
                else:
                    y_data = data[column]
                    label = model_name

                ax.plot(data["epoch"], y_data, label=label, color=color, linewidth=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{column.replace('train/', '').replace('val/', '').replace('metrics/', '').replace('lr/', '')}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        if log_scale:
            ax.set_yscale("log")

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=100, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_best_metrics_summary(
    data_dict: Dict[str, pd.DataFrame], output_dir: str, figsize: tuple, show: bool
):
    """Plot a summary of best achieved metrics for each model."""
    summary_data = []

    for model_name, data in data_dict.items():
        model_summary = {"Model": model_name}

        # Find best values for common metrics
        for column in data.columns:
            if any(
                metric in column.lower()
                for metric in ["precision", "recall", "map", "f1"]
            ):
                if not data[column].isna().all():
                    model_summary[column] = data[column].max()
            elif "loss" in column.lower():
                if not data[column].isna().all():
                    model_summary[column] = data[column].min()

        summary_data.append(model_summary)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # Create a bar plot for each metric
        metric_columns = [col for col in summary_df.columns if col != "Model"]

        if metric_columns:
            n_metrics = len(metric_columns)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(figsize[0] * n_cols // 2, figsize[1] * n_rows // 2),
            )
            if n_metrics == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()

            for idx, metric in enumerate(metric_columns):
                ax = axes[idx]
                summary_df.plot(x="Model", y=metric, kind="bar", ax=ax, legend=False)
                ax.set_title(
                    metric.replace("train/", "")
                    .replace("val/", "")
                    .replace("metrics/", "")
                )
                ax.set_xlabel("Model")
                ax.tick_params(axis="x", rotation=45)

            # Hide unused subplots
            for idx in range(len(metric_columns), len(axes)):
                axes[idx].set_visible(False)

            plt.suptitle("Best Metrics Summary", fontsize=16, fontweight="bold")
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "best_metrics_summary.png"),
                dpi=100,
                bbox_inches="tight",
            )

            if show:
                plt.show()
            else:
                plt.close(fig)


def plot_dashboard(
    data_dict: Dict[str, pd.DataFrame],
    smoothed_dict: Optional[Dict[str, Dict[str, pd.Series]]],
    output_dir: str,
    colors: List[str],
    show: bool,
    smooth: bool,
):
    """Create a comprehensive dashboard view."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # Training losses
    train_loss_columns = get_common_columns(data_dict, "train/")
    train_loss_columns = [col for col in train_loss_columns if "loss" in col]

    for model_idx, (model_name, data) in enumerate(data_dict.items()):
        color = colors[model_idx % len(colors)]

        for column in train_loss_columns:
            if column in data.columns:
                if (
                    smooth
                    and smoothed_dict
                    and model_name in smoothed_dict
                    and column in smoothed_dict[model_name]
                ):
                    y_data = smoothed_dict[model_name][column]
                else:
                    y_data = data[column]

                axes[0, 0].plot(
                    data["epoch"],
                    y_data,
                    label=f"{model_name} - {column.replace('train/', '')}",
                    color=color,
                    linewidth=2,
                )

    axes[0, 0].set_title("Training Losses")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Validation losses
    val_loss_columns = get_common_columns(data_dict, "val/")
    val_loss_columns = [col for col in val_loss_columns if "loss" in col]

    for model_idx, (model_name, data) in enumerate(data_dict.items()):
        color = colors[model_idx % len(colors)]

        for column in val_loss_columns:
            if column in data.columns:
                if (
                    smooth
                    and smoothed_dict
                    and model_name in smoothed_dict
                    and column in smoothed_dict[model_name]
                ):
                    y_data = smoothed_dict[model_name][column]
                else:
                    y_data = data[column]

                axes[0, 1].plot(
                    data["epoch"],
                    y_data,
                    label=f"{model_name} - {column.replace('val/', '')}",
                    color=color,
                    linewidth=2,
                )

    axes[0, 1].set_title("Validation Losses")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Metrics
    metric_columns = get_common_columns(data_dict, "metrics/")

    for model_idx, (model_name, data) in enumerate(data_dict.items()):
        color = colors[model_idx % len(colors)]

        for column in metric_columns:
            if column in data.columns:
                if (
                    smooth
                    and smoothed_dict
                    and model_name in smoothed_dict
                    and column in smoothed_dict[model_name]
                ):
                    y_data = smoothed_dict[model_name][column]
                else:
                    y_data = data[column]

                axes[1, 0].plot(
                    data["epoch"],
                    y_data,
                    label=f"{model_name} - {column.replace('metrics/', '')}",
                    color=color,
                    linewidth=2,
                )

    axes[1, 0].set_title("Performance Metrics")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rates
    lr_columns = get_common_columns(data_dict, "lr/")

    for model_idx, (model_name, data) in enumerate(data_dict.items()):
        color = colors[model_idx % len(colors)]

        for column in lr_columns:
            if column in data.columns:
                if (
                    smooth
                    and smoothed_dict
                    and model_name in smoothed_dict
                    and column in smoothed_dict[model_name]
                ):
                    y_data = smoothed_dict[model_name][column]
                else:
                    y_data = data[column]

                axes[1, 1].plot(
                    data["epoch"],
                    y_data,
                    label=f"{model_name} - {column.replace('lr/', '')}",
                    color=color,
                    linewidth=2,
                )

    axes[1, 1].set_title("Learning Rate Schedule")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        "Training Dashboard - Multi-Model Comparison", fontsize=18, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dashboard.png"), dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_results(csv_file, output_dir=None, show=True, smooth=True, smooth_factor=0.6):
    """
    Plot training results from a single CSV file (original function, maintained for backward compatibility).

    Args:
        csv_file: Path to the results.csv file
        output_dir: Directory to save the plots (if None, uses the directory of the CSV file)
        show: Whether to display the plots
        smooth: Whether to apply exponential moving average smoothing
        smooth_factor: Smoothing factor (0-1, higher = more smoothing)
    """
    # Use the new multi-model function with a single file
    plot_multiple_results(
        [csv_file],
        output_dir=output_dir,
        show=show,
        smooth=smooth,
        smooth_factor=smooth_factor,
    )


def main():
    """Parse command line arguments and run the plotting function."""
    parser = argparse.ArgumentParser(
        description="Plot training results from CSV file(s)"
    )

    parser.add_argument(
        "csv_files", nargs="+", type=str, help="Path(s) to the results.csv file(s)"
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        type=str,
        default=None,
        help="Custom names for the models (must match number of CSV files)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the plots (default: results_comparison in first CSV directory)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plots (just save them)",
    )
    parser.add_argument(
        "--no-smooth", action="store_true", help="Don't apply smoothing to the plots"
    )
    parser.add_argument(
        "--smooth-factor",
        type=float,
        default=0.6,
        help="Smoothing factor (0-1, higher = more smoothing, default: 0.6)",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=int,
        default=[12, 8],
        help="Figure size for individual plots (width height, default: 12 8)",
    )

    args = parser.parse_args()

    # Check if all CSV files exist
    for csv_file in args.csv_files:
        if not os.path.isfile(csv_file):
            print(f"Error: CSV file '{csv_file}' not found")
            return 1

    # Validate model names if provided
    if args.model_names and len(args.model_names) != len(args.csv_files):
        print(
            f"Error: Number of model names ({len(args.model_names)}) must match number of CSV files ({len(args.csv_files)})"
        )
        return 1

    # Plot the results
    plot_multiple_results(
        csv_files=args.csv_files,
        model_names=args.model_names,
        output_dir=args.output_dir,
        show=not args.no_show,
        smooth=not args.no_smooth,
        smooth_factor=args.smooth_factor,
        figsize=tuple(args.figsize),
    )

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
