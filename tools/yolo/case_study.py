#!/usr/bin/env python3
"""
YOLO Case Study Module

This module provides functions to analyze YOLO model performance on images,
compare different models, and visualize predictions.
"""

import argparse
import os
import random
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class YOLOCaseStudy:
    """Class for performing case studies on YOLO models."""

    def __init__(self):
        """Initialize the case study manager."""
        self.model = None

    def load_model(self, model_path):
        """Load a trained YOLO model.

        Args:
            model_path (str): Path to the model weights

        Returns:
            YOLO model object
        """
        from ultralytics import YOLO

        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        return self.model

    def predict_image(self, image_path, conf=0.25, iou=0.45, device=""):
        """Run prediction on an image.

        Args:
            image_path (str): Path to the image
            conf (float): Confidence threshold
            iou (float): IoU threshold for NMS
            device (str): Device to use (cuda device or cpu)

        Returns:
            Prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Ensure conf and iou are float types
        conf = float(conf)
        iou = float(iou)

        # Run prediction
        results = self.model.predict(
            source=image_path, conf=conf, iou=iou, device=device, verbose=False
        )

        return results[0]  # Return the first result (single image)

    def visualize_prediction(
        self,
        image_path,
        results,
        output_path=None,
        show=True,
        figsize=(10, 6),
        model_name=None,
    ):
        """Visualize prediction results.

        Args:
            image_path (str): Path to the original image
            results: Prediction results from YOLO model
            output_path (str, optional): Path to save the visualization
            show (bool): Whether to display the plot
            figsize (tuple): Figure size (width, height) in inches
            model_name (str, optional): Name of the model for the title
        """
        # Load image
        img = Image.open(image_path)
        img_np = np.array(img)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Display the image
        ax.imshow(img_np)

        # Get prediction data
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()

        # Get class names
        class_names = results.names

        # Define colors for different classes (using a colormap)
        # Ensure class_names is not empty before creating cmap
        num_classes = len(class_names) if class_names else 1
        cmap = plt.cm.get_cmap("tab20", num_classes)

        # Plot each bounding box
        for _, (box, cls, conf) in enumerate(
            zip(boxes, classes, confidences, strict=False)
        ):
            # Get coordinates
            x1, y1, x2, y2 = box

            # Get class name and index
            cls_id = int(cls)
            cls_name = (
                class_names[cls_id]
                if class_names and cls_id < len(class_names)
                else f"Class {cls_id}"
            )

            # Create rectangle patch
            color = cmap(
                cls_id % cmap.N
            )  # Use modulo to avoid index out of bounds if cmap is smaller
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )

            # Add rectangle to the plot
            ax.add_patch(rect)

            # Add label
            label = f"{cls_name}: {conf:.2f}"
            plt.text(
                x1,
                y1 - 5,
                label,
                color="white",
                fontsize=10,
                bbox=dict(facecolor=color, alpha=0.8, edgecolor="none", pad=2),
            )

        # Set title with model name and total detections
        title = f"{model_name if model_name else 'YOLO'} Detection: {len(boxes)} objects found"
        plt.title(title, fontsize=14)

        # Remove axis ticks
        plt.axis("off")

        # Tight layout
        plt.tight_layout()

        # Save if output path is provided
        if output_path:
            # Ensure the directory exists
            output_dir_path = os.path.dirname(os.path.abspath(output_path))
            if (
                output_dir_path
            ):  # Check if dirname is not empty (e.g. for relative paths in current dir)
                os.makedirs(output_dir_path, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to {output_path}")

        # Show plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)  # Close the specific figure

    def run_case_study(
        self,
        model_path,
        image_path,  # Can be a single image or a directory
        output_dir="runs/detect/case_study",
        conf=0.25,
        iou=0.45,
        device="",
        show=True,
    ):
        """Run a complete case study on an image or directory of images.

        Args:
            model_path (str): Path to the trained model
            image_path (str): Path to the image or directory of images
            output_dir (str): Directory to save results
            conf (float): Confidence threshold
            iou (float): IoU threshold for NMS
            device (str): Device to use (cuda device or cpu)
            show (bool): Whether to display the plots
        """
        # Load model
        self.load_model(model_path)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get model name from path
        model_name_from_path = os.path.splitext(os.path.basename(model_path))[0]

        # Handle single image or directory
        if os.path.isdir(image_path):
            image_files = [
                os.path.join(image_path, f)
                for f in os.listdir(image_path)
                if f.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
                )
            ]
        elif os.path.isfile(image_path):
            image_files = [image_path]
        else:
            print(f"Error: Image path {image_path} is not a valid file or directory.")
            return

        if not image_files:
            print(f"No valid image files found in {image_path}.")
            return

        # Process each image
        for img_file in image_files:
            print(f"Processing {img_file}...")

            # Get base filename
            base_name = os.path.splitext(os.path.basename(img_file))[0]

            # Run prediction
            results = self.predict_image(img_file, conf, iou, device)

            # Visualize and save
            output_path_vis = os.path.join(output_dir, f"{base_name}_prediction.png")
            self.visualize_prediction(
                img_file,
                results,
                output_path_vis,
                show,
                model_name=model_name_from_path,
            )

            # Print detection summary
            print(f"Detected {len(results.boxes)} objects in {img_file}")

            # Print detailed class breakdown
            classes = results.boxes.cls.cpu().numpy()
            class_names = results.names

            class_counts = {}
            if class_names:  # Ensure class_names is available
                for cls_val in classes:
                    cls_id = int(cls_val)
                    if 0 <= cls_id < len(class_names):
                        cls_name = class_names[cls_id]
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    else:
                        print(
                            f"Warning: Detected class ID {cls_id} is out of bounds for model names."
                        )

            if class_counts:
                print("Class breakdown:")
                for cls_name, count in class_counts.items():
                    print(f"  - {cls_name}: {count}")
            else:
                print("No objects detected or class names unavailable for breakdown.")

            print("-" * 50)

    def batch_analysis(
        self,
        model_path,
        data_yaml,
        num_samples=5,
        output_dir="runs/detect/batch_analysis",
        conf=0.25,
        iou=0.45,
        device="",
        show=False,  # Typically False for batch operations
    ):
        """Run analysis on a batch of random samples from a dataset.

        Args:
            model_path (str): Path to the trained model
            data_yaml (str): Path to dataset YAML file
            num_samples (int): Number of random samples to analyze
            output_dir (str): Directory to save results
            conf (float): Confidence threshold
            iou (float): IoU threshold for NMS
            device (str): Device to use (cuda device or cpu)
            show (bool): Whether to display the plots
        """
        import yaml

        # Load model
        self.load_model(model_path)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get model name from path
        model_name_from_path = os.path.splitext(os.path.basename(model_path))[0]

        # Load dataset configuration
        try:
            with open(data_yaml) as f:
                data_config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Dataset YAML file not found at {data_yaml}")
            return
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {data_yaml}: {e}")
            return

        # Get validation image directory (or train, or a specified path)
        # Prioritize 'path' if it's a directory, then 'val', then 'train'
        base_dir_yaml = data_config.get("path", os.path.dirname(data_yaml))
        img_source_dir_key = None
        if "val" in data_config and data_config["val"]:
            img_source_dir_key = "val"
        elif "train" in data_config and data_config["train"]:
            img_source_dir_key = "train"
        # Add more fallbacks or specific keys if necessary e.g. data_config.get('images_dir')

        if not img_source_dir_key:
            # If 'path' is a directory and contains images, use it.
            if os.path.isdir(base_dir_yaml) and any(
                f.lower().endswith((".png", ".jpg", ".jpeg"))
                for f in os.listdir(base_dir_yaml)
            ):
                image_list_source = base_dir_yaml
            else:
                print(
                    "Validation or training image directory key ('val', 'train') not found or empty in data YAML, and 'path' is not a suitable image directory."
                )
                return
        else:
            image_list_source = data_config.get(img_source_dir_key)

        # Make path absolute if it's relative
        if not os.path.isabs(image_list_source):
            image_list_source = os.path.join(base_dir_yaml, image_list_source)

        image_list_source = os.path.abspath(image_list_source)

        # Get image files
        image_files = []
        if os.path.isdir(image_list_source):
            for root, _, files in os.walk(image_list_source):
                for file in files:
                    if file.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
                    ):
                        image_files.append(os.path.join(root, file))
        elif os.path.isfile(image_list_source):  # If data_yaml points to a list file
            try:
                with open(image_list_source) as f:
                    image_files = [
                        line.strip()
                        for line in f
                        if line.strip().lower().endswith((".png", ".jpg", ".jpeg"))
                    ]
                    # Ensure these paths are absolute or relative to a known base
                    image_files = [
                        os.path.join(base_dir_yaml, p) if not os.path.isabs(p) else p
                        for p in image_files
                    ]
            except Exception as e:
                print(f"Could not read image list from {image_list_source}: {e}")

        if not image_files:
            print(f"No images found in {image_list_source} or specified by it.")
            return

        # Select random samples
        if num_samples > len(image_files):
            print(
                f"Warning: Requested {num_samples} samples, but only {len(image_files)} images available. Using all available images."
            )
            num_samples = len(image_files)

        random_samples = random.sample(image_files, num_samples)

        # Process each sample
        for i, img_file in enumerate(random_samples):
            if not os.path.isfile(img_file):
                print(f"Warning: Sample image file not found: {img_file}. Skipping.")
                continue
            print(f"Processing sample {i + 1}/{num_samples}: {img_file}")

            # Get base filename
            base_name = os.path.splitext(os.path.basename(img_file))[0]

            # Run prediction
            results = self.predict_image(img_file, conf, iou, device)

            # Visualize and save
            output_path_vis = os.path.join(
                output_dir, f"sample_{i + 1}_{base_name}.png"
            )
            self.visualize_prediction(
                img_file,
                results,
                output_path_vis,
                show,  # show is typically False for batch
                model_name=model_name_from_path,
            )

        print(f"Batch analysis complete. Results saved to {output_dir}")

    def compare_models(
        self,
        image_paths,
        model_paths,
        model_names=None,
        output_dir="runs/detect/model_comparison",
        conf=0.25,
        iou=0.45,
        device="",
        show=True,
        export_individual=False,
        grid_cols=None,
        summary_stats=True,
    ):
        """Compare multiple models on the same set of images with tidy grid layout.
        Individual model results (if export_individual is True) will be saved
        in subdirectories named after each model within the output_dir.

        Optimized version that loads each model only once and presents results
        in a clean grid layout with summary statistics.

        Args:
            grid_cols (int, optional): Number of columns in grid. Auto-calculated if None.
            summary_stats (bool): Whether to include summary statistics in the comparison.
        """
        # Create main output directory
        os.makedirs(output_dir, exist_ok=True)

        # Set default model names if not provided
        if model_names is None:
            # Sanitize default model names for use as directory names
            model_names = [
                f"Model_{os.path.splitext(os.path.basename(mp))[0]}".replace(" ", "_")
                .replace(".", "_")
                .replace("/", "_")
                for mp in model_paths
            ]
        elif len(model_names) != len(model_paths):
            print(
                "Warning: model_names and model_paths have different lengths. Using default names for some."
            )
            default_names = [
                f"Model_{os.path.splitext(os.path.basename(mp))[0]}".replace(" ", "_")
                .replace(".", "_")
                .replace("/", "_")
                for mp in model_paths
            ]
            final_names = list(model_names) + default_names[len(model_names) :]
            model_names = final_names[: len(model_paths)]
            # Ensure provided model_names are also sanitized for directory use
            model_names = [
                name.replace(" ", "_").replace(".", "_").replace("/", "_")
                for name in model_names
            ]
        else:
            # Ensure provided model_names are sanitized for directory use
            model_names = [
                name.replace(" ", "_").replace(".", "_").replace("/", "_")
                for name in model_names
            ]

        # Filter out non-existent images upfront
        valid_image_paths = []
        for image_path in image_paths:
            if os.path.isfile(image_path):
                valid_image_paths.append(image_path)
            else:
                print(f"Warning: Image file not found: {image_path}. Skipping.")

        if not valid_image_paths:
            print("No valid images found. Aborting comparison.")
            return {}

        # Phase 1: Collect all results by loading each model only once
        print("Phase 1: Running predictions...")
        results_matrix = {image_path: {} for image_path in valid_image_paths}
        loaded_images = {}  # Cache loaded images to avoid reloading

        for model_idx, (model_path, model_name) in enumerate(
            zip(model_paths, model_names, strict=False)
        ):
            print(f"\nLoading model {model_idx + 1}/{len(model_paths)}: {model_name}")

            try:
                self.load_model(model_path)
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
                continue

            # Run this model on all images
            for image_path in valid_image_paths:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                print(f"  Processing {base_name}...")

                try:
                    results = self.predict_image(image_path, conf, iou, device)
                    results_matrix[image_path][model_name] = results

                    # Cache the loaded image for later visualization
                    if image_path not in loaded_images:
                        img = Image.open(image_path)
                        loaded_images[image_path] = np.array(img)

                except Exception as e:
                    print(f"Error processing {image_path} with {model_name}: {e}")
                    continue

        # Phase 2: Create visualizations using cached results
        print("\nPhase 2: Creating visualizations...")
        all_results_by_image = {}

        for image_path in valid_image_paths:
            if not results_matrix[image_path]:
                print(
                    f"No successful predictions for {image_path}. Skipping visualization."
                )
                continue

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            print(f"\nCreating comparison for: {base_name}")

            img_np = loaded_images[image_path]

            # Collect results for this image across all models that succeeded
            current_image_model_results = []
            successful_models = []

            for model_name in model_names:
                if model_name in results_matrix[image_path]:
                    results = results_matrix[image_path][model_name]
                    current_image_model_results.append((model_name, results))
                    successful_models.append((model_name, results))

            if not successful_models:
                continue

            # Create individual exports if requested
            if export_individual:
                for model_name, results in successful_models:
                    model_specific_output_dir = os.path.join(output_dir, model_name)
                    individual_output_filename = f"{base_name}_prediction.png"
                    individual_output_path = os.path.join(
                        model_specific_output_dir, individual_output_filename
                    )

                    self.visualize_prediction(
                        image_path,
                        results,
                        output_path=individual_output_path,
                        show=False,
                        model_name=model_name,
                    )
                    print(
                        f"  Individual result for {model_name} saved to {individual_output_path}"
                    )

            # Create tidy comparison plot with grid layout
            n_models = len(successful_models)

            # Calculate optimal grid dimensions
            if grid_cols is None:
                if n_models <= 2:
                    grid_cols = n_models
                elif n_models <= 4:
                    grid_cols = 2
                elif n_models <= 6:
                    grid_cols = 3
                else:
                    grid_cols = 4

            grid_rows = (n_models + grid_cols - 1) // grid_cols  # Ceiling division

            # Calculate figure size for better readability
            subplot_width = 5
            subplot_height = 5
            fig_width = grid_cols * subplot_width
            fig_height = grid_rows * subplot_height

            # Add extra height for summary if enabled
            if summary_stats and n_models > 1:
                fig_height += 1.5

            # Create figure with calculated dimensions
            fig = plt.figure(figsize=(fig_width, fig_height))

            # Add summary statistics at the top if enabled and multiple models
            if summary_stats and n_models > 1:
                summary_ax = plt.subplot2grid(
                    (grid_rows + 1, grid_cols), (0, 0), colspan=grid_cols
                )
                self._create_comparison_summary(
                    successful_models, summary_ax, base_name
                )
                start_row = 1
            else:
                start_row = 0

            # Create model comparison subplots
            model_colors = plt.cm.Set3(np.linspace(0, 1, n_models))

            for i, (model_name, results) in enumerate(successful_models):
                row = start_row + i // grid_cols
                col = i % grid_cols
                ax = plt.subplot2grid((grid_rows + start_row, grid_cols), (row, col))

                ax.imshow(img_np)
                boxes = results.boxes.xyxy.cpu().numpy()
                pred_classes = results.boxes.cls.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                class_names_map = results.names
                num_model_classes = len(class_names_map) if class_names_map else 1
                cmap = plt.cm.get_cmap("tab20", num_model_classes)

                # Improved text positioning to reduce overlap
                text_positions = []
                for box, cls, pred_conf in zip(
                    boxes, pred_classes, confidences, strict=False
                ):
                    x1, y1, x2, y2 = box
                    cls_id = int(cls)
                    cls_name = (
                        class_names_map[cls_id]
                        if class_names_map and cls_id < len(class_names_map)
                        else f"C{cls_id}"
                    )

                    color = cmap(cls_id % cmap.N)
                    rect = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        edgecolor=color,
                        facecolor="none",
                    )
                    ax.add_patch(rect)

                    # Smart label positioning
                    label = f"{cls_name}: {pred_conf:.2f}"
                    label_y = y1 - 10

                    # Check for text overlap and adjust position
                    for prev_x, prev_y in text_positions:
                        if abs(x1 - prev_x) < 100 and abs(label_y - prev_y) < 20:
                            label_y = prev_y - 25

                    text_positions.append((x1, label_y))

                    ax.text(
                        x1,
                        label_y,
                        label,
                        color="white",
                        fontsize=9,
                        fontweight="bold",
                        bbox=dict(facecolor=color, alpha=0.9, edgecolor="white", pad=2),
                        verticalalignment="top",
                    )

                # Enhanced title with color coding and stats
                avg_conf = np.mean(confidences) if len(confidences) > 0 else 0
                title_color = model_colors[i]
                ax.set_title(
                    f"{model_name}\n{len(boxes)} objects | Avg Conf: {avg_conf:.2f}",
                    fontsize=11,
                    fontweight="bold",
                    bbox=dict(
                        facecolor=title_color, alpha=0.3, edgecolor="none", pad=5
                    ),
                )
                ax.axis("off")

            all_results_by_image[image_path] = current_image_model_results

            # Enhanced main title
            main_title = f"Model Comparison: {base_name}"
            if summary_stats and n_models > 1:
                main_title += f" ({n_models} models)"

            fig.suptitle(main_title, fontsize=16, fontweight="bold", y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            comparison_output_filename = f"{base_name}_model_comparison.png"
            comparison_output_path = os.path.join(
                output_dir, comparison_output_filename
            )
            plt.savefig(comparison_output_path, dpi=300, bbox_inches="tight")
            print(f"  Comparison saved to {comparison_output_path}")

            if show:
                plt.show()

            plt.close(fig)

        # Clean up cached images
        loaded_images.clear()

        return all_results_by_image

    def _create_comparison_summary(self, successful_models, ax, base_name):
        """Create a summary statistics section for model comparison.

        Args:
            successful_models: List of (model_name, results) tuples
            ax: Matplotlib axis for the summary
            base_name: Base name of the image being processed
        """
        ax.axis("off")

        # Collect summary data
        summary_data = []
        for model_name, results in successful_models:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            # Calculate statistics
            total_detections = len(boxes)
            avg_confidence = np.mean(confidences) if len(confidences) > 0 else 0
            max_confidence = np.max(confidences) if len(confidences) > 0 else 0
            min_confidence = np.min(confidences) if len(confidences) > 0 else 0

            # Count unique classes
            unique_classes = len(np.unique(classes)) if len(classes) > 0 else 0

            summary_data.append(
                {
                    "model": model_name,
                    "detections": total_detections,
                    "avg_conf": avg_confidence,
                    "max_conf": max_confidence,
                    "min_conf": min_confidence,
                    "classes": unique_classes,
                }
            )

        # Create summary table
        if summary_data:
            # Table headers
            headers = [
                "Model",
                "Detections",
                "Avg Conf",
                "Max Conf",
                "Min Conf",
                "Classes",
            ]

            # Prepare table data
            table_data = []
            for data in summary_data:
                row = [
                    data["model"][:15] + "..."
                    if len(data["model"]) > 15
                    else data["model"],
                    str(data["detections"]),
                    f"{data['avg_conf']:.3f}",
                    f"{data['max_conf']:.3f}",
                    f"{data['min_conf']:.3f}",
                    str(data["classes"]),
                ]
                table_data.append(row)

            # Create table
            table = ax.table(
                cellText=table_data,
                colLabels=headers,
                cellLoc="center",
                loc="center",
                bbox=[0.1, 0.2, 0.8, 0.6],
            )

            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)

            # Color code the header
            for i in range(len(headers)):
                table[(0, i)].set_facecolor("#4CAF50")
                table[(0, i)].set_text_props(weight="bold", color="white")

            # Color code the model names to match the plots
            model_colors = plt.cm.Set3(np.linspace(0, 1, len(summary_data)))
            for i, color in enumerate(model_colors):
                table[(i + 1, 0)].set_facecolor(color)
                table[(i + 1, 0)].set_text_props(weight="bold")

        # Add title for summary
        ax.text(
            0.5,
            0.9,
            f"Detection Summary for {base_name}",
            horizontalalignment="center",
            fontsize=12,
            fontweight="bold",
            transform=ax.transAxes,
        )

    # Command line argument additions for main function
    def add_comparison_args(parser):
        """Add comparison-specific arguments to parser"""
        parser.add_argument(
            "--export-individual",
            action="store_true",
            help="In --compare mode, also export individual model prediction images separately",
        )
        parser.add_argument(
            "--grid-cols",
            type=int,
            default=None,
            help="Number of columns in comparison grid (auto-calculated if not specified)",
        )
        parser.add_argument(
            "--no-summary",
            action="store_true",
            help="Disable summary statistics in comparison view",
        )

    # Updated compare mode call in main function
    def run_compare_mode(case_study, args):
        """Run comparison mode with updated parameters"""
        print(
            f"\nRunning compare mode for models '{', '.join(args.models)}' on images '{', '.join(args.image)}'..."
        )
        case_study.compare_models(
            image_paths=args.image,
            model_paths=args.models,
            model_names=args.model_names,
            output_dir=args.output_dir,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            show=not args.no_show,
            export_individual=args.export_individual,
            grid_cols=args.grid_cols,
            summary_stats=not args.no_summary,
        )


def main():
    """Main function to parse arguments and run case study."""
    parser = argparse.ArgumentParser(description="YOLO Case Study Analyzer")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--single",
        action="store_true",
        help="Run on a single image or directory of images",
    )
    mode_group.add_argument(
        "--batch",
        action="store_true",
        help="Run on a batch of images from a dataset YAML",
    )
    mode_group.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple models on one or more images",
    )

    # Arguments for specific modes
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",  # Can now take multiple images
        help="Path(s) to image(s) or directory of images. Used by --single and --compare modes.",
    )
    parser.add_argument(
        "--model", type=str, help="Path to the trained model (for --single or --batch)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Paths to models to compare (for --compare mode, requires at least two)",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="+",
        help="Custom names for the models in --compare mode (must match number of --models)",
    )
    parser.add_argument(
        "--data", type=str, help="Path to dataset YAML file (for --batch mode)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of random samples for --batch mode (default: 5)",
    )

    # Common arguments for prediction and output
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45, help="IoU threshold for NMS (default: 0.45)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use (e.g., 'cpu', '0', '0,1,2,3') (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save results (defaults based on mode)",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display plots automatically"
    )
    parser.add_argument(
        "--export-individual",
        action="store_true",
        help="In --compare mode, also export individual model prediction images separately",
    )

    args = parser.parse_args()
    case_study = YOLOCaseStudy()

    # Determine default output directory if not specified
    if args.output_dir is None:
        if args.single:
            args.output_dir = "runs/detect/case_study"
        elif args.batch:
            args.output_dir = "runs/detect/batch_analysis"
        elif args.compare:
            args.output_dir = "runs/detect/model_comparison"
        else:  # Should not happen due to mutually exclusive group
            args.output_dir = "runs/detect/misc"

    os.makedirs(args.output_dir, exist_ok=True)

    if args.single:
        if not args.model:
            parser.error("--model is required for --single mode.")
        if not args.image:
            parser.error("--image is required for --single mode.")

        for img_path_item in args.image:  # args.image is a list
            print(
                f"\nRunning single mode for model '{args.model}' on '{img_path_item}'..."
            )
            case_study.run_case_study(
                model_path=args.model,
                image_path=img_path_item,  # run_case_study handles if it's a file or dir
                output_dir=args.output_dir,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                show=not args.no_show,
            )

    elif args.batch:
        if not args.model:
            parser.error("--model is required for --batch mode.")
        if not args.data:
            parser.error("--data (dataset YAML) is required for --batch mode.")

        print(
            f"\nRunning batch mode for model '{args.model}' with dataset YAML '{args.data}'..."
        )
        case_study.batch_analysis(
            model_path=args.model,
            data_yaml=args.data,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            show=not args.no_show,  # Usually False for batch, but respect --no-show
        )

    elif args.compare:
        if (
            not args.models or len(args.models) < 1
        ):  # Allow 1 model for "comparison" which acts like single view
            parser.error(
                "--models requires at least one model path for --compare mode."
            )
        if not args.image:
            parser.error(
                "--image (at least one image path) is required for --compare mode."
            )
        if args.model_names and len(args.model_names) != len(args.models):
            parser.error(
                "--model-names, if provided, must match the number of --models."
            )

        print(
            f"\nRunning compare mode for models '{', '.join(args.models)}' on images '{', '.join(args.image)}'..."
        )
        case_study.compare_models(
            image_paths=args.image,  # args.image is a list
            model_paths=args.models,
            model_names=args.model_names,
            output_dir=args.output_dir,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            show=not args.no_show,
            export_individual=args.export_individual,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
