#!/usr/bin/env python3
"""
YOLO Training Module

Provides comprehensive functionality for training YOLOv12 models for object detection,
including custom datasets, hyperparameter tuning, and various training configurations.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")


class YOLOTrainer:
    """Class for training YOLO models."""

    def __init__(self):
        """Initialize the YOLO trainer."""

    def create_dataset_yaml(
        self,
        train_path,
        val_path,
        test_path=None,
        class_names=None,
        output_path="dataset.yaml",
    ):
        """Create a YAML file for the dataset configuration.

        Args:
            train_path (str): Path to training images
            val_path (str): Path to validation images
            test_path (str, optional): Path to test images
            class_names (list): List of class names
            output_path (str): Path to save the YAML file

        Returns:
            str: Path to the created YAML file
        """
        if class_names is None:
            raise ValueError("Class names must be provided")

        # Determine common base path
        common_path = os.path.dirname(os.path.commonprefix([train_path, val_path]))

        # Create dataset config dictionary
        dataset_config = {
            "path": common_path,
            "train": os.path.relpath(train_path, os.path.dirname(output_path)),
            "val": os.path.relpath(val_path, os.path.dirname(output_path)),
            "names": {i: name for i, name in enumerate(class_names)},
        }

        if test_path:
            dataset_config["test"] = os.path.relpath(
                test_path, os.path.dirname(output_path)
            )

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Write the YAML file
        with open(output_path, "w") as f:
            yaml.dump(dataset_config, f, default_flow_style=False)

        print(f"Dataset configuration saved to {output_path}")
        return output_path

    def train(
        self,
        data_yaml,
        model=None,
        model_size="m",
        epochs=100,
        batch_size=16,
        imgsz=640,
        device="",
        project="runs/train",
        name="exp",
        pretrained=True,
        resume=False,
        optimizer="auto",
        lr0=0.01,
        patience=50,
        save_period=10,
        workers=8,
        exist_ok=False,
        verbose=False,
        model_dir=None,
    ):
        """Train a YOLOv12 model.

        Args:
            data_yaml (str): Path to dataset YAML file
            model (str, optional): Custom model path or name
            model_size (str): Model size (n, s, m, l, x) when using standard models
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            imgsz (int): Image size
            device (str): Device to use (cuda device or cpu)
            project (str): Project name
            name (str): Experiment name
            pretrained (bool): Use pretrained weights
            resume (bool): Resume training from last checkpoint
            optimizer (str): Optimizer (SGD, Adam, etc.)
            lr0 (float): Initial learning rate
            patience (int): Early stopping patience
            save_period (int): Save checkpoint every x epochs
            workers (int): Number of worker threads for data loading
            exist_ok (bool): Overwrite existing experiment
            verbose (bool): Print verbose output
            model_dir (str, optional): Directory to download/load models from

        Returns:
            str: Path to the trained model
        """
        # Import after dependencies check
        from ultralytics import YOLO

        # Set model directory if specified
        if model_dir:
            import os

            os.environ["TORCH_HOME"] = model_dir

        # Determine model path
        if model is not None:
            # Use custom model path or name
            model_path = model
        else:
            # Select model based on size
            model_path = (
                f"yolo12{model_size}.pt" if pretrained else f"yolo12{model_size}.yaml"
            )

        # Load model
        model = YOLO(model_path)

        # Train the model
        _ = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            project=project,
            name=name,
            resume=resume,
            optimizer=optimizer,
            lr0=lr0,
            patience=patience,
            save_period=save_period,
            workers=workers,
            exist_ok=exist_ok,
            verbose=verbose,
        )

        # Return path to best model
        return str(Path(project) / name / "weights" / "best.pt")

    def validate(self, model_path, data_yaml, batch_size=16, imgsz=640, device=""):
        """Validate a trained YOLO model.

        Args:
            model_path (str): Path to the trained model
            data_yaml (str): Path to dataset YAML file
            batch_size (int): Batch size
            imgsz (int): Image size
            device (str): Device to use (cuda device or cpu)

        Returns:
            dict: Validation metrics
        """
        # Import after dependencies check
        from ultralytics import YOLO

        # Load model
        model = YOLO(model_path)

        # Validate the model
        metrics = model.val(
            data=data_yaml, batch=batch_size, imgsz=imgsz, device=device
        )

        return metrics

    def export(
        self, model_path, format="onnx", imgsz=640, half=False, simplify=True, opset=12
    ):
        """Export a trained YOLO model to different formats.

        Args:
            model_path (str): Path to the trained model
            format (str): Export format (onnx, torchscript, openvino, etc.)
            imgsz (int): Image size
            half (bool): Use FP16 half-precision
            simplify (bool): Simplify ONNX model
            opset (int): ONNX opset version

        Returns:
            str: Path to the exported model
        """
        # Import after dependencies check
        from ultralytics import YOLO

        # Load model
        model = YOLO(model_path)

        # Export the model
        exported_path = model.export(
            format=format, imgsz=imgsz, half=half, simplify=simplify, opset=opset
        )

        return exported_path


def main():
    """Parse command line arguments and run YOLO training."""
    parser = argparse.ArgumentParser(description="Train YOLOv12 models")

    # Dataset arguments
    dataset_group = parser.add_argument_group("Dataset")
    dataset_group.add_argument("--data", type=str, help="Path to dataset YAML file")
    dataset_group.add_argument("--train-path", type=str, help="Path to training images")
    dataset_group.add_argument("--val-path", type=str, help="Path to validation images")
    dataset_group.add_argument("--test-path", type=str, help="Path to test images")
    dataset_group.add_argument("--classes", type=str, nargs="+", help="Class names")

    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model",
        type=str,
        help="Custom model path or name (overrides model-size if provided)",
    )
    model_group.add_argument(
        "--model-size",
        type=str,
        default="m",
        choices=["n", "s", "m", "l", "x"],
        help="Model size (n, s, m, l, x) when using standard models",
    )
    model_group.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights (only applies when using model-size)",
    )
    model_group.add_argument(
        "--resume", action="store_true", help="Resume training from last checkpoint"
    )

    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    train_group.add_argument("--batch-size", type=int, default=16, help="Batch size")
    train_group.add_argument("--imgsz", type=int, default=640, help="Image size")
    train_group.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use (cuda device, i.e. 0 or 0,1,2,3 or cpu)",
    )
    train_group.add_argument(
        "--optimizer", type=str, default="auto", help="Optimizer (SGD, Adam, etc.)"
    )
    train_group.add_argument(
        "--lr0", type=float, default=0.01, help="Initial learning rate"
    )
    train_group.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (epochs without improvement)",
    )
    train_group.add_argument(
        "--save-period", type=int, default=10, help="Save checkpoint every x epochs"
    )
    train_group.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads for data loading",
    )

    # Output arguments
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--project", type=str, default="runs/train", help="Project name"
    )
    output_group.add_argument("--name", type=str, default="exp", help="Experiment name")
    output_group.add_argument(
        "--exist-ok", action="store_true", help="Overwrite existing experiment"
    )
    output_group.add_argument(
        "--verbose", action="store_true", help="Print verbose output"
    )

    # Export arguments
    export_group = parser.add_argument_group("Export")
    export_group.add_argument(
        "--export", action="store_true", help="Export model after training"
    )
    export_group.add_argument(
        "--export-format",
        type=str,
        default="onnx",
        choices=[
            "onnx",
            "torchscript",
            "openvino",
            "coreml",
            "saved_model",
            "pb",
            "tflite",
            "edgetpu",
            "tfjs",
            "paddle",
            "ncnn",
        ],
        help="Export format",
    )
    export_group.add_argument(
        "--half", action="store_true", help="Use FP16 half-precision for export"
    )
    export_group.add_argument(
        "--simplify", action="store_true", help="Simplify ONNX model"
    )
    export_group.add_argument(
        "--opset", type=int, default=12, help="ONNX opset version"
    )

    # Model directory setting
    model_group.add_argument(
        "--model-dir",
        type=str,
        help="Directory to store downloaded models (sets TORCH_HOME environment variable)",
    )

    args = parser.parse_args()

    # Create the trainer
    trainer = YOLOTrainer()

    # Create dataset YAML if not provided
    if args.data is None:
        if args.train_path is None or args.val_path is None or args.classes is None:
            parser.error(
                "Either --data or --train-path, --val-path, and --classes must be provided"
            )

        args.data = trainer.create_dataset_yaml(
            train_path=args.train_path,
            val_path=args.val_path,
            test_path=args.test_path,
            class_names=args.classes,
            output_path=os.path.join(args.project, args.name, "dataset.yaml"),
        )

    # Train the model
    model_path = trainer.train(
        data_yaml=args.data,
        model=args.model,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=args.pretrained,
        resume=args.resume,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        save_period=args.save_period,
        workers=args.workers,
        exist_ok=args.exist_ok,
        verbose=args.verbose,
        # Model directory
        model_dir=args.model_dir,
    )

    print(f"\nTraining complete. Best model saved at: {model_path}")

    # Validate the model
    print("\nValidating model...")
    metrics = trainer.validate(
        model_path=model_path,
        data_yaml=args.data,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
    )

    # Print detailed metrics
    print("\nValidation metrics:")

    # Access metrics attributes directly
    if hasattr(metrics, "box"):
        print(f"  box.map: {metrics.box.map}")
        print(f"  box.map50: {metrics.box.map50}")
        print(f"  box.map75: {metrics.box.map75}")

        if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
            print(f"  box.maps: {metrics.box.maps}")

    # Print speed metrics if available
    if hasattr(metrics, "speed"):
        print("  Speed metrics:")
        if isinstance(metrics.speed, dict):
            for k, v in metrics.speed.items():
                print(f"    {k}: {v}")
        else:
            print(f"    speed: {metrics.speed}")
    else:
        # If metrics doesn't have the expected structure, print it directly
        print(f"  metrics: {metrics}")

    # Export the model if requested
    if args.export:
        print(f"\nExporting model to {args.export_format}...")
        exported_path = trainer.export(
            model_path=model_path,
            format=args.export_format,
            imgsz=args.imgsz,
            half=args.half,
            simplify=args.simplify,
            opset=args.opset,
        )

        print(f"Model exported to: {exported_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
