#!/usr/bin/env python3
"""
YOLO Test Validation Module

This module provides functionality to test and validate YOLO models
by running a minimal validation on a test dataset.
"""

import sys


class YOLOValidator:
    """Class for validating YOLO models."""

    def __init__(self):
        """Initialize the YOLO validator."""

    def run_validation(self, model_path, data_yaml, batch=2, imgsz=320):
        """Run validation test on a YOLO model.

        Args:
            model_path (str): Path to the model weights
            data_yaml (str): Path to the dataset YAML file
            batch (int): Batch size for validation
            imgsz (int): Image size for validation

        Returns:
            object: Validation metrics
        """
        # Import here to avoid importing before installing
        from ultralytics import YOLO

        # Load model
        try:
            model = YOLO(model_path)
            print(f"Loaded model: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

        # Run validation
        try:
            print("Running validation...")
            metrics = model.val(data=data_yaml, batch=batch, imgsz=imgsz)

            # Print metrics information
            print(f"\nType of metrics: {type(metrics).__name__}")

            # Try to access box metrics
            if hasattr(metrics, "box"):
                print("\nBox metrics:")
                print(f"  map: {metrics.box.map}")
                print(f"  map50: {metrics.box.map50}")
                print(f"  map75: {metrics.box.map75}")

                if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
                    print(f"  maps: {metrics.box.maps}")

            # Try to access speed metrics
            if hasattr(metrics, "speed"):
                print("\nSpeed metrics:")
                if isinstance(metrics.speed, dict):
                    for k, v in metrics.speed.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"  speed: {metrics.speed}")

            print("\nValidation test completed successfully!")
            return metrics

        except Exception as e:
            print(f"Error during validation: {e}")
            import traceback

            traceback.print_exc()
            return None


def main():
    """Run a test validation on a YOLO model."""
    import argparse

    parser = argparse.ArgumentParser(description="Test and validate a YOLO model")

    parser.add_argument(
        "--model", type=str, default="yolo12m.pt", help="Path to the model weights"
    )
    parser.add_argument(
        "--data", type=str, default="coco128.yaml", help="Path to the dataset YAML file"
    )
    parser.add_argument(
        "--batch", type=int, default=2, help="Batch size for validation"
    )
    parser.add_argument(
        "--imgsz", type=int, default=320, help="Image size for validation"
    )

    args = parser.parse_args()

    # Create validator and run validation
    validator = YOLOValidator()
    results = validator.run_validation(
        model_path=args.model, data_yaml=args.data, batch=args.batch, imgsz=args.imgsz
    )

    return 0 if results is not None else 1


if __name__ == "__main__":
    sys.exit(main())
