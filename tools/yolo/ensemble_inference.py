import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class UltralyticsEnsemble:
    """
    Ensemble multiple Ultralytics models for improved performance.
    Supports object detection, classification, and segmentation tasks.
    """

    def __init__(
        self,
        model_paths: List[str],
        weights: Optional[List[float]] = None,
        device: str = "auto",
    ):
        """
        Initialize the ensemble with multiple models.

        Args:
            model_paths: List of paths to model files (.pt files)
            weights: List of weights for each model (default: equal weights)
            device: Device to run inference on ('auto', 'cpu', 'cuda', etc.)
        """
        self.model_paths = model_paths

        # Handle 'auto' device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.models = []

        # Set equal weights if not provided
        if weights is None:
            self.weights = [1.0 / len(model_paths)] * len(model_paths)
        else:
            if len(weights) != len(model_paths):
                raise ValueError("Number of weights must match number of models")
            # Normalize weights
            # total_weight = sum(weights)
            # self.weights = [w / total_weight for w in weights]
            self.weights = weights

        # Load all models
        self._load_models()

    def _load_models(self):
        """Load all YOLO models."""
        print(f"Loading {len(self.model_paths)} models...")
        print(f"Using device: {self.device}")

        for i, model_path in enumerate(self.model_paths):
            try:
                # Load model with device specification
                model = YOLO(model_path)

                # Move model to device (Ultralytics handles device properly)
                if hasattr(model, "to"):
                    model.to(self.device)

                self.models.append(model)
                print(f"✓ Loaded model {i + 1}: {model_path}")
            except Exception as e:
                print(f"✗ Failed to load model {model_path}: {e}")
                raise

    def predict(
        self,
        source: Union[str, np.ndarray, List],
        ensemble_method: str = "nms",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 300,
        save_dir: Optional[str] = None,
    ) -> List:
        """
        Run ensemble prediction on input source.

        Args:
            source: Input source (image path, image array, video path, etc.)
            ensemble_method: Method for ensembling ('nms', 'wbf', 'avg')
            conf_threshold: Confidence threshold for filtering predictions
            iou_threshold: IoU threshold for NMS
            max_det: Maximum number of detections
            save_dir: Directory to save results (optional)

        Returns:
            List of ensemble results
        """
        # Get predictions from all models
        all_predictions = []

        print("Running inference on all models...")
        for i, model in enumerate(self.models):
            try:
                results = model.predict(
                    source=source,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    max_det=max_det,
                    device=self.device,  # Pass device to predict method
                    verbose=False,
                )
                all_predictions.append(results)
                print(f"✓ Model {i + 1} inference complete")
            except Exception as e:
                print(f"✗ Model {i + 1} inference failed: {e}")
                continue

        if not all_predictions:
            raise RuntimeError("All models failed to produce predictions")

        # Ensemble the predictions
        ensemble_results = self._ensemble_predictions(
            all_predictions, ensemble_method, conf_threshold, iou_threshold
        )

        return ensemble_results

    def _ensemble_predictions(
        self,
        all_predictions: List,
        method: str,
        conf_threshold: float,
        iou_threshold: float,
    ) -> List:
        """
        Ensemble predictions from multiple models.

        Args:
            all_predictions: List of prediction results from each model
            method: Ensemble method ('nms', 'wbf', 'avg')
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold

        Returns:
            Ensembled results
        """
        if method == "nms":
            return self._ensemble_nms(all_predictions, conf_threshold, iou_threshold)
        elif method == "wbf":
            return self._ensemble_wbf(all_predictions, conf_threshold, iou_threshold)
        elif method == "avg":
            return self._ensemble_average(
                all_predictions, conf_threshold, iou_threshold
            )
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

    def _ensemble_nms(
        self, all_predictions: List, conf_threshold: float, iou_threshold: float
    ) -> List:
        """
        Ensemble using Non-Maximum Suppression across all model predictions.
        """
        ensemble_results = []

        # Process each image/frame
        num_images = len(all_predictions[0])

        for img_idx in range(num_images):
            # Collect all boxes, scores, and classes for this image
            all_boxes = []
            all_scores = []
            all_classes = []

            for model_idx, predictions in enumerate(all_predictions):
                result = predictions[img_idx]

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()

                    # Apply model weight to scores
                    scores = scores * self.weights[model_idx]

                    all_boxes.extend(boxes)
                    all_scores.extend(scores)
                    all_classes.extend(classes)

            if len(all_boxes) > 0:
                # Convert to numpy arrays
                all_boxes = np.array(all_boxes)
                all_scores = np.array(all_scores)
                all_classes = np.array(all_classes)

                # Apply NMS
                keep_indices = self._apply_nms(
                    all_boxes, all_scores, iou_threshold, conf_threshold
                )

                # Create ensemble result
                ensemble_result = {
                    "boxes": all_boxes[keep_indices],
                    "scores": all_scores[keep_indices],
                    "classes": all_classes[keep_indices],
                    "image_shape": all_predictions[0][img_idx].orig_shape,
                }
            else:
                ensemble_result = {
                    "boxes": np.array([]),
                    "scores": np.array([]),
                    "classes": np.array([]),
                    "image_shape": all_predictions[0][img_idx].orig_shape,
                }

            ensemble_results.append(ensemble_result)

        return ensemble_results

    def _ensemble_wbf(
        self, all_predictions: List, conf_threshold: float, iou_threshold: float
    ) -> List:
        """
        Ensemble using Weighted Box Fusion (simplified version).
        """
        # This is a simplified WBF implementation
        # For a full WBF implementation, consider using the ensemble-boxes library
        print(
            "Note: Using simplified WBF. For full WBF, install ensemble-boxes library."
        )
        return self._ensemble_average(all_predictions, conf_threshold, iou_threshold)

    def _ensemble_average(
        self, all_predictions: List, conf_threshold: float, iou_threshold: float
    ) -> List:
        """
        Ensemble by averaging overlapping predictions.
        """
        ensemble_results = []
        num_images = len(all_predictions[0])

        for img_idx in range(num_images):
            # Collect all predictions for this image
            all_boxes = []
            all_scores = []
            all_classes = []

            for model_idx, predictions in enumerate(all_predictions):
                result = predictions[img_idx]

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy() * self.weights[model_idx]
                    classes = result.boxes.cls.cpu().numpy()

                    all_boxes.extend(boxes)
                    all_scores.extend(scores)
                    all_classes.extend(classes)

            if len(all_boxes) > 0:
                # Group overlapping boxes and average their properties
                final_boxes, final_scores, final_classes = (
                    self._average_overlapping_boxes(
                        np.array(all_boxes),
                        np.array(all_scores),
                        np.array(all_classes),
                        iou_threshold,
                        conf_threshold,
                    )
                )

                ensemble_result = {
                    "boxes": final_boxes,
                    "scores": final_scores,
                    "classes": final_classes,
                    "image_shape": all_predictions[0][img_idx].orig_shape,
                }
            else:
                ensemble_result = {
                    "boxes": np.array([]),
                    "scores": np.array([]),
                    "classes": np.array([]),
                    "image_shape": all_predictions[0][img_idx].orig_shape,
                }

            ensemble_results.append(ensemble_result)

        return ensemble_results

    def _apply_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
        conf_threshold: float,
    ) -> np.ndarray:
        """Apply Non-Maximum Suppression."""
        # Filter by confidence threshold
        valid_indices = scores >= conf_threshold
        if not np.any(valid_indices):
            return np.array([], dtype=int)

        boxes = boxes[valid_indices]
        scores = scores[valid_indices]

        # Convert to torch tensors for torchvision NMS
        boxes_tensor = torch.from_numpy(boxes).float()
        scores_tensor = torch.from_numpy(scores).float()

        # Apply NMS
        keep_indices = torch.ops.torchvision.nms(
            boxes_tensor, scores_tensor, iou_threshold
        )

        # Convert back to numpy and map to original indices
        original_indices = np.where(valid_indices)[0]
        return original_indices[keep_indices.numpy()]

    def _average_overlapping_boxes(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        iou_threshold: float,
        conf_threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Average overlapping boxes of the same class."""
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        # Filter by confidence
        valid_mask = scores >= conf_threshold
        if not np.any(valid_mask):
            return np.array([]), np.array([]), np.array([])

        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        classes = classes[valid_mask]

        final_boxes = []
        final_scores = []
        final_classes = []
        used = np.zeros(len(boxes), dtype=bool)

        for i in range(len(boxes)):
            if used[i]:
                continue

            current_class = classes[i]
            same_class_mask = (classes == current_class) & (~used)

            if np.sum(same_class_mask) == 1:
                final_boxes.append(boxes[i])
                final_scores.append(scores[i])
                final_classes.append(classes[i])
                used[i] = True
                continue

            # Find overlapping boxes of the same class
            same_class_indices = np.where(same_class_mask)[0]
            overlapping_indices = [i]

            for j in same_class_indices:
                if j != i and self._calculate_iou(boxes[i], boxes[j]) > iou_threshold:
                    overlapping_indices.append(j)

            # Average overlapping boxes
            if len(overlapping_indices) > 1:
                overlapping_boxes = boxes[overlapping_indices]
                overlapping_scores = scores[overlapping_indices]

                # Weighted average of boxes
                weights = overlapping_scores / np.sum(overlapping_scores)
                avg_box = np.average(overlapping_boxes, axis=0, weights=weights)
                avg_score = np.mean(overlapping_scores)

                final_boxes.append(avg_box)
                final_scores.append(avg_score)
                final_classes.append(current_class)

                used[overlapping_indices] = True
            else:
                final_boxes.append(boxes[i])
                final_scores.append(scores[i])
                final_classes.append(classes[i])
                used[i] = True

        return np.array(final_boxes), np.array(final_scores), np.array(final_classes)

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU) between two boxes."""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])

        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0

        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def visualize_results(
        self,
        source: Union[str, List[str]],
        results: List,
        save_dir: Optional[str] = None,
    ):
        """
        Visualize ensemble results on input images.

        Args:
            source: Path to input image/folder or list of image paths
            results: Ensemble results
            save_dir: Directory to save visualizations (for batch processing)
        """
        # Handle different source types
        if isinstance(source, str):
            if Path(source).is_dir():
                # Get all images from directory
                image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
                image_paths = []
                for ext in image_extensions:
                    image_paths.extend(Path(source).glob(ext))
                    image_paths.extend(Path(source).glob(ext.upper()))
                image_paths = sorted([str(p) for p in image_paths])
            elif Path(source).is_file():
                image_paths = [source]
            else:
                print(f"Could not find source: {source}")
                return
        else:
            image_paths = source

        if len(image_paths) != len(results):
            print(
                f"Warning: Number of images ({len(image_paths)}) doesn't match results ({len(results)})"
            )
            return

        # Process each image
        for i, (img_path, result) in enumerate(zip(image_paths, results)):
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not load image: {img_path}")
                continue

            # Draw detections
            boxes = result["boxes"]
            scores = result["scores"]
            classes = result["classes"]

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box.astype(int)

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                label = f"Class {int(cls)}: {score:.2f}"
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # Save or display
            if save_dir:
                # Create save directory
                Path(save_dir).mkdir(parents=True, exist_ok=True)

                # Generate save path
                img_name = Path(img_path).stem
                save_path = Path(save_dir) / f"{img_name}_ensemble.jpg"

                cv2.imwrite(str(save_path), img)
                print(f"✓ Visualization saved: {save_path}")
            else:
                # Display (only for single images)
                if len(image_paths) == 1:
                    cv2.imshow("Ensemble Results", img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print(
                        "Warning: Cannot display multiple images. Use save_dir parameter."
                    )


def main():
    """Example usage of the UltralyticsEnsemble class."""
    parser = argparse.ArgumentParser(description="Ensemble multiple Ultralytics models")
    parser.add_argument(
        "--models", nargs="+", required=True, help="Paths to model files"
    )
    parser.add_argument("--source", required=True, help="Path to input image/video")
    parser.add_argument(
        "--weights", nargs="+", type=float, help="Model weights (optional)"
    )
    parser.add_argument(
        "--method", default="nms", choices=["nms", "wbf", "avg"], help="Ensemble method"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--save", help="Path to save visualization")
    parser.add_argument("--save-dir", help="Directory to save batch visualizations")

    args = parser.parse_args()

    # Create ensemble
    ensemble = UltralyticsEnsemble(
        model_paths=args.models, weights=args.weights, device=args.device
    )

    # Run prediction
    results = ensemble.predict(
        source=args.source,
        ensemble_method=args.method,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    # Print results summary
    for i, result in enumerate(results):
        print(f"\nImage {i + 1}:")
        print(f"  Detections: {len(result['boxes'])}")
        if len(result["boxes"]) > 0:
            print(f"  Classes: {result['classes']}")
            print(f"  Scores: {result['scores']}")

    # Visualize results
    if Path(args.source).suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        # Single image
        ensemble.visualize_results(args.source, results, args.save)
    elif Path(args.source).is_dir():
        # Folder of images
        save_dir = args.save_dir or "./results/ensemble_visualizations"
        ensemble.visualize_results(args.source, results, save_dir)
        print(f"\n📁 All visualizations saved to: {save_dir}")
    else:
        print("📹 Video processing - visualizations not supported for videos")


if __name__ == "__main__":
    main()
