#!/usr/bin/env python3
"""
Synthetic Data Generation Module

This module provides functionality to generate synthetic data for training YOLO
object detection models. It places icons on backgrounds with various transformations
to create realistic training datasets.
"""

import fnmatch
import os
import random
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


class SyntheticDataGenerator:
    """Class for generating synthetic object detection data."""

    def __init__(self, seed=None):
        """Initialize the synthetic data generator.

        Args:
            seed (int, optional): Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def load_icons(
        self, icons_folder: str, min_size: int = 30, max_size: int = 100
    ) -> List[Dict]:
        """Load icon images from a folder, ensuring all icons meet size requirements.

        Args:
            icons_folder (str): Path to the folder containing icon images
            min_size (int): Minimum size (width or height) for icons after resizing
            max_size (int): Maximum size (width or height) for icons

        Returns:
            List[Dict]: List of dictionaries containing icon data and metadata
        """
        print(f"Loading icons from {icons_folder}...")
        icons = []

        # Define supported image extensions
        image_extensions = [
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.bmp",
            "*.gif",
            "*.tiff",
            "*.webp",
        ]

        # Counter for assigning unique IDs
        icon_id = 0

        # First collect all valid image files
        all_image_files = []
        for root, _, files in os.walk(icons_folder):
            for file in files:
                if any(fnmatch.fnmatch(file.lower(), ext) for ext in image_extensions):
                    all_image_files.append((root, file))

        # Iterate through all valid image files with a progress bar
        for root, file in tqdm(all_image_files, desc="Loading icons"):
            # Check if the file has a valid image extension
            if any(fnmatch.fnmatch(file.lower(), ext) for ext in image_extensions):
                icon_path = os.path.join(root, file)
                try:
                    # Get class name from filename before processing
                    class_name = os.path.splitext(os.path.basename(icon_path))[0]
                    print(f"Processing icon: {icon_path} (class: {class_name})")

                    icon = Image.open(icon_path)

                    # Convert to RGBA to handle transparency consistently
                    if icon.mode != "RGBA":
                        icon = icon.convert("RGBA")

                    # Get original dimensions
                    orig_width, orig_height = icon.size
                    print(f"  Original size: {orig_width}x{orig_height}")

                    # Calculate new dimensions maintaining aspect ratio
                    if orig_width > orig_height:
                        # Width is larger, so we scale based on width
                        if orig_width > max_size:
                            scale = max_size / orig_width
                        elif orig_width < min_size:
                            scale = min_size / orig_width
                        else:
                            scale = 1.0
                    else:
                        # Height is larger or equal, so we scale based on height
                        if orig_height > max_size:
                            scale = max_size / orig_height
                        elif orig_height < min_size:
                            scale = min_size / orig_height
                        else:
                            scale = 1.0

                    # Apply scaling
                    new_width = int(orig_width * scale)
                    new_height = int(orig_height * scale)

                    # Ensure both dimensions are within bounds
                    if new_width > max_size or new_height > max_size:
                        # Recalculate scale to ensure neither dimension exceeds max_size
                        scale = min(max_size / orig_width, max_size / orig_height)
                        new_width = int(orig_width * scale)
                        new_height = int(orig_height * scale)

                    if new_width < min_size or new_height < min_size:
                        # Recalculate scale to ensure neither dimension is below min_size
                        scale = max(min_size / orig_width, min_size / orig_height)
                        new_width = int(orig_width * scale)
                        new_height = int(orig_height * scale)

                    # Perform resizing
                    if scale != 1.0:
                        print(f"  Resizing to: {new_width}x{new_height}")
                        icon = icon.resize((new_width, new_height), Image.LANCZOS)

                    # Add to our list
                    icons.append(
                        {
                            "image": icon,
                            "width": new_width,
                            "height": new_height,
                            "path": icon_path,
                            "class_name": class_name,
                            "class_id": icon_id,  # Assign a unique ID to each icon class
                        }
                    )

                    icon_id += 1
                    print(f"  Successfully loaded icon with ID {icon_id - 1}")

                except Exception as e:
                    print(f"  Error loading icon {icon_path}: {e}")

        print(f"Loaded {len(icons)} icons out of {icon_id} attempted")

        # Print summary of loaded icons
        if icons:
            print("Icon classes loaded:")
            class_names = [icon["class_name"] for icon in icons]
            unique_class_names = set(class_names)
            print(
                f"  {len(unique_class_names)} unique class names out of {len(icons)} icons"
            )

            if len(unique_class_names) < len(icons):
                print("  WARNING: Some icons have duplicate class names:")
                from collections import Counter

                class_counts = Counter(class_names)
                for name, count in class_counts.items():
                    if count > 1:
                        print(f"    '{name}' appears {count} times")

        return icons

    def load_backgrounds(
        self,
        backgrounds_folder: str,
        min_size: Tuple[int, int] = (640, 640),
        augment_backgrounds: bool = True,
        target_count: int = None,
    ) -> List[Dict]:
        """Load background images from a folder.

        Args:
            backgrounds_folder (str): Path to the folder containing background images
            min_size (Tuple[int, int]): Minimum size (width, height) for backgrounds
            augment_backgrounds (bool): Whether to augment backgrounds to create more variations
            target_count (int, optional): Target number of backgrounds to generate

        Returns:
            List[Dict]: List of dictionaries containing background data and metadata
        """
        import random

        print(f"Loading backgrounds from {backgrounds_folder}...")
        backgrounds = []

        # Define supported image extensions
        image_extensions = [
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.bmp",
            "*.gif",
            "*.tiff",
            "*.webp",
        ]

        # Create a standard white background of size (1024, 768)
        standard_size = (1024, 768)

        # First collect all valid background image files
        all_bg_files = []
        for root, _, files in os.walk(backgrounds_folder):
            for file in files:
                if any(fnmatch.fnmatch(file.lower(), ext) for ext in image_extensions):
                    all_bg_files.append((root, file))

        # Iterate with progress bar
        for root, file in tqdm(all_bg_files, desc="Loading backgrounds"):
            bg_path = os.path.join(root, file)
            try:
                # Load the original image
                original_img = Image.open(bg_path).convert("RGB")

                # Create a white background
                white_bg = Image.new("RGB", standard_size, (255, 255, 255))

                # Resize the original image to fit within the white background while maintaining aspect ratio
                orig_width, orig_height = original_img.size
                aspect_ratio = orig_width / orig_height

                # Calculate new dimensions to fit within the white background
                if (
                    aspect_ratio > standard_size[0] / standard_size[1]
                ):  # Width is the limiting factor
                    new_width = min(orig_width, standard_size[0])
                    new_height = int(new_width / aspect_ratio)
                else:  # Height is the limiting factor
                    new_height = min(orig_height, standard_size[1])
                    new_width = int(new_height * aspect_ratio)

                # Resize the original image
                resized_img = original_img.resize(
                    (new_width, new_height), Image.LANCZOS
                )

                # Calculate position to center the image on the white background
                paste_x = (standard_size[0] - new_width) // 2
                paste_y = (standard_size[1] - new_height) // 2

                # Paste the resized image onto the white background
                white_bg.paste(resized_img, (paste_x, paste_y))

                # Use the composite image as our background
                bg = white_bg
                width, height = bg.size

                backgrounds.append(
                    {"image": bg, "width": width, "height": height, "path": bg_path}
                )

            except Exception as e:
                print(f"Error loading background {bg_path}: {e}")

        print(f"Loaded {len(backgrounds)} backgrounds")

        # Generate augmented backgrounds if needed
        if augment_backgrounds and (
            target_count is None or len(backgrounds) < target_count
        ):
            augmented_backgrounds = self._augment_backgrounds(
                backgrounds,
                target_count=target_count if target_count else len(backgrounds) * 3,
            )
            backgrounds.extend(augmented_backgrounds)
            print(
                f"Generated {len(augmented_backgrounds)} additional augmented backgrounds"
            )

        # Shuffle the backgrounds for random access
        random.shuffle(backgrounds)

        # If target count specified, trim to that count
        if target_count and len(backgrounds) > target_count:
            backgrounds = backgrounds[:target_count]
            print(f"Trimmed to {target_count} backgrounds as requested")

        print(f"Total backgrounds available: {len(backgrounds)}")
        return backgrounds

    def _augment_backgrounds(
        self, backgrounds: List[Dict], target_count: int
    ) -> List[Dict]:
        """Generate augmented variations of backgrounds to increase diversity.

        Args:
            backgrounds (List[Dict]): List of background dictionaries
            target_count (int): Target number of backgrounds to reach

        Returns:
            List[Dict]: List of augmented background dictionaries
        """
        import numpy as np

        # Setup background augmentation pipeline with Albumentations
        bg_augmenter = A.Compose(
            [
                # Color transformations
                A.OneOf(
                    [
                        A.HueSaturationValue(
                            hue_shift_limit=20,
                            sat_shift_limit=30,
                            val_shift_limit=20,
                            p=0.7,
                        ),
                        A.RGBShift(
                            r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.7
                        ),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2, p=0.7
                        ),
                        A.ChannelShuffle(p=0.3),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.25),
                    ],
                    p=0.7,
                ),
                # Geometric transformations (with lower probability)
                A.OneOf(
                    [
                        A.RandomCrop(height=768, width=1024, p=0.3),
                        A.CenterCrop(height=768, width=1024, p=0.3),
                        A.Perspective(scale=(0.05, 0.1), p=0.3),
                        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                        A.RandomRotate90(p=0.3),
                    ],
                    p=0.5,
                ),
                # Weather effects
                A.OneOf(
                    [
                        A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.4, p=0.3),
                        A.RandomRain(drop_length=10, blur_value=3, p=0.3),
                        A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, p=0.3),
                    ],
                    p=0.3,
                ),
            ]
        )

        augmented_backgrounds = []
        num_augmentations_needed = max(0, target_count - len(backgrounds))

        if num_augmentations_needed <= 0:
            return augmented_backgrounds

        # Calculate how many variations per original background we need
        avg_variations_per_bg = num_augmentations_needed / len(backgrounds)
        variations_per_bg = max(1, int(np.ceil(avg_variations_per_bg)))

        print(
            f"Creating approximately {variations_per_bg} variations per background..."
        )

        # Generate augmented variations with progress bar
        with tqdm(
            total=min(
                len(backgrounds), num_augmentations_needed // variations_per_bg + 1
            ),
            desc="Augmenting backgrounds",
        ) as pbar:
            for i, bg in enumerate(backgrounds):
                if len(augmented_backgrounds) >= num_augmentations_needed:
                    break

                variations_created = 0
                # Create multiple variations of each background
                for j in range(variations_per_bg):
                    if len(augmented_backgrounds) >= num_augmentations_needed:
                        break

                    try:
                        # Convert PIL image to numpy array for Albumentations
                        bg_array = np.array(bg["image"])

                        # Apply augmentation - Albumentations returns a dict with the transformed image
                        transformed = bg_augmenter(image=bg_array)
                        aug_bg_array = transformed["image"]

                        # Convert back to PIL image
                        aug_bg = Image.fromarray(aug_bg_array)

                        # Create augmented background metadata
                        augmented_backgrounds.append(
                            {
                                "image": aug_bg,
                                "width": aug_bg.width,
                                "height": aug_bg.height,
                                "path": f"augmented_{i}_{j}_{bg['path']}",
                                "source_image": bg["path"],
                                "is_augmented": True,
                            }
                        )

                        # Count successful variations
                        variations_created += 1

                    except Exception as e:
                        print(f"Error augmenting background {bg['path']}: {e}")

                # Update progress bar
                pbar.update(1)

        return augmented_backgrounds

    def _calculate_average_color(
        self, images: List[Image.Image]
    ) -> Tuple[int, int, int]:
        """Calculate the average color from a list of images.

        Args:
            images (List[Image.Image]): List of PIL Image objects

        Returns:
            Tuple[int, int, int]: RGB tuple representing the average color
        """
        # Initialize variables to accumulate color values
        r_total, g_total, b_total = 0, 0, 0
        pixel_count = 0

        # Sample pixels from each image (using downsampling for efficiency)
        for img in images:
            # Resize to a small image for faster processing
            small_img = img.resize((50, 50), Image.LANCZOS)
            img_array = np.array(small_img)

            # Accumulate RGB values
            r_total += img_array[:, :, 0].sum()
            g_total += img_array[:, :, 1].sum()
            b_total += img_array[:, :, 2].sum()
            pixel_count += img_array.shape[0] * img_array.shape[1]

        # Calculate averages and ensure they're in the valid range
        if pixel_count > 0:
            r_avg = min(255, max(0, int(r_total / pixel_count)))
            g_avg = min(255, max(0, int(g_total / pixel_count)))
            b_avg = min(255, max(0, int(b_total / pixel_count)))
            return (r_avg, g_avg, b_avg)
        else:
            # Fallback to light gray if no pixels were processed
            return (255, 255, 255)

    def setup_augmentation(self, augmentation_strength="medium") -> A.Compose:
        """Setup the augmentation pipeline for icons.

        Args:
            augmentation_strength (str): Level of augmentation ('light', 'medium', 'strong')

        Returns:
            A.Compose: The augmentation pipeline
        """
        if augmentation_strength == "none":
            return None
        # Define scale parameters for each strength level
        if augmentation_strength == "light":
            scale_range = (0.9, 1.1)
            rotate_range = (-10, 10)
            shear_range = (-5, 5)

        elif augmentation_strength == "strong":
            scale_range = (0.7, 1.3)
            rotate_range = (-30, 30)
            shear_range = (-15, 15)

        else:  # medium (default)
            scale_range = (0.8, 1.2)
            rotate_range = (-15, 15)
            shear_range = (-10, 10)

        # Create augmentation pipeline with Albumentations
        # For RGB portion (without alpha channel)
        self.rgb_augmenter = A.Compose(
            [
                A.Affine(
                    scale=(scale_range[0], scale_range[1]),
                    rotate=rotate_range,
                    shear=shear_range,
                    interpolation=cv2.INTER_LINEAR,
                    cval=0,
                    p=1.0,
                ),
                A.HorizontalFlip(0.5),
                A.VerticalFlip(0.5),
                A.RandomRotate90(0.6),
            ]
        )

        # For alpha channel (only geometric transformations)
        self.alpha_augmenter = A.Compose(
            [
                A.Affine(
                    scale=(scale_range[0], scale_range[1]),
                    rotate=rotate_range,
                    shear=shear_range,
                    interpolation=cv2.INTER_LINEAR,
                    cval=0,
                    p=1.0,
                ),
                A.HorizontalFlip(0.5),
                A.VerticalFlip(0.5),
                A.RandomRotate90(0.6),
            ]
        )

        return self.rgb_augmenter

    def augment_icon(self, icon: Dict) -> Dict:
        """Apply augmentation to an icon with padding instead of cropping.

        Args:
            icon (Dict): Icon data dictionary

        Returns:
            Dict: Augmented icon data dictionary
        """
        # Ensure augmentation pipeline is set up
        if not hasattr(self, "rgb_augmenter"):
            self.setup_augmentation()

        # Convert PIL image to numpy array for Albumentations
        icon_array = np.array(icon["image"])

        # Apply augmentation
        # If the image has an alpha channel (RGBA), we need to handle it separately
        if icon_array.shape[2] == 4:
            # Extract RGB and alpha channels
            rgb = icon_array[..., :3]
            alpha = icon_array[..., 3]

            # Create random seed to ensure same transformations for RGB and alpha
            random_seed = np.random.randint(0, 2**32 - 1)

            # Set the random seed for numpy to ensure reproducible transformations
            np.random.seed(random_seed)

            # Augment the RGB part
            transformed_rgb = self.rgb_augmenter(image=rgb)
            aug_rgb = transformed_rgb["image"]

            # Reset the random seed to the same value for alpha channel
            np.random.seed(random_seed)

            # Apply the same geometric transformations to the alpha channel
            # Convert alpha to 3-channel grayscale for Albumentations compatibility
            alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1)
            transformed_alpha = self.alpha_augmenter(image=alpha_3ch)
            aug_alpha_3ch = transformed_alpha["image"]

            # Extract just the first channel (they should all be the same)
            aug_alpha = aug_alpha_3ch[..., 0]

            # Ensure dimensions match before concatenating
            if aug_rgb.shape[:2] != aug_alpha.shape[:2]:
                # Resize alpha to match RGB dimensions if they don't match
                from PIL import Image as PILImage

                alpha_pil = PILImage.fromarray(aug_alpha, mode="L")
                alpha_pil = alpha_pil.resize(
                    (aug_rgb.shape[1], aug_rgb.shape[0]), PILImage.LANCZOS
                )
                aug_alpha = np.array(alpha_pil)

            # Reshape alpha to add channel dimension back
            aug_alpha = aug_alpha[..., np.newaxis]

            # Merge RGB and alpha back together
            aug_icon_array = np.concatenate([aug_rgb, aug_alpha], axis=2)
            aug_icon = Image.fromarray(aug_icon_array.astype(np.uint8), mode="RGBA")
        else:
            # For RGB images (no transparency), just apply the augmentation
            transformed = self.rgb_augmenter(image=icon_array)
            aug_icon_array = transformed["image"]
            aug_icon = Image.fromarray(aug_icon_array.astype(np.uint8))

        # Create a copy of the icon dict with augmented data
        aug_icon_dict = icon.copy()
        aug_icon_dict["image"] = aug_icon
        aug_icon_dict["width"] = aug_icon.width
        aug_icon_dict["height"] = aug_icon.height

        return aug_icon_dict

    def place_icons_on_background(
        self,
        background: Dict,
        icons: List[Dict],
        num_icons: int,
        min_scale: float = 0.05,
        max_scale: float = 0.15,
        max_attempts: int = 100,
    ) -> Tuple[Image.Image, List[Dict]]:
        """Place icons on a background image without overlapping.

        Args:
            background (Dict): Background image data
            icons (List[Dict]): List of icon data dictionaries
            num_icons (int): Number of icons to place on the background
            min_scale (float): Minimum scale of icons relative to background
            max_scale (float): Maximum scale of icons relative to background
            max_attempts (int): Maximum number of placement attempts per icon

        Returns:
            Tuple[Image.Image, List[Dict]]: The composite image and list of placed icon data
        """
        # Import here to ensure it's available
        from shapely.geometry import box

        bg_img = background["image"].copy()
        bg_width, bg_height = bg_img.size

        # Create a canvas for the composite image
        composite = bg_img.copy()

        # Keep track of placed icons and their bounding boxes
        placed_icons = []
        bboxes = []

        # Try to place each icon
        icons_to_place = min(num_icons, len(icons))
        for _ in range(icons_to_place):
            # Randomly select an icon
            icon_dict = random.choice(icons)

            # Augment the icon
            aug_icon_dict = self.augment_icon(icon_dict)
            icon_img = aug_icon_dict["image"]

            # Determine the scale for this icon (relative to background)
            scale_factor = random.uniform(min_scale, max_scale)
            target_width = int(bg_width * scale_factor)

            # Calculate the scaled height while preserving aspect ratio
            aspect_ratio = icon_img.height / icon_img.width
            target_height = int(target_width * aspect_ratio)

            # Resize the icon
            icon_img = icon_img.resize((target_width, target_height), Image.LANCZOS)

            # Try to find a suitable position without overlapping
            placed = False
            for _ in range(max_attempts):
                # Generate random position, ensuring icon is fully within background
                x = random.randint(0, bg_width - target_width)
                y = random.randint(0, bg_height - target_height)

                # Create a bounding box for collision detection
                new_bbox = box(x, y, x + target_width, y + target_height)

                # Check for overlap with existing placements
                overlaps = False
                for existing_bbox in bboxes:
                    if new_bbox.intersects(existing_bbox):
                        overlaps = True
                        break

                if not overlaps:
                    # No overlap, we can place the icon here
                    composite.paste(icon_img, (x, y), icon_img)

                    # Calculate YOLO format coordinates (midpoint x, midpoint y, width, height)
                    # All values normalized to [0, 1]
                    bbox_width = target_width / bg_width
                    bbox_height = target_height / bg_height
                    bbox_x = (x + target_width / 2) / bg_width
                    bbox_y = (y + target_height / 2) / bg_height

                    # Add the bounding box to our list for collision detection
                    bboxes.append(new_bbox)

                    # Record the icon placement
                    placed_icons.append(
                        {
                            "class_id": aug_icon_dict["class_id"],
                            "class_name": aug_icon_dict["class_name"],
                            "bbox": [bbox_x, bbox_y, bbox_width, bbox_height],
                            "x": x,
                            "y": y,
                            "width": target_width,
                            "height": target_height,
                        }
                    )

                    placed = True
                    break

            if not placed:
                print(f"Warning: Failed to place an icon after {max_attempts} attempts")

        return composite, placed_icons

    def generate_yolo_dataset(
        self,
        output_dir: str,
        backgrounds: List[Dict],
        icons: List[Dict],
        num_images: int = None,  # Parameter for controlling number of outputs
        icons_per_image: Tuple[int, int] = (0, 25),
        splits: Dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1},
        min_scale: float = 0.03,
        max_scale: float = 0.05,
        augmentation_strength: str = "medium",
    ):
        """Generate a YOLO dataset by placing icons on backgrounds.

        Args:
            output_dir (str): Directory to save the generated dataset
            backgrounds (List[Dict]): List of background image dictionaries
            icons (List[Dict]): List of icon dictionaries
            num_images (int, optional): Total number of images to generate. If None, use all backgrounds.
            icons_per_image (Tuple[int, int]): Range of number of icons to place on each background
            splits (Dict[str, float]): Dataset splits (train, val, test)
            min_scale (float): Minimum scale of icons relative to background
            max_scale (float): Maximum scale of icons relative to background
            augmentation_strength (str): Level of augmentation ('light', 'medium', 'strong')
        """
        import os

        # Setup the augmentation pipeline
        self.setup_augmentation(augmentation_strength)

        # Create output directories - adjusted for the required structure
        images_dir = os.path.join(output_dir, "images")
        labels_dir = os.path.join(output_dir, "labels")

        # Create the main image and label directories
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Create split directories
        for split in splits.keys():
            if (
                split == "test"
            ):  # Skip test directory as it's not in the required structure
                continue
            split_dir = os.path.join(output_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(split_dir, "labels"), exist_ok=True)

        # Create class mapping
        unique_classes = []
        class_id_map = {}

        # Create a clean class mapping (ensuring class IDs are sequential starting from 0)
        for icon in icons:
            class_name = icon["class_name"]
            if class_name not in class_id_map:
                class_id_map[class_name] = len(unique_classes)
                unique_classes.append(class_name)

        # Handle user-specified number of images
        prepared_backgrounds = backgrounds.copy()
        if num_images is not None:
            if num_images > len(backgrounds):
                print(
                    f"Requested {num_images} images but only {len(backgrounds)} backgrounds available."
                )
                print(
                    "Creating variations with different icon placements for each background..."
                )

                # Calculate how many variations of each background we need
                variations_needed = num_images // len(backgrounds) + 1

                # Create a new list with each background appearing multiple times
                # but each copy will get different random icon placements
                extended_backgrounds = []
                for _ in range(variations_needed):
                    # Shuffle to get different order each time
                    random.shuffle(backgrounds)
                    for bg in backgrounds:
                        if len(extended_backgrounds) < num_images:
                            # Create a deep copy to ensure we don't modify the original
                            bg_copy = bg.copy()
                            extended_backgrounds.append(bg_copy)

                prepared_backgrounds = extended_backgrounds[:num_images]
            else:
                # Use a random subset if we have more backgrounds than needed
                random.shuffle(prepared_backgrounds)
                prepared_backgrounds = prepared_backgrounds[:num_images]

        print(f"Generating dataset with {len(prepared_backgrounds)} images")

        # Shuffle backgrounds to ensure random distribution
        random.shuffle(prepared_backgrounds)

        # Calculate splits
        n_backgrounds = len(prepared_backgrounds)
        n_train = int(n_backgrounds * splits["train"])
        n_val = int(n_backgrounds * splits["val"])

        # Ensure at least one image in each split if we have enough backgrounds
        min_per_split = 1
        if n_backgrounds >= len(splits):
            if n_train < min_per_split and splits["train"] > 0:
                n_train = min_per_split
            if n_val < min_per_split and splits["val"] > 0:
                n_val = min_per_split

        # Assign backgrounds to splits
        train_backgrounds = prepared_backgrounds[:n_train]
        val_backgrounds = prepared_backgrounds[n_train : n_train + n_val]
        test_backgrounds = prepared_backgrounds[n_train + n_val :]

        split_assignments = {
            "train": train_backgrounds,
            "val": val_backgrounds,
            "test": test_backgrounds,
        }

        print(
            f"Dataset splits: Train {len(train_backgrounds)}, Val {len(val_backgrounds)}, Test {len(test_backgrounds)}"
        )

        # Create classes.txt file
        classes_file_path = os.path.join(output_dir, "classes.txt")
        with open(classes_file_path, "w") as f:
            for class_name in unique_classes:
                f.write(f"{class_name}\n")

        # Create data.yaml for Ultralytics
        data_yaml_path = os.path.join(output_dir, "data.yaml")
        with open(data_yaml_path, "w") as f:
            f.write(f"path: {os.path.abspath(output_dir)}\n")
            f.write("train: train/images\n")
            f.write("val: val/images\n")
            f.write("test: test/images\n\n")
            f.write(f"nc: {len(unique_classes)}\n")
            f.write(f"names: {unique_classes}\n")

        # Process each split
        for split, bg_list in split_assignments.items():
            print(f"Processing {split} split...")

            # Use tqdm to show progress
            for i, bg in enumerate(tqdm(bg_list, desc=f"Generating {split} images")):
                # Determine number of icons for this image
                num_icons = random.randint(icons_per_image[0], icons_per_image[1])

                # Update icons to use the sequential class IDs
                icons_with_sequential_ids = []
                for icon in icons:
                    icon_copy = icon.copy()
                    icon_copy["class_id"] = class_id_map[icon["class_name"]]
                    icons_with_sequential_ids.append(icon_copy)

                # Place icons on background
                composite, placed_icons = self.place_icons_on_background(
                    bg, icons_with_sequential_ids, num_icons, min_scale, max_scale
                )

                # Generate filename
                image_filename = f"{split}_{i:06d}.jpg"
                label_filename = f"{split}_{i:06d}.txt"

                # Save image to the common images directory
                image_path = os.path.join(images_dir, image_filename)
                composite.save(image_path, "JPEG", quality=95)

                # Create and save label file to the common labels directory
                label_path = os.path.join(labels_dir, label_filename)
                with open(label_path, "w") as f:
                    for icon in placed_icons:
                        # YOLO format: class_id x_center y_center width height
                        bbox = icon["bbox"]
                        line = f"{icon['class_id']} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                        f.write(line)

                # Create symbolic links for train and val splits (skip test as it's not in the required structure)
                if split in ["train", "val"]:
                    # Create symbolic links in the split-specific directories
                    split_image_link = os.path.join(
                        output_dir, split, "images", image_filename
                    )
                    split_label_link = os.path.join(
                        output_dir, split, "labels", label_filename
                    )

                    # Use relative paths for better portability
                    rel_image_path = os.path.relpath(
                        image_path, os.path.dirname(split_image_link)
                    )
                    rel_label_path = os.path.relpath(
                        label_path, os.path.dirname(split_label_link)
                    )

                    # Create the symbolic links
                    if os.path.exists(split_image_link):
                        os.remove(split_image_link)
                    if os.path.exists(split_label_link):
                        os.remove(split_label_link)

                    os.symlink(rel_image_path, split_image_link)
                    os.symlink(rel_label_path, split_label_link)

        print("Dataset generation complete!")
        print(f"Generated dataset saved to: {os.path.abspath(output_dir)}")
        print("Class mapping:")
        for class_name, class_id in class_id_map.items():
            print(f"  {class_id}: {class_name}")


def main():
    import argparse
    import os

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Generate synthetic object detection dataset in YOLO format"
    )

    # Required arguments
    parser.add_argument(
        "--icons", required=True, help="Path to the folder containing icon images"
    )
    parser.add_argument(
        "--backgrounds",
        required=True,
        help="Path to the folder containing background images",
    )
    parser.add_argument(
        "--output", required=True, help="Path to the output directory for the dataset"
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--num-images",
        type=int,
        help="Number of images to generate (default: use all backgrounds)",
    )
    parser.add_argument(
        "--min-icons",
        type=int,
        default=1,
        help="Minimum number of icons per image (default: 1)",
    )
    parser.add_argument(
        "--max-icons",
        type=int,
        default=5,
        help="Maximum number of icons per image (default: 5)",
    )
    parser.add_argument(
        "--min-scale",
        type=float,
        default=0.05,
        help="Minimum scale of icons relative to background (default: 0.05)",
    )
    parser.add_argument(
        "--max-scale",
        type=float,
        default=0.15,
        help="Maximum scale of icons relative to background (default: 0.15)",
    )
    parser.add_argument(
        "--augmentation",
        choices=["none", "light", "medium", "strong"],
        default="medium",
        help="Strength of icon augmentation (default: medium)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Proportion of images for training (default: 0.7)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Proportion of images for validation (default: 0.2)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Proportion of images for testing (default: 0.1)",
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducibility (default: None)"
    )
    parser.add_argument(
        "--min-icon-size",
        type=int,
        default=30,
        help="Minimum size for icons (default: 30)",
    )
    parser.add_argument(
        "--max-icon-size",
        type=int,
        default=200,
        help="Maximum size for icons (default: 200)",
    )
    parser.add_argument(
        "--augment-backgrounds",
        action="store_true",
        help="Augment backgrounds to increase variety",
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate arguments
    if args.min_icons > args.max_icons:
        parser.error("--min-icons cannot be greater than --max-icons")

    if args.min_scale > args.max_scale:
        parser.error("--min-scale cannot be greater than --max-scale")

    if args.train_split + args.val_split + args.test_split != 1.0:
        print(
            "Warning: Train, validation, and test splits do not sum to 1.0. They will be normalized."
        )
        total = args.train_split + args.val_split + args.test_split
        args.train_split /= total
        args.val_split /= total
        args.test_split /= total

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Initialize the generator
    generator = SyntheticDataGenerator(seed=args.seed)

    # Print configuration
    print("=== Synthetic Dataset Generation Configuration ===")
    print(f"Icons folder: {args.icons}")
    print(f"Backgrounds folder: {args.backgrounds}")
    print(f"Output directory: {args.output}")
    print(
        f"Number of images: {args.num_images if args.num_images else 'All backgrounds'}"
    )
    print(f"Icons per image: {args.min_icons} to {args.max_icons}")
    print(f"Icon scale: {args.min_scale} to {args.max_scale}")
    print(f"Augmentation strength: {args.augmentation}")
    print(
        f"Dataset splits: Train {args.train_split:.1%}, Val {args.val_split:.1%}, Test {args.test_split:.1%}"
    )
    print(f"Random seed: {args.seed if args.seed else 'Not set (random)'}")
    print(f"Augment backgrounds: {'Yes' if args.augment_backgrounds else 'No'}")
    print("=" * 45)

    # Load backgrounds
    backgrounds = generator.load_backgrounds(
        args.backgrounds,
        min_size=(640, 640),
        augment_backgrounds=args.augment_backgrounds,
        target_count=args.num_images,
    )

    # Load icons
    icons = generator.load_icons(
        args.icons, min_size=args.min_icon_size, max_size=args.max_icon_size
    )

    # Check if we have any icons and backgrounds
    if not icons:
        print("Error: No icons were loaded. Please check the icons folder path.")
        return

    if not backgrounds:
        print(
            "Error: No backgrounds were loaded. Please check the backgrounds folder path."
        )
        return

    # Generate the dataset
    generator.generate_yolo_dataset(
        output_dir=args.output,
        backgrounds=backgrounds,
        icons=icons,
        num_images=args.num_images,
        icons_per_image=(args.min_icons, args.max_icons),
        splits={
            "train": args.train_split,
            "val": args.val_split,
            "test": args.test_split,
        },
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        augmentation_strength=args.augmentation,
    )

    print(
        f"\nDataset generation complete. Output saved to: {os.path.abspath(args.output)}"
    )
    print(
        "You can now use this dataset with Ultralytics YOLOv5/YOLOv8 using the data.yaml file."
    )


if __name__ == "__main__":
    main()
