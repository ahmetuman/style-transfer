#!/usr/bin/env python3
"""
Test script for visualizing data augmentation effects.

This script loads a content image and applies each augmentation technique 
individually and in combination to demonstrate their effects. It's designed
to help understand the impact of different augmentations recommended by the
Texture Preserving Photo Style Transfer paper.

For demonstration purposes, the effects are slightly exaggerated compared
to the subtle parameters used in production.

Usage:
    python test_augmentations.py [--image_path IMAGE_PATH] [--size SIZE] [--subtle]
"""

import os
import sys
import argparse
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image

# Add project root to Python path
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(project_root))

# Import project modules
from src.data_loader import AddGaussianNoise, denormalize_tensor
from src.config import IMAGE_SIZE, CONTENT_DIR


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test data augmentation effects")
    parser.add_argument("--image_path", type=str, help="Path to the image to augment")
    parser.add_argument("--size", type=int, default=IMAGE_SIZE, help="Target image size")
    parser.add_argument("--subtle", action="store_true", 
                       help="Use subtle parameters as in production (default: use exaggerated for demonstration)")
    return parser.parse_args()


def get_default_image(size):
    """Get a default image if no path is provided."""
    content_dir = Path(CONTENT_DIR)
    image_files = [f for f in content_dir.iterdir() if f.is_file() and 
                  f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {content_dir}")
    
    # Choose a random image
    img_path = random.choice(image_files)
    print(f"Using image: {img_path}")
    
    # Load and resize the image
    img = Image.open(img_path).convert('RGB')
    img = img.resize((size, size), Image.LANCZOS)
    
    return img, img_path.name


def apply_transforms_and_show(original_img, transforms_dict):
    """Apply different transforms and show the results."""
    fig_rows = len(transforms_dict) + 1
    fig, axes = plt.subplots(fig_rows, 3, figsize=(15, 5 * fig_rows))
    
    # Display original image in first row
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Apply no transformation to the other cells in first row
    for i in range(1, 3):
        axes[0, i].imshow(original_img)
        axes[0, i].set_title("Original Image (Copy)")
        axes[0, i].axis('off')
    
    # Apply each transform and show 3 examples
    for i, (transform_name, transform) in enumerate(transforms_dict.items(), 1):
        for j in range(3):
            # Apply the transform
            augmented_img = transform(original_img)
            
            # Show the result
            axes[i, j].imshow(augmented_img)
            axes[i, j].set_title(f"{transform_name} (Example {j+1})")
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(project_root / "data" / "results" / "augmentation_effects.png", dpi=200)
    plt.show()
    
    # Create a before-after comparison for combined augmentations
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    combined_img = transforms_dict["Combined Augmentations"](original_img)
    plt.imshow(combined_img)
    plt.title("Combined Augmentations")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(project_root / "data" / "results" / "augmentation_before_after.png", dpi=200)
    plt.show()


def create_individual_transforms(image_size, subtle=False):
    """Create a dictionary of individual transformation techniques."""
    
    # Production parameters (subtle)
    if subtle:
        flip_prob = 0.5
        crop_scale = (0.95, 1.0)
        crop_ratio = (0.95, 1.05)
        translate = (0.05, 0.05)
        scale = (0.98, 1.02)
        brightness = 0.05
        contrast = 0.05
        saturation = 0.05
        hue = 0.02
        noise_std = 0.015
    # Demonstration parameters (exaggerated)
    else:
        flip_prob = 1.0  # Always flip for demo
        crop_scale = (0.75, 0.95)  # More significant crop
        crop_ratio = (0.9, 1.1)  # More aspect ratio change
        translate = (0.2, 0.2)  # More translation
        scale = (0.8, 1.2)  # More scaling
        brightness = 0.3  # More brightness variation
        contrast = 0.4  # More contrast variation
        saturation = 0.4  # More saturation variation
        hue = 0.1  # More hue variation
        noise_std = 0.08  # More visible noise
    
    return {
        # 1. Random Horizontal Flip
        "Random Horizontal Flip": T.Compose([
            T.RandomHorizontalFlip(p=1.0)  # Always flip for demonstration
        ]),
        
        # 2. Random Resized Crop
        "Random Resized Crop": T.Compose([
            T.RandomResizedCrop(
                size=(image_size, image_size),
                scale=crop_scale,
                ratio=crop_ratio,
                antialias=True
            )
        ]),
        
        # 3. Random Affine (Translation)
        "Random Translation": T.Compose([
            T.RandomAffine(
                degrees=0,
                translate=translate,
                scale=(1.0, 1.0)  # No scaling
            )
        ]),
        
        # 4. Random Affine (Scale)
        "Random Scaling": T.Compose([
            T.RandomAffine(
                degrees=0,
                translate=(0.0, 0.0),  # No translation
                scale=scale
            )
        ]),
        
        # 5. Color Jitter (Brightness)
        "Brightness Jitter": T.Compose([
            T.ColorJitter(brightness=brightness, contrast=0, saturation=0, hue=0)
        ]),
        
        # 6. Color Jitter (Contrast)
        "Contrast Jitter": T.Compose([
            T.ColorJitter(brightness=0, contrast=contrast, saturation=0, hue=0)
        ]),
        
        # 7. Color Jitter (Saturation)
        "Saturation Jitter": T.Compose([
            T.ColorJitter(brightness=0, contrast=0, saturation=saturation, hue=0)
        ]),
        
        # 8. Color Jitter (Hue)
        "Hue Jitter": T.Compose([
            T.ColorJitter(brightness=0, contrast=0, saturation=0, hue=hue)
        ]),
        
        # 9. Gaussian Noise
        "Gaussian Noise": AddGaussianNoise(mean=0, std=noise_std),
        
        # 10. All Combined (as in our data_loader.py)
        "Combined Augmentations": T.Compose([
            T.RandomHorizontalFlip(p=flip_prob),
            T.RandomResizedCrop(
                size=(image_size, image_size),
                scale=crop_scale,
                ratio=crop_ratio,
                antialias=True
            ),
            T.RandomAffine(
                degrees=0,
                translate=translate,
                scale=scale
            ),
            T.ColorJitter(
                brightness=brightness, 
                contrast=contrast, 
                saturation=saturation,
                hue=hue
            ),
            AddGaussianNoise(mean=0, std=noise_std)
        ])
    }


def create_comparison_grid(original_img, subtle_transforms, exaggerated_transforms):
    """Create a grid comparing subtle and exaggerated transformations."""
    transform_names = [
        "Random Resized Crop", 
        "Random Translation", 
        "Brightness Jitter", 
        "Gaussian Noise",
        "Combined Augmentations"
    ]
    
    # For each transformation type, create a separate figure
    for name in transform_names:
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_img)
        plt.title("Original")
        plt.axis('off')
        
        # Subtle transformation (production)
        subtle_img = subtle_transforms[name](original_img)
        plt.subplot(1, 3, 2)
        plt.imshow(subtle_img)
        plt.title(f"{name} (Subtle/Production)")
        plt.axis('off')
        
        # Exaggerated transformation (demo)
        exaggerated_img = exaggerated_transforms[name](original_img)
        plt.subplot(1, 3, 3)
        plt.imshow(exaggerated_img)
        plt.title(f"{name} (Exaggerated/Demo)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(project_root / "data" / "results" / f"comparison_{name.lower().replace(' ', '_')}.png", dpi=200)
        plt.show()


def main():
    """Main function."""
    args = parse_args()
    
    # Create results directory if it doesn't exist
    results_dir = project_root / "data" / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Get image to augment
    if args.image_path:
        original_img = Image.open(args.image_path).convert('RGB')
        original_img = original_img.resize((args.size, args.size), Image.LANCZOS)
        img_name = Path(args.image_path).name
    else:
        original_img, img_name = get_default_image(args.size)
    
    print("\n" + "="*70)
    print(f"DATA AUGMENTATION VISUALIZATION FOR: {img_name}")
    print("="*70)
    
    if args.subtle:
        print("\nUsing subtle parameters (as in production)")
        transforms_dict = create_individual_transforms(args.size, subtle=True)
        apply_transforms_and_show(original_img, transforms_dict)
    else:
        print("\nUsing exaggerated parameters (for demonstration)")
        transforms_dict = create_individual_transforms(args.size, subtle=False)
        apply_transforms_and_show(original_img, transforms_dict)
        
        # Also create a comparison between subtle and exaggerated
        print("\nCreating comparison between subtle and exaggerated parameters...")
        subtle_transforms = create_individual_transforms(args.size, subtle=True)
        create_comparison_grid(original_img, subtle_transforms, transforms_dict)
    
    print("\nAugmentation visualization complete. Results saved to:")
    print(f"{results_dir}/augmentation_effects.png")
    print(f"{results_dir}/augmentation_before_after.png")
    if not args.subtle:
        print("Individual comparison images saved to results directory.")


if __name__ == "__main__":
    main() 