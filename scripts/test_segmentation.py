#!/usr/bin/env python3
"""
Test script for semantic segmentation and region matching.

This script demonstrates the semantic segmentation capabilities using DeepLabV3
for both content and style images. It visualizes the segmentation results 
and matches corresponding semantic regions between the images.

Usage:
    python test_segmentation.py [--content_image PATH] [--style_image PATH] 
                               [--size SIZE] [--threshold THRESHOLD]
                               [--output_dir DIR]
"""

import os
import sys
import argparse
from pathlib import Path
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to Python path
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(project_root))

# Import project modules
from src.segmentation.segmentation_model import (
    SegmentationModel,
    SemanticMatcher,
    visualize_segmentation,
    visualize_region_matching
)
from src.config import IMAGE_SIZE, CONTENT_DIR, STYLE_DIR


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test semantic segmentation and region matching")
    parser.add_argument("--content_image", type=str, help="Path to the content image")
    parser.add_argument("--style_image", type=str, help="Path to the style image")
    parser.add_argument("--size", type=int, default=IMAGE_SIZE, help="Target image size")
    parser.add_argument("--threshold", type=float, default=0.01, 
                       help="Minimum percentage of pixels to consider a class present (0-1)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save visualizations")
    parser.add_argument("--model", type=str, default="deeplabv3_resnet101",
                       choices=["deeplabv3_resnet101", "deeplabv3_resnet50", "fcn_resnet101"],
                       help="Segmentation model architecture to use")
    return parser.parse_args()


def get_default_images():
    """Get default content and style images if no paths are provided."""
    content_dir = Path(CONTENT_DIR)
    style_dir = Path(STYLE_DIR)
    
    content_files = list(content_dir.glob("*.jpg")) + list(content_dir.glob("*.png"))
    style_files = list(style_dir.glob("*.jpg")) + list(style_dir.glob("*.png"))
    
    if not content_files:
        raise FileNotFoundError(f"No images found in {content_dir}")
    if not style_files:
        raise FileNotFoundError(f"No images found in {style_dir}")
    
    # Choose a random content and style image
    content_img = random.choice(content_files)
    style_img = random.choice(style_files)
    
    return content_img, style_img


def load_resize_image(image_path, size):
    """Load and resize an image while preserving aspect ratio."""
    img = Image.open(image_path).convert('RGB')
    
    # Resize while preserving aspect ratio
    width, height = img.size
    if width > height:
        new_width = size
        new_height = int(height * size / width)
    else:
        new_height = size
        new_width = int(width * size / height)
    
    img = img.resize((new_width, new_height), Image.LANCZOS)
    return img


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory if specified
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    else:
        # Default to results directory
        output_dir = project_root / "data" / "results"
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get content and style images
    if args.content_image and args.style_image:
        content_path = Path(args.content_image)
        style_path = Path(args.style_image)
        content_name = content_path.stem
        style_name = style_path.stem
    else:
        content_path, style_path = get_default_images()
        content_name = content_path.stem
        style_name = style_path.stem
        print(f"Using random content image: {content_path}")
        print(f"Using random style image: {style_path}")
    
    # Load and resize images
    content_img = load_resize_image(content_path, args.size)
    style_img = load_resize_image(style_path, args.size)
    
    print("\n" + "="*70)
    print(f"SEMANTIC SEGMENTATION TEST: {content_name} + {style_name}")
    print("="*70)
    
    # Try different models if default model fails
    models_to_try = [args.model]
    if args.model == "deeplabv3_resnet101":
        models_to_try += ["fcn_resnet101", "deeplabv3_resnet50"]
    
    success = False
    for model_name in models_to_try:
        try:
            # Initialize segmentation model
            print(f"\nInitializing {model_name}...")
            model = SegmentationModel(pretrained=True, model_name=model_name)
            
            # Segment content image
            print("\nSegmenting content image...")
            content_result = model.segment(content_img)
            
            # Debug info
            print(f"Content image size: {content_img.size[0]}x{content_img.size[1]}")
            print(f"Content mask shape: {content_result['mask'].shape}")
            
            # Save segmentation visualization
            content_seg_path = output_dir / f"content_segmentation_{content_name}.png"
            visualize_segmentation(
                content_img, 
                content_result, 
                save_path=str(content_seg_path)
            )
            print(f"Content segmentation saved to: {content_seg_path}")
            
            # Print content classes
            print("\nContent image classes:")
            for cls, count in sorted(content_result['class_counts'].items()):
                percentage = count / content_img.size[0] / content_img.size[1] * 100
                if cls in content_result['class_names']:
                    class_name = content_result['class_names'][cls]
                    print(f"  Class {cls}: {class_name} - {percentage:.2f}% of image")
                else:
                    print(f"  Class {cls} - {percentage:.2f}% of image")
            
            # Check if we got more than just background
            if len(content_result['class_counts']) > 1 or 0 not in content_result['class_counts']:
                # We found something other than background!
                success = True
                break
                
            print("\nModel only found background. Trying next model if available...")
            
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            print("Trying next model if available...")
            continue
    
    if not success and len(models_to_try) > 1:
        print(f"\nAll {len(models_to_try)} models failed to find non-background classes.")
        print("Proceeding with last model attempted.")
    
    # Segment style image
    print("\nSegmenting style image...")
    style_result = model.segment(style_img)
    
    # Debug info
    print(f"Style image size: {style_img.size[0]}x{style_img.size[1]}")
    print(f"Style mask shape: {style_result['mask'].shape}")
    
    # Save segmentation visualization
    style_seg_path = output_dir / f"style_segmentation_{style_name}.png"
    visualize_segmentation(
        style_img, 
        style_result, 
        save_path=str(style_seg_path)
    )
    print(f"Style segmentation saved to: {style_seg_path}")
    
    # Print style classes
    print("\nStyle image classes:")
    for cls, count in sorted(style_result['class_counts'].items()):
        percentage = count / style_img.size[0] / style_img.size[1] * 100
        if cls in style_result['class_names']:
            class_name = style_result['class_names'][cls]
            print(f"  Class {cls}: {class_name} - {percentage:.2f}% of image")
        else:
            print(f"  Class {cls} - {percentage:.2f}% of image")
    
    # Match semantic regions
    print("\nMatching semantic regions between content and style images...")
    matcher = SemanticMatcher(model)
    matching_result = matcher.match_regions(
        content_img, 
        style_img,
        threshold_percentage=args.threshold
    )
    
    # Save region matching visualization
    matching_path = output_dir / f"region_matching_{content_name}_{style_name}.png"
    visualize_region_matching(
        content_img, 
        style_img, 
        matching_result,
        save_path=str(matching_path)
    )
    print(f"Region matching visualization saved to: {matching_path}")
    
    # Print matching information
    print("\nMatched regions:")
    if len(matching_result['matches']) == 0:
        print("  No matching regions found. Try lowering the threshold.")
    else:
        for content_cls, style_cls in matching_result['matches'].items():
            content_percentage = matching_result['significant_content_classes'][content_cls] * 100
            style_percentage = matching_result['significant_style_classes'][style_cls] * 100
            
            if content_cls in content_result['class_names']:
                class_name = content_result['class_names'][content_cls]
                print(f"  Class {content_cls}: {class_name}")
                print(f"    Content: {content_percentage:.2f}% of image")
                print(f"    Style: {style_percentage:.2f}% of image")
            else:
                print(f"  Class {content_cls}")
                print(f"    Content: {content_percentage:.2f}% of image")
                print(f"    Style: {style_percentage:.2f}% of image")
    
    print("\nSegmentation test complete!")


if __name__ == "__main__":
    main() 