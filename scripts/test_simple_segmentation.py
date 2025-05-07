#!/usr/bin/env python3
"""
Test script for simple scene decomposition (sky/ground).

This script demonstrates a simpler approach to image segmentation based on
horizontal division of the image into sky (upper) and ground (lower) regions.
It's an alternative to deep learning-based segmentation when only basic
regions are needed.

Usage:
    python test_simple_segmentation.py [--content_image PATH] [--style_image PATH] 
                                      [--sky_ratio RATIO] [--output_dir DIR]
"""

import os
import sys
import argparse
from pathlib import Path
import random
from PIL import Image

# Add project root to Python path
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(project_root))

# Import project modules
from src.segmentation.simple_segmentation import (
    split_sky_ground,
    match_regions,
    visualize_simple_segmentation,
    visualize_region_matching
)
from src.config import CONTENT_DIR, STYLE_DIR


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test simple scene decomposition (sky/ground segmentation)")
    parser.add_argument("--content_image", type=str, help="Path to the content image")
    parser.add_argument("--style_image", type=str, help="Path to the style image")
    parser.add_argument("--sky_ratio", type=float, default=0.4, 
                       help="Ratio of image height to consider as sky (0-1)")
    parser.add_argument("--style_sky_ratio", type=float, default=None,
                       help="Separate sky ratio for style image (if different)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save visualizations")
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


def load_image(image_path):
    """Load an image while preserving its original size."""
    img = Image.open(image_path).convert('RGB')
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
    
    # Load images
    content_img = load_image(content_path)
    style_img = load_image(style_path)
    
    # Get style sky ratio (use content ratio if not specified)
    style_sky_ratio = args.style_sky_ratio if args.style_sky_ratio is not None else args.sky_ratio
    
    print("\n" + "="*70)
    print(f"SIMPLE SEGMENTATION TEST: {content_name} + {style_name}")
    print("="*70)
    print(f"Content sky ratio: {args.sky_ratio}")
    print(f"Style sky ratio: {style_sky_ratio}")
    
    # Segment content image
    print("\nSegmenting content image...")
    content_seg_path = output_dir / f"content_simple_segmentation_{content_name}.png"
    content_result = split_sky_ground(
        content_img, 
        sky_ratio=args.sky_ratio,
        save_path=str(content_seg_path)
    )
    print(f"Content segmentation saved to: {content_seg_path}")
    
    # Print content region info
    content_total = sum(content_result['class_counts'].values())
    print("\nContent image regions:")
    for region, count in content_result['class_counts'].items():
        percentage = count / content_total * 100
        print(f"  {region.capitalize()}: {percentage:.2f}% of image")
    
    # Segment style image
    print("\nSegmenting style image...")
    style_seg_path = output_dir / f"style_simple_segmentation_{style_name}.png"
    style_result = split_sky_ground(
        style_img, 
        sky_ratio=style_sky_ratio,
        save_path=str(style_seg_path)
    )
    print(f"Style segmentation saved to: {style_seg_path}")
    
    # Print style region info
    style_total = sum(style_result['class_counts'].values())
    print("\nStyle image regions:")
    for region, count in style_result['class_counts'].items():
        percentage = count / style_total * 100
        print(f"  {region.capitalize()}: {percentage:.2f}% of image")
    
    # Match regions
    print("\nMatching regions between content and style images...")
    matching_result = match_regions(content_result, style_result)
    
    # Save region matching visualization
    matching_path = output_dir / f"simple_region_matching_{content_name}_{style_name}.png"
    visualize_region_matching(
        content_img, 
        style_img, 
        matching_result,
        save_path=str(matching_path)
    )
    print(f"Region matching visualization saved to: {matching_path}")
    
    # Print matching information
    print("\nMatched regions:")
    for content_region, style_region in matching_result['matches'].items():
        content_percentage = matching_result['significant_content_classes'][content_region] * 100
        style_percentage = matching_result['significant_style_classes'][style_region] * 100
        
        print(f"  {content_region.capitalize()} → {style_region.capitalize()}")
        print(f"    Content: {content_percentage:.2f}% of image")
        print(f"    Style: {style_percentage:.2f}% of image")
    
    print("\nSimple segmentation test complete!")


if __name__ == "__main__":
    main() 