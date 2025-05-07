#!/usr/bin/env python3
"""
Test script for the data_loader module.

This script tests the functionality of the data_loader module by loading sample content and
style images and visualizing them. It's useful for verifying that our data pipeline works
correctly before implementing the style transfer components.

Usage:
    python test_data_loader.py [--paired] [--num_samples NUM_SAMPLES]

Note: This script is created by LLM.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(project_root))

# Import project modules
from src.data_loader import (
    StyleTransferDataset, get_dataloader, 
    inspect_dataset, visualize_sample_pairs, AsyncImageLoader
)
from src.config import CONTENT_DIR, STYLE_DIR, IMAGE_SIZE


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test data loader for style transfer")
    parser.add_argument("--paired", action="store_true", help="Use paired content and style images")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to visualize")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE, help="Image size for processing")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for dataloader")
    return parser.parse_args()


def main():
    """Main function to test the data loader."""
    args = parse_args()
    
    print("\n" + "="*50)
    print("TESTING STYLE TRANSFER DATA LOADER")
    print("="*50)
    
    # Create dataset
    print("\nCreating StyleTransferDataset...")
    dataset = StyleTransferDataset(
        content_dir=CONTENT_DIR,
        style_dir=STYLE_DIR,
        image_size=args.image_size,
        augment=True,
        paired=args.paired
    )
    
    # Inspect dataset statistics
    print("\nInspecting dataset statistics:")
    inspect_dataset(dataset, num_samples=args.num_samples)
    
    # Visualize sample pairs
    print("\nVisualizing sample pairs:")
    visualize_sample_pairs(dataset, num_pairs=args.num_samples)
    
    # Test data loader
    print("\nTesting DataLoader...")
    dataloader = get_dataloader(
        content_dir=CONTENT_DIR,
        style_dir=STYLE_DIR,
        batch_size=args.batch_size,
        image_size=args.image_size,
        shuffle=True,
        augment=True,
        paired=args.paired
    )
    
    # Verify dataloader iteration
    batch = next(iter(dataloader))
    print(f"Successfully loaded batch with {args.batch_size} samples")
    print(f"Content batch shape: {batch['content'].shape}")
    print(f"Style batch shape: {batch['style'].shape}")
    
    # Test async image loader
    print("\nTesting AsyncImageLoader...")
    async_loader = AsyncImageLoader(image_size=args.image_size)
    
    # Get first content image path
    first_content_path = dataset.content_images[0]
    print(f"Loading image from: {first_content_path}")
    
    # Load image synchronously
    img_tensor = async_loader.load_image(first_content_path)
    print(f"Loaded image tensor shape: {img_tensor.shape}")
    
    # Load image asynchronously
    future = async_loader.load_image_async(first_content_path)
    async_img_tensor = future.result()
    print(f"Asynchronously loaded image tensor shape: {async_img_tensor.shape}")
    
    print("\n" + "="*50)
    print("DATA LOADER TESTS COMPLETED SUCCESSFULLY")
    print("="*50 + "\n")


if __name__ == "__main__":
    main() 