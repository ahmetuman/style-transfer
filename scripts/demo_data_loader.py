#!/usr/bin/env python3
"""
Data loader demonstration script.

This script showcases the complete functionality of our data_loader module,
demonstrating how to use it in different contexts (single image loading,
batch processing, paired vs. unpaired). It also provides visualization of 
results and performance benchmarks.

Usage:
    python demo_data_loader.py

Note: This script is created by LLM.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
import numpy as np

# Add project root to path
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(project_root))

# Import project modules
from src.data_loader import (
    StyleTransferDataset, get_dataloader, 
    AsyncImageLoader, denormalize_tensor
)
from src.config import CONTENT_DIR, STYLE_DIR, IMAGE_SIZE


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Data Loader Demonstration")
    parser.add_argument("--image_size", type=int, default=512, help="Target image size")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for demos")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--no_augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--paired", action="store_true", help="Use paired content-style images")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    return parser.parse_args()


def demonstrate_dataset_modes(args):
    """Demonstrate different dataset modes (paired vs unpaired)."""
    print("\n[1] Demonstrating dataset modes (paired vs unpaired)...")
    
    # Create datasets with different modes
    unpaired_dataset = StyleTransferDataset(
        content_dir=CONTENT_DIR,
        style_dir=STYLE_DIR,
        image_size=args.image_size,
        augment=not args.no_augment,
        paired=False,
        uniform_size=True
    )
    
    paired_dataset = StyleTransferDataset(
        content_dir=CONTENT_DIR,
        style_dir=STYLE_DIR,
        image_size=args.image_size,
        augment=not args.no_augment,
        paired=True,
        uniform_size=True
    )
    
    # Display size information
    print(f"Unpaired dataset size: {len(unpaired_dataset)}")
    print(f"Paired dataset size: {len(paired_dataset)}")
    
    # Show sample from each
    unpaired_sample = unpaired_dataset[0]
    paired_sample = paired_dataset[0]
    
    print(f"\nUnpaired sample:")
    print(f"  Content: {unpaired_sample['content_name']}")
    print(f"  Style: {unpaired_sample['style_name']}")
    
    print(f"\nPaired sample:")
    print(f"  Content: {paired_sample['content_name']}")
    print(f"  Style: {paired_sample['style_name']}")
    
    # Visualize samples
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Unpaired
    content_img = denormalize_tensor(unpaired_sample['content']).permute(1, 2, 0).cpu().numpy()
    style_img = denormalize_tensor(unpaired_sample['style']).permute(1, 2, 0).cpu().numpy()
    
    axes[0, 0].imshow(content_img)
    axes[0, 0].set_title(f"Unpaired Content: {unpaired_sample['content_name']}")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(style_img)
    axes[0, 1].set_title(f"Unpaired Style: {unpaired_sample['style_name']}")
    axes[0, 1].axis('off')
    
    # Paired
    content_img = denormalize_tensor(paired_sample['content']).permute(1, 2, 0).cpu().numpy()
    style_img = denormalize_tensor(paired_sample['style']).permute(1, 2, 0).cpu().numpy()
    
    axes[1, 0].imshow(content_img)
    axes[1, 0].set_title(f"Paired Content: {paired_sample['content_name']}")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(style_img)
    axes[1, 1].set_title(f"Paired Style: {paired_sample['style_name']}")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(project_root / "data" / "results" / "demo_dataset_modes.png")
    plt.show()


def demonstrate_batch_processing(args):
    """Demonstrate batch processing with DataLoader."""
    print("\n[2] Demonstrating batch processing...")
    
    dataloader = get_dataloader(
        content_dir=CONTENT_DIR,
        style_dir=STYLE_DIR,
        batch_size=args.batch_size,
        image_size=args.image_size,
        shuffle=True,
        augment=not args.no_augment,
        paired=args.paired,
        num_workers=args.num_workers,
        uniform_size=True
    )
    
    # Get a batch
    start_time = time.time()
    batch = next(iter(dataloader))
    load_time = time.time() - start_time
    
    content_batch = batch['content']
    style_batch = batch['style']
    
    print(f"Loaded batch of {args.batch_size} samples in {load_time:.3f} seconds")
    print(f"Content batch shape: {content_batch.shape}")
    print(f"Style batch shape: {style_batch.shape}")
    
    # Display content and style grid
    content_grid = vutils.make_grid(
        content_batch, nrow=2, normalize=True, scale_each=True
    )
    style_grid = vutils.make_grid(
        style_batch, nrow=2, normalize=True, scale_each=True
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(content_grid.permute(1, 2, 0).cpu())
    axes[0].set_title("Content Images")
    axes[0].axis('off')
    
    axes[1].imshow(style_grid.permute(1, 2, 0).cpu())
    axes[1].set_title("Style Images")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(project_root / "data" / "results" / "demo_batch_processing.png")
    plt.show()
    
    # Print batch metadata
    print("\nBatch metadata:")
    for i in range(min(3, args.batch_size)):  # Show metadata for first 3 samples
        print(f"  Sample {i+1}:")
        print(f"    Content: {batch['content_name'][i]}")
        print(f"    Style: {batch['style_name'][i]}")


def demonstrate_async_loading(args):
    """Demonstrate asynchronous image loading."""
    print("\n[3] Demonstrating asynchronous image loading...")
    
    # Get a few content images
    content_images = [f for f in Path(CONTENT_DIR).iterdir() if f.is_file()][:3]
    
    # Create async loader
    async_loader = AsyncImageLoader(
        image_size=args.image_size,
        uniform_size=True
    )
    
    # Synchronous loading for comparison
    sync_start = time.time()
    sync_results = []
    for img_path in content_images:
        sync_results.append(async_loader.load_image(img_path))
    sync_time = time.time() - sync_start
    
    # Asynchronous loading
    async_start = time.time()
    futures = []
    for img_path in content_images:
        futures.append(async_loader.load_image_async(img_path))
    
    # Wait for results
    async_results = [future.result() for future in futures]
    async_time = time.time() - async_start
    
    print(f"Synchronous loading time: {sync_time:.3f} seconds")
    print(f"Asynchronous loading time: {async_time:.3f} seconds")
    print(f"Speedup: {sync_time/async_time:.2f}x")
    
    # Visualize loaded images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, tensor in enumerate(async_results):
        img = denormalize_tensor(tensor.squeeze(0)).permute(1, 2, 0).cpu()
        axes[i].imshow(img)
        axes[i].set_title(f"Image {i+1}: {content_images[i].stem}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(project_root / "data" / "results" / "demo_async_loading.png")
    plt.show()


def demonstrate_augmentation_effects(args):
    """Demonstrate the effects of data augmentation."""
    print("\n[4] Demonstrating data augmentation effects...")
    
    # Create datasets with and without augmentation
    dataset_with_aug = StyleTransferDataset(
        content_dir=CONTENT_DIR,
        style_dir=STYLE_DIR,
        image_size=args.image_size,
        augment=True,
        uniform_size=True
    )
    
    dataset_no_aug = StyleTransferDataset(
        content_dir=CONTENT_DIR,
        style_dir=STYLE_DIR,
        image_size=args.image_size,
        augment=False,
        uniform_size=True
    )
    
    # Select a few content images to demonstrate augmentation
    num_samples = 3
    img_indices = np.random.choice(len(dataset_with_aug.content_images), num_samples, replace=False)
    
    # Create figure for visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i, idx in enumerate(img_indices):
        # Get original image path
        img_path = dataset_with_aug.content_images[idx]
        
        # Load original image
        original_img = dataset_no_aug.load_and_preprocess(img_path, is_content=True)
        original_img = denormalize_tensor(original_img).permute(1, 2, 0).cpu().numpy()
        
        # Get two different augmentations
        aug1 = dataset_with_aug.load_and_preprocess(img_path, is_content=True)
        aug1 = denormalize_tensor(aug1).permute(1, 2, 0).cpu().numpy()
        
        aug2 = dataset_with_aug.load_and_preprocess(img_path, is_content=True)
        aug2 = denormalize_tensor(aug2).permute(1, 2, 0).cpu().numpy()
        
        # Display images
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(f"Original: {img_path.stem}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(aug1)
        axes[i, 1].set_title(f"Augmentation 1")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(aug2)
        axes[i, 2].set_title(f"Augmentation 2")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(project_root / "data" / "results" / "demo_augmentation.png")
    plt.show()


def benchmark_performance(args):
    """Benchmark dataloader performance with different configurations."""
    print("\n[5] Benchmarking dataloader performance...")
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8]
    
    # Test different numbers of workers
    worker_counts = [0, 1, 2, 4]
    
    # Store results
    results = []
    
    # Run benchmarks
    for batch_size in batch_sizes:
        for num_workers in worker_counts:
            # Create dataloader
            dataloader = get_dataloader(
                content_dir=CONTENT_DIR,
                style_dir=STYLE_DIR,
                batch_size=batch_size,
                image_size=args.image_size,
                shuffle=True,
                augment=not args.no_augment,
                paired=args.paired,
                num_workers=num_workers,
                uniform_size=True
            )
            
            # Measure time to iterate through some batches
            start_time = time.time()
            batches_to_test = 5
            
            for i, _ in enumerate(dataloader):
                if i >= batches_to_test - 1:
                    break
            
            elapsed = time.time() - start_time
            avg_time_per_batch = elapsed / batches_to_test
            images_per_sec = batch_size / avg_time_per_batch
            
            results.append({
                'batch_size': batch_size,
                'num_workers': num_workers,
                'avg_time_per_batch': avg_time_per_batch,
                'images_per_sec': images_per_sec
            })
            
            print(f"Batch size: {batch_size}, Workers: {num_workers}, "
                  f"Time per batch: {avg_time_per_batch:.3f}s, "
                  f"Images/sec: {images_per_sec:.2f}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Group by batch size
    for batch_size in batch_sizes:
        batch_results = [r for r in results if r['batch_size'] == batch_size]
        workers = [r['num_workers'] for r in batch_results]
        times = [r['avg_time_per_batch'] for r in batch_results]
        axes[0].plot(workers, times, marker='o', label=f'Batch size {batch_size}')
    
    axes[0].set_xlabel('Number of Workers')
    axes[0].set_ylabel('Average Time per Batch (s)')
    axes[0].set_title('Loading Time vs. Worker Count')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot throughput (images/sec)
    for num_workers in worker_counts:
        worker_results = [r for r in results if r['num_workers'] == num_workers]
        batch_sizes_plot = [r['batch_size'] for r in worker_results]
        throughputs = [r['images_per_sec'] for r in worker_results]
        axes[1].plot(batch_sizes_plot, throughputs, marker='o', label=f'{num_workers} workers')
    
    axes[1].set_xlabel('Batch Size')
    axes[1].set_ylabel('Throughput (images/sec)')
    axes[1].set_title('Throughput vs. Batch Size')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(project_root / "data" / "results" / "dataloader_benchmark.png")
    plt.show()


def main():
    """Main function to run all demonstrations."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("STYLE TRANSFER DATA LOADER DEMONSTRATION")
    print("="*70)
    
    # Create results directory if it doesn't exist
    results_dir = project_root / "data" / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Run demonstrations
    demonstrate_dataset_modes(args)
    demonstrate_batch_processing(args)
    demonstrate_async_loading(args)
    demonstrate_augmentation_effects(args)
    
    # Run benchmarks if requested
    if args.benchmark:
        benchmark_performance(args)
    
    print("\n" + "="*70)
    print("DATA LOADER DEMONSTRATION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main() 