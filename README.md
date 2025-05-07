# style-transfer

Photorealistic Style and Lighting Transfer using Gradient-Domain Optimization and Edge-Aware Filtering

## Overview

This project implements a photorealistic style transfer system that combines semantic segmentation, depth estimation, gradient-domain editing, and edge-aware filtering to produce high-quality, photorealistic style transfers between images.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Pipeline steps:
   - Content Image + Style Image
   - Semantic Segmentation (DeepLab)
   - Depth Estimation (MiDaS) 
   - Gradient Editing for Lighting (basic Laplacian blending)
   - Edge-Aware Smoothing (Guided Filter)
   - Controlled Style Transfer (AdaIN limited by mask etc.)
   - Output Photo

## Implementation Progress

### Completed

- **Data Loading and Preprocessing**
  - Image loading with aspect ratio preservation
  - Advanced data augmentation based on photorealistic style transfer research
  - Efficient batch processing with uniform resizing
  - Asynchronous image loading support for real-time applications
  - M1/M2 Mac optimization with Metal Performance Shaders (MPS)

### In Progress

- Semantic Segmentation
- Depth Estimation
- Gradient Domain Editing
- Edge-Aware Filtering
- Style Transfer

## Data Augmentation

We've implemented photography-inspired data augmentation techniques to enhance the robustness of our style transfer:

- **Random Horizontal Flip**: Simulates different camera angles
- **Subtle Random Resized Crop**: Simulates framing variations
- **Small Affine Transformations**: Simulates small camera movements
- **Color Jittering**: Simulates lighting and exposure variations
- **Gaussian Noise**: Simulates sensor noise and image imperfections

For detailed documentation about our augmentation approach, see:
- [Data Augmentation Documentation](docs/data_augmentation.md) - Implementation details and philosophy
- [Data Augmentation Analysis](docs/augmentation_analysis.md) - Analysis of subtle vs. exaggerated parameters

### Demonstration and Visualization

To visualize the effects of different augmentation techniques:

```bash
# View subtle augmentations (as used in production)
python scripts/test_augmentations.py --subtle

# View exaggerated augmentations (for demonstration purposes)
python scripts/test_augmentations.py
```

The visualizations help illustrate why we've chosen subtle parameters for our augmentations, ensuring they enhance robustness while preserving photorealism.

## Usage

### Testing the Data Loader

Run the basic test script to verify the data loader functionality:

```bash
python scripts/test_data_loader.py
```

### Data Loader Demonstration

To see all features of the data loader in action:

```bash
python scripts/demo_data_loader.py
```

Additional options:
- `--image_size SIZE`: Set custom image size (default: 512)
- `--batch_size SIZE`: Set batch size for testing (default: 4)
- `--num_workers NUM`: Set number of worker processes (default: 2)
- `--no_augment`: Disable data augmentation
- `--paired`: Use paired content-style images
- `--benchmark`: Run performance benchmarks for different configurations

### Testing Data Augmentation

To visualize the effects of different augmentation techniques:

```bash
python scripts/test_augmentations.py
```

Options:
- `--image_path PATH`: Specify a custom image to augment
- `--size SIZE`: Set the target image size (default: 512)

## Directory Structure

- `src/`: Source code
  - `data_loader.py`: Image loading and preprocessing
  - `segmentation/`: Semantic segmentation models
  - `depth/`: Depth estimation models
  - `gradient_domain/`: Gradient domain manipulation
  - `edge_aware/`: Edge-aware filtering
  - `style_transfer/`: Neural style transfer
  - `utils/`: Utility functions
- `scripts/`: Helper scripts for testing and demonstration
- `data/`: Data directory
  - `content/`: Content images
  - `style/`: Style images
  - `results/`: Output results
- `models/`: Pre-trained model weights
- `notebooks/`: Jupyter notebooks for experimentation
- `docs/`: Documentation files

## Notes for M1/M2 Mac Users

This project includes optimizations for Apple Silicon (M1/M2) using PyTorch's Metal Performance Shaders (MPS). To ensure optimal performance:

1. Install PyTorch with MPS support:
   ```
   pip install torch torchvision torchaudio
   ```

2. The data loader automatically detects MPS availability and adjusts worker count accordingly.

3. For best performance on M1/M2, use batch sizes between 2-8 with 1-2 worker processes.
