import os
import random
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union, Any

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Ensure Metal performance on M1 Mac
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) on M1 Pro")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Import project configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    CONTENT_DIR, STYLE_DIR, IMAGE_SIZE, MAX_IMAGE_SIZE
)


class StyleTransferDataset(Dataset):
    """
    Args:
        content_dir (str or Path): Directory containing content images
        style_dir (str or Path): Directory containing style images
        image_size (int): Target size for image processing
        max_size (int): Maximum image dimension to prevent memory issues
        augment (bool): Whether to apply data augmentation
        paired (bool): Whether content and style images are paired by filename
        style_count_multiplier (int): How many times to repeat each style image
        uniform_size (bool): Whether to resize all images to exact dimensions
    """
    
    def __init__(
        self,
        content_dir: Union[str, Path] = CONTENT_DIR,
        style_dir: Union[str, Path] = STYLE_DIR,
        image_size: int = IMAGE_SIZE,
        max_size: int = MAX_IMAGE_SIZE,
        augment: bool = True,
        paired: bool = False,
        style_count_multiplier: int = 1,
        uniform_size: bool = True
    ):
        self.content_dir = Path(content_dir)
        self.style_dir = Path(style_dir)
        self.image_size = image_size
        self.max_size = max_size
        self.augment = augment
        self.paired = paired
        self.uniform_size = uniform_size
        
        self.content_images = self._get_image_paths(self.content_dir)
        self.style_images = self._get_image_paths(self.style_dir)
        
        if self.paired:
            min_len = min(len(self.content_images), len(self.style_images))
            self.content_images = self.content_images[:min_len]
            self.style_images = self.style_images[:min_len]
        else:
            self.style_images = self.style_images * style_count_multiplier
        
        self.transform = self._get_transform()
        self.augmentation = self._get_augmentation() if augment else None
        
        print(f"Dataset initialized with {len(self)} image pairs")
        print(f"Content images: {len(self.content_images)}")
        print(f"Style images: {len(self.style_images)}")
    
    def _get_image_paths(self, dir_path: Path) -> List[Path]:
        """Get paths of all images in directory with valid extensions."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        return [
            f for f in dir_path.iterdir() 
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]
    
    def _get_transform(self) -> T.Compose:
        """Define standard transformations for style transfer."""
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_augmentation(self) -> T.Compose:
        """Define augmentation transformations for content images."""
        return T.Compose([
            # Random horizontal flip (50% chance)
            T.RandomHorizontalFlip(p=0.5),
            
            # Small random resized crop (subtle to preserve structure)
            T.RandomResizedCrop(
                size=(self.image_size, self.image_size),
                scale=(0.95, 1.0),  # Only crop 0-5% of the image
                ratio=(0.95, 1.05),  # Keep aspect ratio nearly the same
                antialias=True
            ),
            
            # Small random affine transforms (translation)
            T.RandomAffine(
                degrees=0,  # No rotation to preserve structure
                translate=(0.05, 0.05),  # Reduced from 0.1 to be more subtle
                scale=(0.98, 1.02)  # More subtle scaling
            ),
            
            # Subtle color jittering
            T.ColorJitter(
                brightness=0.05, 
                contrast=0.05, 
                saturation=0.05,
                hue=0.02  # Added subtle hue variation
            ),
            
            # Add subtle Gaussian noise
            AddGaussianNoise(mean=0., std=0.015)
        ])
    
    def __len__(self) -> int:
        """Get dataset length based on content images."""
        if self.paired:
            return len(self.content_images)
        else:
            return len(self.content_images) * len(self.style_images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a pair of content and style images.
        
        For paired mode, content and style images with same index are returned.
        For unpaired mode, combinations of content and style images are created.
        
        Args:
            idx (int): Index of the pair to retrieve
            
        Returns:
            dict: Dictionary containing content and style tensors and metadata
        """
        if self.paired:
            content_idx = style_idx = idx
        else:
            content_idx = idx % len(self.content_images)
            style_idx = (idx // len(self.content_images)) % len(self.style_images)
        
        # Load images
        content_img = self.load_and_preprocess(self.content_images[content_idx], is_content=True)
        style_img = self.load_and_preprocess(self.style_images[style_idx], is_content=False)
        
        # Extract filenames without extension for metadata
        content_name = self.content_images[content_idx].stem
        style_name = self.style_images[style_idx].stem
        
        return {
            'content': content_img,
            'style': style_img,
            'content_path': str(self.content_images[content_idx]),
            'style_path': str(self.style_images[style_idx]),
            'content_name': content_name,
            'style_name': style_name
        }
    
    def load_and_preprocess(self, img_path: Path, is_content: bool = True) -> torch.Tensor:
        """
        Load and preprocess an image for style transfer.
        
        Args:
            img_path (Path): Path to the image
            is_content (bool): Whether this is a content image (affects augmentation)
            
        Returns:
            torch.Tensor: Processed image tensor
        """
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Resize image based on settings
        if self.uniform_size:
            # For batch processing, we need uniform dimensions
            # Resize to exact dimensions, potentially distorting aspect ratio
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        else:
            # For single image processing, preserve aspect ratio
            img = self.resize_preserve_aspect_ratio(img, self.image_size, self.max_size)
        
        # Apply augmentation to content images if enabled
        if self.augment and is_content:
            img = self.augmentation(img)
        
        # Apply normalization and convert to tensor
        img_tensor = self.transform(img)
        
        return img_tensor
    
    @staticmethod
    def resize_preserve_aspect_ratio(img: Image.Image, target_size: int, max_size: int) -> Image.Image:
        """
        Resize image preserving aspect ratio.
        
        Args:
            img (PIL.Image): Input image
            target_size (int): Target size for smaller dimension
            max_size (int): Maximum allowed size for larger dimension
            
        Returns:
            PIL.Image: Resized image
        """
        width, height = img.size
        
        # Determine which dimension to match to target_size
        if width < height:
            # Width is smaller, so set it to target_size
            new_width = target_size
            new_height = int(height * target_size / width)
            
            # Check if height exceeds max_size
            if new_height > max_size:
                new_height = max_size
                new_width = int(width * max_size / height)
        else:
            # Height is smaller, so set it to target_size
            new_height = target_size
            new_width = int(width * target_size / height)
            
            # Check if width exceeds max_size
            if new_width > max_size:
                new_width = max_size
                new_height = int(height * max_size / width)
        
        # Resize the image
        return img.resize((new_width, new_height), Image.LANCZOS)


def get_dataloader(
    content_dir: Union[str, Path] = CONTENT_DIR,
    style_dir: Union[str, Path] = STYLE_DIR,
    batch_size: int = 1,
    image_size: int = IMAGE_SIZE,
    shuffle: bool = True,
    augment: bool = True,
    paired: bool = False,
    num_workers: int = 4,
    uniform_size: bool = True
) -> DataLoader:
    """
    Create a DataLoader for style transfer.
    
    Args:
        content_dir (str or Path): Directory of content images
        style_dir (str or Path): Directory of style images
        batch_size (int): Batch size
        image_size (int): Target image size
        shuffle (bool): Whether to shuffle the dataset
        augment (bool): Whether to apply data augmentation
        paired (bool): Whether content and style images are paired
        num_workers (int): Number of worker processes
        uniform_size (bool): Whether to use uniform image sizes for batching
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for style transfer
    """
    # For M1 Pro, optimal workers is usually 2-4 depending on workload
    # Set to 0 if encountering MPS-related issues
    if device.type == 'mps':
        num_workers = min(2, num_workers)  # Reduce workers for MPS
    
    dataset = StyleTransferDataset(
        content_dir=content_dir,
        style_dir=style_dir,
        image_size=image_size,
        augment=augment,
        paired=paired,
        uniform_size=uniform_size
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if device.type != 'cpu' else False
    )


class AsyncImageLoader:
    """
    Asynchronous image loader for real-time processing.
    
    This class supports loading and preprocessing images in a separate thread,
    which is useful for real-time processing pipelines.
    
    Args:
        image_size (int): Target image size
        max_size (int): Maximum allowed dimension
        device (torch.device): Device to load tensors to
        uniform_size (bool): Whether to resize images to exact dimensions
    """
    
    def __init__(
        self,
        image_size: int = IMAGE_SIZE,
        max_size: int = MAX_IMAGE_SIZE,
        device: torch.device = device,
        uniform_size: bool = True
    ):
        self.image_size = image_size
        self.max_size = max_size
        self.device = device
        self.uniform_size = uniform_size
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def load_image(self, img_path: Union[str, Path]) -> torch.Tensor:
        """
        Load and preprocess image synchronously.
        
        Args:
            img_path (str or Path): Path to image
            
        Returns:
            torch.Tensor: Processed image tensor on target device
        """
        img = Image.open(img_path).convert('RGB')
        
        if self.uniform_size:
            # Use exact dimensions for batch processing
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        else:
            # Preserve aspect ratio for single image processing
            img = StyleTransferDataset.resize_preserve_aspect_ratio(
                img, self.image_size, self.max_size
            )
            
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        return img_tensor
    
    def load_image_async(self, img_path: Union[str, Path]) -> Any:
        """
        Load and preprocess image asynchronously.
        
        Args:
            img_path (str or Path): Path to image
            
        Returns:
            concurrent.futures.Future: Future object for the loaded tensor
        """
        return self.executor.submit(self.load_image, img_path)


# Utility functions for dataset inspection and visualization

def inspect_dataset(dataset: StyleTransferDataset, num_samples: int = 5) -> None:
    """
    Inspect dataset by displaying image statistics and sample images.
    
    Args:
        dataset (StyleTransferDataset): Dataset to inspect
        num_samples (int): Number of samples to display
    """
    import matplotlib.pyplot as plt
    
    print(f"Dataset contains {len(dataset)} samples")
    
    # Get sample indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        content = sample['content']
        style = sample['style']
        
        print(f"\nSample {i+1}:")
        print(f"Content: {sample['content_name']} - Shape: {content.shape}")
        print(f"Style: {sample['style_name']} - Shape: {style.shape}")
        
        # Calculate statistics
        for name, img in [('Content', content), ('Style', style)]:
            print(f"{name} stats - Min: {img.min():.4f}, Max: {img.max():.4f}, "
                  f"Mean: {img.mean():.4f}, Std: {img.std():.4f}")
    
    print("\nDataset inspection complete")


def visualize_sample_pairs(dataset: StyleTransferDataset, num_pairs: int = 3) -> None:
    """
    Visualize sample pairs from the dataset.
    
    Args:
        dataset (StyleTransferDataset): Dataset to visualize
        num_pairs (int): Number of pairs to display
    """
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    
    # Get sample indices
    indices = random.sample(range(len(dataset)), min(num_pairs, len(dataset)))
    
    # Create figure
    fig, axes = plt.subplots(num_pairs, 2, figsize=(10, 4 * num_pairs))
    if num_pairs == 1:
        axes = axes.reshape(1, -1)
    
    # Display pairs
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        content = sample['content']
        style = sample['style']
        
        # De-normalize images for display
        content_display = denormalize_tensor(content)
        style_display = denormalize_tensor(style)
        
        # Display images
        axes[i, 0].imshow(content_display.permute(1, 2, 0).cpu())
        axes[i, 0].set_title(f"Content: {sample['content_name']}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(style_display.permute(1, 2, 0).cpu())
        axes[i, 1].set_title(f"Style: {sample['style_name']}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize a tensor from ImageNet normalization.
    
    Args:
        tensor (torch.Tensor): Normalized tensor
        
    Returns:
        torch.Tensor: Denormalized tensor with values in [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    return tensor.cpu() * std + mean


class AddGaussianNoise:
    """
    Add Gaussian noise to image tensors.
    
    This transform adds subtle Gaussian noise to simulate real-world
    image imperfections while preserving structure.
    
    Args:
        mean (float): Mean of the Gaussian noise (typically 0)
        std (float): Standard deviation of the noise (0.01-0.02 recommended)
    """
    
    def __init__(self, mean=0., std=0.015):
        self.mean = mean
        self.std = std
        
    def __call__(self, img):
        """
        Add Gaussian noise to a PIL image.
        
        Args:
            img (PIL.Image): Input image
            
        Returns:
            PIL.Image: Image with added noise
        """
        # Convert PIL image to tensor
        to_tensor = T.ToTensor()
        img_tensor = to_tensor(img)
        
        # Add Gaussian noise
        noise = torch.randn_like(img_tensor) * self.std + self.mean
        noisy_img = img_tensor + noise
        
        # Clamp to valid range [0, 1]
        noisy_img = torch.clamp(noisy_img, 0., 1.)
        
        # Convert back to PIL image
        to_pil = T.ToPILImage()
        return to_pil(noisy_img)


if __name__ == "__main__":
    print("Testing StyleTransferDataset and DataLoader...")
    
    dataset = StyleTransferDataset(
        content_dir=CONTENT_DIR,
        style_dir=STYLE_DIR,
        image_size=IMAGE_SIZE,
        augment=True,
        uniform_size=True  
    )
    
    inspect_dataset(dataset, num_samples=2)
    
    dataloader = get_dataloader(
        batch_size=2,
        shuffle=True,
        num_workers=2 if device.type != 'cpu' else 0,
        uniform_size=True
    )
    
    batch = next(iter(dataloader))
    print(f"\nBatch shapes:")
    print(f"Content batch shape: {batch['content'].shape}")
    print(f"Style batch shape: {batch['style'].shape}")
    
    print("\nDataloader test complete") 