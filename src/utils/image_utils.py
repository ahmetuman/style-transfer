"""
Utility functions for image processing.
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

# Standard image transforms
def get_image_transform(size=None):
    """
    Returns a transform that normalizes the image to the range expected by pre-trained models.
    
    Args:
        size (int or tuple, optional): Size to resize the image to. If None, no resizing is applied.
        
    Returns:
        torchvision.transforms.Compose: The composed transforms.
    """
    transform_list = []
    
    if size is not None:
        transform_list.append(transforms.Resize(size))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transform_list)

def load_image(img_path, size=None):
    """
    Load an image from path, resize if needed, and convert to a normalized tensor.
    
    Args:
        img_path (str): Path to the image file.
        size (int or tuple, optional): Size to resize the image to. If None, no resizing is applied.
        
    Returns:
        torch.Tensor: The image tensor.
        PIL.Image.Image: The original PIL image.
    """
    image = Image.open(img_path).convert('RGB')
    
    if size is not None:
        # Preserve aspect ratio when resizing
        image = resize_preserve_aspect_ratio(image, size)
    
    transform = get_image_transform()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor, image

def resize_preserve_aspect_ratio(image, target_size):
    """
    Resize the image while preserving its aspect ratio.
    
    Args:
        image (PIL.Image.Image): The image to resize.
        target_size (int or tuple): Target size.
        
    Returns:
        PIL.Image.Image: The resized image.
    """
    if isinstance(target_size, int):
        width, height = image.size
        if width > height:
            new_width = target_size
            new_height = int(target_size * height / width)
        else:
            new_height = target_size
            new_width = int(target_size * width / height)
        target_size = (new_width, new_height)
    
    return image.resize(target_size, Image.LANCZOS)

def tensor_to_image(tensor):
    """
    Convert a tensor to a PIL Image.
    
    Args:
        tensor (torch.Tensor): Image tensor with shape (1, C, H, W) or (C, H, W).
        
    Returns:
        PIL.Image.Image: The converted image.
    """
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    
    # Undo normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    
    # Convert to PIL Image
    tensor = tensor.cpu().detach()
    img = F.to_pil_image(tensor)
    
    return img

def save_image(tensor, path):
    """
    Save a tensor as an image.
    
    Args:
        tensor (torch.Tensor): Image tensor.
        path (str): Output path.
    """
    img = tensor_to_image(tensor)
    img.save(path)

def plot_images(images, titles=None, figsize=(15, 5)):
    """
    Plot multiple images side by side.
    
    Args:
        images (list): List of images (PIL Images or tensors).
        titles (list, optional): List of titles for each image.
        figsize (tuple, optional): Figure size.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    for i, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            img = tensor_to_image(img)
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        if titles is not None and i < len(titles):
            axes[i].set_title(titles[i])
    
    plt.tight_layout()
    plt.show()

def gram_matrix(features):
    """
    Compute the Gram matrix of a given feature.
    
    Args:
        features (torch.Tensor): Feature tensor of shape (1, C, H, W).
        
    Returns:
        torch.Tensor: Gram matrix.
    """
    batch_size, ch, h, w = features.size()
    features_reshaped = features.view(batch_size, ch, -1)
    return torch.bmm(features_reshaped, features_reshaped.transpose(1, 2)) / (ch * h * w) 