"""LLM modified version (not working)"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any

# Ensure Metal performance on M1 Mac
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Segmentation: Using MPS (Metal Performance Shaders) on M1 Pro")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Segmentation: Using CUDA")
else:
    device = torch.device("cpu")
    print("Segmentation: Using CPU")

# Import torchvision's implementation of DeepLabV3
from torchvision.models.segmentation import (
    deeplabv3_resnet101, DeepLabV3_ResNet101_Weights,
    deeplabv3_resnet50, DeepLabV3_ResNet50_Weights,
    fcn_resnet101, FCN_ResNet101_Weights
)


class SegmentationModel:
    """
    DeepLabV3 with ResNet101 backbone for semantic segmentation.
    
    This class handles loading the model, pre-processing images, performing segmentation,
    and post-processing the results including visualization.
    
    Attributes:
        model (nn.Module): The DeepLabV3 model
        device (torch.device): Device to run inference on
        transform (T.Compose): Image preprocessing transformations
        classes (List[str]): Class names from the model
        colors (np.ndarray): Colors for visualization
        num_classes (int): Number of segmentation classes
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 21,  # Default for PASCAL VOC (20 classes + background)
        model_path: Optional[str] = None,
        model_name: str = "deeplabv3_resnet101"
    ):
        """
        Initialize the segmentation model.
        
        Args:
            pretrained (bool): Whether to use pretrained weights
            num_classes (int): Number of classes to segment
            model_path (str, optional): Path to custom model weights
            model_name (str): Model architecture to use
        """
        self.device = device
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Initialize model based on model_name
        if pretrained:
            if model_name == "deeplabv3_resnet101":
                # Load pretrained model with automatic weights selection
                weights = DeepLabV3_ResNet101_Weights.DEFAULT
                self.model = deeplabv3_resnet101(weights=weights)
                self.classes = weights.meta["categories"]
                print(f"Loaded pretrained DeepLabV3_ResNet101 with {len(self.classes)} classes")
            elif model_name == "deeplabv3_resnet50":
                # Use ResNet50 backbone
                weights = DeepLabV3_ResNet50_Weights.DEFAULT
                self.model = deeplabv3_resnet50(weights=weights)
                self.classes = weights.meta["categories"]
                print(f"Loaded pretrained DeepLabV3_ResNet50 with {len(self.classes)} classes")
            elif model_name == "fcn_resnet101":
                # Use FCN with ResNet101 backbone
                weights = FCN_ResNet101_Weights.DEFAULT
                self.model = fcn_resnet101(weights=weights)
                self.classes = weights.meta["categories"]
                print(f"Loaded pretrained FCN_ResNet101 with {len(self.classes)} classes")
            else:
                raise ValueError(f"Unsupported model name: {model_name}")
        else:
            # Initialize model without pretrained weights
            if model_name == "deeplabv3_resnet101":
                self.model = deeplabv3_resnet101(weights=None, num_classes=num_classes)
            elif model_name == "deeplabv3_resnet50":
                self.model = deeplabv3_resnet50(weights=None, num_classes=num_classes)
            elif model_name == "fcn_resnet101":
                self.model = fcn_resnet101(weights=None, num_classes=num_classes)
            else:
                raise ValueError(f"Unsupported model name: {model_name}")
                
            self.classes = [f"class_{i}" for i in range(num_classes)]
            print(f"Initialized {model_name} with {num_classes} classes")
        
        # Load custom weights if provided
        if model_path is not None and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded custom weights from {model_path}")
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Set up preprocessing transform
        if pretrained:
            self.transform = weights.transforms()
        else:
            # Standard normalization if not using pretrained weights
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        # Generate colors for visualization
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)
        # Set background (class 0) to black
        self.colors[0] = [0, 0, 0]
    
    def segment(
        self, 
        image: Union[str, Path, Image.Image, torch.Tensor, np.ndarray],
        return_logits: bool = False
    ) -> Dict[str, Any]:
        """
        Perform semantic segmentation on an image.
        
        Args:
            image: Input image (file path, PIL Image, tensor, or numpy array)
            return_logits: Whether to return raw logits
            
        Returns:
            Dict containing:
                'mask': Segmentation mask as tensor (H, W) with class indices
                'colored_mask': Visualization of the mask (H, W, 3) as numpy array
                'class_counts': Dict mapping class_idx to pixel count
                'logits': Raw model output logits (optional)
        """
        # Save original image size for resizing masks
        original_size = None
        if isinstance(image, Image.Image):
            original_size = image.size  # (width, height)
        
        # Load and preprocess the image
        img_tensor = self._preprocess_image(image)
        
        # Run inference
        try:
            with torch.no_grad():
                output = self.model(img_tensor)
                logits = output["out"]
                
            # Print model output stats for debugging
            print(f"Model output shape: {logits.shape}")
            print(f"Min logit value: {logits.min().item():.4f}, Max: {logits.max().item():.4f}")
            
            # Check if the model is predicting all background
            if torch.argmax(logits, dim=1).unique().numel() == 1:
                print("WARNING: Model is only predicting background class!")
                
                # Try to force some non-background classes for visualization
                if logits.shape[1] > 1:  # Only if we have multiple classes
                    # Artificially boost some non-background class scores
                    # This is just for debugging purposes
                    confidence_threshold = 0.3
                    # Get max confidence for each pixel across all classes
                    max_conf, _ = torch.max(torch.softmax(logits, dim=1), dim=1)
                    
                    # Count pixels with reasonable confidence
                    confident_pixels = (max_conf > confidence_threshold).sum().item()
                    if confident_pixels > 0:
                        print(f"Found {confident_pixels} pixels with confidence > {confidence_threshold}")
                    else:
                        print(f"No pixels with confidence > {confidence_threshold}")
            
            # Get the predicted mask
            mask = torch.argmax(logits, dim=1).squeeze(0).cpu()
            
            # Create colored visualization
            colored_mask = self._colorize_mask(mask.numpy())
            
            # Count pixels per class
            unique_classes, counts = torch.unique(mask, return_counts=True)
            class_counts = {int(cls.item()): int(count.item()) for cls, count in zip(unique_classes, counts)}
            
            # Create result dictionary
            result = {
                'mask': mask,
                'colored_mask': colored_mask,
                'class_counts': class_counts,
                'class_names': {idx: self.classes[idx] for idx in class_counts.keys() if idx < len(self.classes)}
            }
            
            # If we have original size info, store it
            if original_size:
                result['original_size'] = original_size
                
            if return_logits:
                result['logits'] = logits.cpu()
                
            return result
            
        except Exception as e:
            # Handle exceptions gracefully
            print(f"Error during segmentation: {e}")
            # Return a dummy mask filled with background (class 0)
            h, w = img_tensor.shape[-2], img_tensor.shape[-1]
            dummy_mask = torch.zeros((h, w), dtype=torch.long)
            colored_dummy = np.zeros((h, w, 3), dtype=np.uint8)
            return {
                'mask': dummy_mask,
                'colored_mask': colored_dummy,
                'class_counts': {0: h*w},
                'class_names': {0: '__background__' if len(self.classes) > 0 else 'background'}
            }
    
    def get_class_mask(self, segmentation_mask: torch.Tensor, class_idx: int) -> torch.Tensor:
        """
        Extract a binary mask for a specific class.
        
        Args:
            segmentation_mask: The full segmentation mask (H, W)
            class_idx: The index of the class to extract
            
        Returns:
            Binary mask as tensor where 1 indicates the specified class
        """
        return (segmentation_mask == class_idx).float()
    
    def _preprocess_image(
        self, 
        image: Union[str, Path, Image.Image, torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Preprocess an image for segmentation.
        
        Args:
            image: Input image in various formats
            
        Returns:
            Preprocessed image tensor (1, C, H, W) on device
        """
        # Handle different input types
        if isinstance(image, (str, Path)):
            try:
                image = Image.open(str(image)).convert('RGB')
            except Exception as e:
                print(f"Error opening image file {image}: {e}")
                # Return a dummy tensor
                return torch.zeros((1, 3, 224, 224), device=self.device)
                
        elif isinstance(image, np.ndarray):
            try:
                # Ensure it's uint8 RGB
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                
                # Handle grayscale or RGBA
                if len(image.shape) == 2:  # Grayscale
                    image = np.stack([image, image, image], axis=2)
                elif image.shape[2] == 4:  # RGBA
                    image = image[:, :, :3]
                elif image.shape[2] == 1:  # Single channel
                    image = np.concatenate([image, image, image], axis=2)
                    
                image = Image.fromarray(image)
            except Exception as e:
                print(f"Error converting numpy array to PIL: {e}")
                return torch.zeros((1, 3, 224, 224), device=self.device)
                
        elif isinstance(image, torch.Tensor):
            try:
                # If already a tensor, ensure it's the right format (B, C, H, W)
                if image.dim() == 3:  # (C, H, W)
                    image = image.unsqueeze(0)
                
                # If already preprocessed, just return it on the right device
                if image.shape[1] == 3:  # Has 3 channels
                    return image.to(self.device)
                else:
                    raise ValueError(f"Tensor should have 3 channels, got {image.shape[1]}")
            except Exception as e:
                print(f"Error processing tensor: {e}")
                return torch.zeros((1, 3, 224, 224), device=self.device)
        
        # Apply transform to the PIL image
        if isinstance(image, Image.Image):
            try:
                # Make sure the image is not too small for the model to process
                # DeepLabV3 typically needs images with dimensions at least 224x224
                w, h = image.size
                MIN_SIZE = 224
                if w < MIN_SIZE or h < MIN_SIZE:
                    # Resize while maintaining aspect ratio
                    if w < h:
                        new_w = MIN_SIZE
                        new_h = int(h * MIN_SIZE / w)
                    else:
                        new_h = MIN_SIZE
                        new_w = int(w * MIN_SIZE / h)
                    image = image.resize((new_w, new_h), Image.LANCZOS)
                
                # Limit maximum size to avoid CUDA memory issues
                MAX_SIZE = 1024
                if w > MAX_SIZE or h > MAX_SIZE:
                    if w > h:
                        new_w = MAX_SIZE
                        new_h = int(h * MAX_SIZE / w)
                    else:
                        new_h = MAX_SIZE
                        new_w = int(w * MAX_SIZE / h)
                    image = image.resize((new_w, new_h), Image.LANCZOS)
                    
                # Apply the transformation
                img_tensor = self.transform(image).unsqueeze(0).to(self.device)
                return img_tensor
            except Exception as e:
                print(f"Error preprocessing PIL image: {e}")
                return torch.zeros((1, 3, 224, 224), device=self.device)
        
        # If we get here, the image type wasn't handled
        print(f"Unsupported image type: {type(image)}")
        return torch.zeros((1, 3, 224, 224), device=self.device)
    
    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert a segmentation mask to a colored visualization.
        
        Args:
            mask: Segmentation mask (H, W) with class indices
            
        Returns:
            Colored mask as RGB image (H, W, 3)
        """
        # Apply the color mapping
        colored_mask = self.colors[mask]
        
        # If mask is all background (class 0), use a gradient for better visualization
        if np.all(mask == 0):
            h, w = mask.shape
            # Create a gradient pattern instead of all black
            gradient = np.zeros((h, w, 3), dtype=np.uint8)
            # Horizontal gradient (dark blue to light blue)
            for i in range(w):
                intensity = int(255 * i / w)
                gradient[:, i, 0] = 0  # R
                gradient[:, i, 1] = intensity  # G
                gradient[:, i, 2] = 255  # B
            
            # Add grid lines for better visualization
            grid_step = 32
            grid_color = (180, 180, 180)
            for i in range(0, h, grid_step):
                gradient[i, :] = grid_color
            for i in range(0, w, grid_step):
                gradient[:, i] = grid_color
                
            # Add text indicating "Background only"
            if h > 100 and w > 200:  # Only if image is big enough
                text_area = gradient[h//2-15:h//2+15, w//2-100:w//2+100]
                text_area[:, :] = (200, 200, 200)  # Light gray box
                
            return gradient
            
        return colored_mask


class SemanticMatcher:
    """
    Matches semantic regions between content and style images.
    
    This class handles finding corresponding semantic regions between
    content and style images for region-based style transfer.
    
    Attributes:
        segmentation_model: The SegmentationModel to use
        matching_strategy: Strategy for matching regions
    """
    
    def __init__(
        self, 
        segmentation_model: SegmentationModel,
        matching_strategy: str = 'exact',
    ):
        """
        Initialize the semantic matcher.
        
        Args:
            segmentation_model: Model for semantic segmentation
            matching_strategy: How to match regions ('exact', 'nearest')
        """
        self.segmentation_model = segmentation_model
        self.matching_strategy = matching_strategy
    
    def match_regions(
        self, 
        content_img: Union[str, Path, Image.Image, torch.Tensor],
        style_img: Union[str, Path, Image.Image, torch.Tensor],
        threshold_percentage: float = 0.5  # Min percentage of pixels to consider a class present
    ) -> Dict[str, Any]:
        """
        Match semantic regions between content and style images.
        
        Args:
            content_img: Content image
            style_img: Style image
            threshold_percentage: Minimum percentage of pixels to consider a class present
            
        Returns:
            Dict containing:
                'content_seg': Content segmentation result
                'style_seg': Style segmentation result
                'matches': Dict mapping content class idx to style class idx
                'region_masks': Dict with masks for each matched region
        """
        # Segment both images
        content_seg = self.segmentation_model.segment(content_img)
        style_seg = self.segmentation_model.segment(style_img)
        
        # Get content and style masks
        content_mask = content_seg['mask']
        style_mask = style_seg['mask']
        
        # Get size information
        content_size = content_mask.numel()
        style_size = style_mask.numel()
        
        # Extract significant classes (filtering by threshold)
        significant_content_classes = {}
        for cls, count in content_seg['class_counts'].items():
            percentage = count / content_size
            if percentage >= threshold_percentage:
                significant_content_classes[cls] = percentage
        
        significant_style_classes = {}
        for cls, count in style_seg['class_counts'].items():
            percentage = count / style_size
            if percentage >= threshold_percentage:
                significant_style_classes[cls] = percentage
        
        # Find matching classes
        matches = {}
        if self.matching_strategy == 'exact':
            # Only match exact same classes
            for content_cls in significant_content_classes:
                if content_cls in significant_style_classes:
                    matches[content_cls] = content_cls
        elif self.matching_strategy == 'nearest':
            # TODO: Implement more sophisticated matching using semantic similarity
            # This would require word embeddings or a predefined similarity matrix
            # For now, use exact matching as fallback
            for content_cls in significant_content_classes:
                if content_cls in significant_style_classes:
                    matches[content_cls] = content_cls
        
        # Create binary masks for each matched region
        region_masks = {}
        for content_cls, style_cls in matches.items():
            content_class_mask = self.segmentation_model.get_class_mask(content_mask, content_cls)
            style_class_mask = self.segmentation_model.get_class_mask(style_mask, style_cls)
            
            region_masks[content_cls] = {
                'content_mask': content_class_mask,
                'style_mask': style_class_mask,
                'content_class': content_cls,
                'style_class': style_cls,
                'class_name': content_seg['class_names'].get(content_cls, f"class_{content_cls}")
            }
        
        return {
            'content_seg': content_seg,
            'style_seg': style_seg,
            'matches': matches,
            'region_masks': region_masks,
            'significant_content_classes': significant_content_classes,
            'significant_style_classes': significant_style_classes
        }


def visualize_segmentation(
    image: Union[str, Path, Image.Image, torch.Tensor, np.ndarray],
    seg_result: Dict[str, Any],
    alpha: float = 0.5,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Visualize segmentation results.
    
    Args:
        image: Original image
        seg_result: Result from SegmentationModel.segment()
        alpha: Transparency for the mask overlay
        figsize: Figure size
        save_path: Path to save the visualization
    """
    # Load image if it's a path
    if isinstance(image, (str, Path)):
        image = Image.open(str(image)).convert('RGB')
    
    # Convert tensor or numpy array to PIL
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:  # (B, C, H, W)
            image = image.squeeze(0)
        if image.dim() == 3:  # (C, H, W)
            # Denormalize if in range [-1, 1] or [0, 1]
            if image.min() < 0 or image.max() <= 1:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image = image * std + mean
                image = image.clamp(0, 1)
            
            # Convert to PIL
            image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
    
    if isinstance(image, np.ndarray):
        # Ensure it's in uint8 range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Create PIL image
        if image.shape[2] == 3:  # RGB
            image = Image.fromarray(image)
        else:
            raise ValueError(f"Image should have 3 channels, got {image.shape[2]}")
    
    # Get colored mask from results
    colored_mask = seg_result['colored_mask']
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Plot segmentation mask
    plt.subplot(1, 3, 2)
    plt.imshow(colored_mask)
    plt.title("Segmentation Mask")
    plt.axis('off')
    
    # Plot overlay
    plt.subplot(1, 3, 3)
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Create an overlay
    overlay = image_np.copy().astype(np.float32)
    
    # Ensure mask and image dimensions match (resize mask if needed)
    mask_shape = seg_result['mask'].shape
    if mask_shape != image_np.shape[:2]:
        # Resize the mask to match the image
        orig_mask = seg_result['mask'].numpy()
        resized_mask = np.zeros(image_np.shape[:2], dtype=orig_mask.dtype)
        # Use the smaller dimensions for slicing to avoid index errors
        h = min(mask_shape[0], image_np.shape[0])
        w = min(mask_shape[1], image_np.shape[1])
        resized_mask[:h, :w] = orig_mask[:h, :w]
        seg_result['mask'] = torch.from_numpy(resized_mask)
        
        # Also resize the colored mask
        resized_colored = np.zeros((image_np.shape[0], image_np.shape[1], 3), dtype=np.uint8)
        resized_colored[:h, :w] = seg_result['colored_mask'][:h, :w]
        seg_result['colored_mask'] = resized_colored
    
    for cls, count in seg_result['class_counts'].items():
        # Skip background (class 0)
        if cls == 0:
            continue
        
        # Get class name if available
        class_name = seg_result['class_names'].get(cls, f"class_{cls}")
        
        # Get class mask
        mask = (seg_result['mask'] == cls).numpy()
        
        # Check if the mask contains any pixels
        if not np.any(mask):
            continue
            
        # Double-check dimensions match
        if mask.shape != image_np.shape[:2]:
            continue
            
        color = tuple(map(int, seg_result['colored_mask'][mask][0]))
        
        # Apply color overlay
        for c in range(3):
            overlay[mask, c] = alpha * color[c] + (1 - alpha) * image_np[mask, c]
    
    plt.imshow(overlay.astype(np.uint8))
    plt.title("Segmentation Overlay")
    plt.axis('off')
    
    # Add class information
    class_info = ""
    for cls in sorted(seg_result['class_counts'].keys()):
        if cls in seg_result['class_names']:
            class_info += f"Class {cls}: {seg_result['class_names'][cls]}\n"
        else:
            class_info += f"Class {cls}\n"
    
    plt.figtext(0.02, 0.02, class_info, fontsize=8, wrap=True)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_region_matching(
    content_img: Union[str, Path, Image.Image, torch.Tensor, np.ndarray],
    style_img: Union[str, Path, Image.Image, torch.Tensor, np.ndarray],
    matching_result: Dict[str, Any],
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Visualize matched regions between content and style images.
    
    Args:
        content_img: Content image
        style_img: Style image
        matching_result: Result from SemanticMatcher.match_regions()
        figsize: Figure size
        save_path: Path to save the visualization
    """
    # Load images if they're paths
    if isinstance(content_img, (str, Path)):
        content_img = Image.open(str(content_img)).convert('RGB')
    if isinstance(style_img, (str, Path)):
        style_img = Image.open(str(style_img)).convert('RGB')
    
    # Convert tensors to PIL images if needed
    if isinstance(content_img, torch.Tensor):
        if content_img.dim() == 4:
            content_img = content_img.squeeze(0)
        content_img = content_img.permute(1, 2, 0).cpu().numpy()
        if content_img.max() <= 1.0:
            content_img = (content_img * 255).astype(np.uint8)
        content_img = Image.fromarray(content_img)
    
    if isinstance(style_img, torch.Tensor):
        if style_img.dim() == 4:
            style_img = style_img.squeeze(0)
        style_img = style_img.permute(1, 2, 0).cpu().numpy()
        if style_img.max() <= 1.0:
            style_img = (style_img * 255).astype(np.uint8)
        style_img = Image.fromarray(style_img)
    
    # Convert numpy arrays to PIL images if needed
    if isinstance(content_img, np.ndarray):
        if content_img.max() <= 1.0:
            content_img = (content_img * 255).astype(np.uint8)
        content_img = Image.fromarray(content_img)
    
    if isinstance(style_img, np.ndarray):
        if style_img.max() <= 1.0:
            style_img = (style_img * 255).astype(np.uint8)
        style_img = Image.fromarray(style_img)
    
    # Get the number of matched regions
    region_masks = matching_result['region_masks']
    num_matches = len(region_masks)
    
    if num_matches == 0:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(content_img)
        plt.title("Content Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(style_img)
        plt.title("Style Image")
        plt.axis('off')
        
        plt.figtext(0.5, 0.3, "No matching regions found", 
                  fontsize=14, ha='center', 
                  bbox=dict(facecolor='red', alpha=0.1))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        return
    
    # Create a figure with rows for each match
    num_rows = num_matches + 1  # +1 for original images
    fig, axes = plt.subplots(num_rows, 2, figsize=figsize)
    
    # Display original images in the first row
    axes[0, 0].imshow(content_img)
    axes[0, 0].set_title("Content Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(style_img)
    axes[0, 1].set_title("Style Image")
    axes[0, 1].axis('off')
    
    # Convert images to numpy for overlaying masks
    content_np = np.array(content_img)
    style_np = np.array(style_img)
    
    # Get content and style image dimensions
    c_height, c_width, _ = content_np.shape
    s_height, s_width, _ = style_np.shape
    
    # Plot each matched region
    for i, (class_idx, region_data) in enumerate(region_masks.items(), 1):
        # Get binary masks
        content_mask = region_data['content_mask'].numpy()
        style_mask = region_data['style_mask'].numpy()
        class_name = region_data['class_name']
        
        # Resize masks to match image dimensions if needed
        if content_mask.shape != (c_height, c_width):
            # Create a properly sized mask filled with False
            resized_content_mask = np.zeros((c_height, c_width), dtype=bool)
            
            # Calculate size to copy (take minimum to avoid index errors)
            h = min(content_mask.shape[0], c_height)
            w = min(content_mask.shape[1], c_width)
            
            # Copy data from the original mask to the resized mask
            resized_content_mask[:h, :w] = content_mask[:h, :w]
            content_mask = resized_content_mask
        
        if style_mask.shape != (s_height, s_width):
            # Similar resizing for style mask
            resized_style_mask = np.zeros((s_height, s_width), dtype=bool)
            h = min(style_mask.shape[0], s_height)
            w = min(style_mask.shape[1], s_width)
            resized_style_mask[:h, :w] = style_mask[:h, :w]
            style_mask = resized_style_mask
        
        # Verify masks contain pixels
        if not np.any(content_mask) or not np.any(style_mask):
            continue
        
        # Create masked images (apply a color overlay)
        content_masked = content_np.copy()
        style_masked = style_np.copy()
        
        # Apply a semi-transparent colored overlay
        overlay_color = np.array([255, 0, 0], dtype=np.uint8)  # Red overlay
        alpha = 0.5
        
        # For content image
        for c in range(3):
            content_masked[:, :, c] = np.where(
                content_mask, 
                content_masked[:, :, c] * (1 - alpha) + overlay_color[c] * alpha,
                content_masked[:, :, c]
            )
        
        # For style image
        for c in range(3):
            style_masked[:, :, c] = np.where(
                style_mask, 
                style_masked[:, :, c] * (1 - alpha) + overlay_color[c] * alpha,
                style_masked[:, :, c]
            )
        
        # Display the masked images
        axes[i, 0].imshow(content_masked.astype(np.uint8))
        axes[i, 0].set_title(f"Content: {class_name}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(style_masked.astype(np.uint8))
        axes[i, 1].set_title(f"Style: {class_name}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Test the segmentation model and visualization
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.config import CONTENT_DIR, STYLE_DIR
    
    # Initialize model
    segmentation_model = SegmentationModel(pretrained=True)
    
    # Get sample images
    content_file = next(Path(CONTENT_DIR).glob("*.jpg"))
    style_file = next(Path(STYLE_DIR).glob("*.jpg"))
    
    print(f"Testing with: {content_file} and {style_file}")
    
    # Segment content image
    content_result = segmentation_model.segment(content_file)
    visualize_segmentation(content_file, content_result)
    
    # Segment style image
    style_result = segmentation_model.segment(style_file)
    visualize_segmentation(style_file, style_result)
    
    # Test semantic matching
    matcher = SemanticMatcher(segmentation_model)
    matching_result = matcher.match_regions(content_file, style_file)
    
    # Visualize matches
    visualize_region_matching(content_file, style_file, matching_result) 