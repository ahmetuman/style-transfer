"""
Configuration parameters for the photorealistic style transfer project.
"""

import os
from pathlib import Path

# Project paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT_DIR / 'data'
CONTENT_DIR = DATA_DIR / 'content'
STYLE_DIR = DATA_DIR / 'style'
RESULTS_DIR = DATA_DIR / 'results'
MODELS_DIR = ROOT_DIR / 'models'

# Create directories if they don't exist
for dir_path in [CONTENT_DIR, STYLE_DIR, RESULTS_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Model parameters
VGG_MODEL = 'vgg19'  # vgg16 or vgg19
SEGMENTATION_MODEL = 'deeplabv3_resnet101'
DEPTH_MODEL = 'MiDaS_small'  # Options: MiDaS_small, DPT_Large, etc.

# Style transfer parameters
CONTENT_WEIGHT = 1.0
STYLE_WEIGHT = 10.0
TV_WEIGHT = 1e-4  # Total Variation weight for smoothing
SEMANTIC_WEIGHT = 10.0  # Weight for semantic region matching

# Optimizer parameters
LEARNING_RATE = 0.01
NUM_ITERATIONS = 500

# Image parameters
IMAGE_SIZE = 512  # Default image size for processing
MAX_IMAGE_SIZE = 1024  # Max image size to prevent memory issues

# Gradient domain parameters
GRAD_WEIGHT = 1.0
LAPLACIAN_WEIGHT = 50.0

# Edge-aware filtering parameters
BILATERAL_SIGMA_SPACE = 10.0
BILATERAL_SIGMA_COLOR = 0.1
GUIDED_FILTER_RADIUS = 8
GUIDED_FILTER_EPS = 1e-2 