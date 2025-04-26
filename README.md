# style-transfer

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
