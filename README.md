# style-transfer

pip install -r requirements.txt

Content Image + Style Image
      ↓
Semantic Segmentation (DeepLab)
      ↓
Depth Estimation (MiDaS)
      ↓
Gradient Editing for Lighting (basic Laplacian blending)
      ↓
Edge-Aware Smoothing (Guided Filter)
      ↓
Controlled Style Transfer (AdaIN limited by mask etc.)
      ↓
Output Photo
