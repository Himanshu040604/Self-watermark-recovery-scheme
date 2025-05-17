# Self-watermark-recovery-scheme
```markdown
# Self Watermark Recovery Scheme

This repository implements a deep learning pipeline for robust image watermark recovery using an autoencoder, U-Net, and latent space transformations. The system combines traditional reconstruction losses with perceptual losses to enhance output quality and watermark fidelity.

---

## 📋 Overview

This project presents a hybrid architecture for self-recovering watermarks in 512×512 RGB images. The pipeline leverages:
- **Autoencoder** for latent feature extraction
- **U-Net** with expanded input channels for detail reconstruction
- **Latent Decoder** for cycle-consistent latent space validation
- **LPIPS perceptual loss** for human-aligned quality assessment

---

## 🧩 Key Components

### 1. Autoencoder Architecture

```
Autoencoder(
  (encoder): Sequential(  # Input: 3×512×512
    Conv2d(3→64), ReLU, Downsample,
    Conv2d(64→128), ReLU, Downsample,
    Conv2d(128→256), ReLU, Downsample,
    Conv2d(256→512)      # Latent: 512×32×32
  )
  (decoder): Sequential(  # Input: 512×32×32
    ConvTranspose2d(512→256), ReLU, Upsample,
    ConvTranspose2d(256→128), ReLU, Upsample,
    ConvTranspose2d(128→64), ReLU, Upsample,
    ConvTranspose2d(64→3), Sigmoid  # Output: 3×512×512
  )
)
```

### 2. U-Net with Expanded Input

```
UNet(
  (input_layer): Conv2d(5→64, kernel=3x3)  # 3 RGB + 2 latent channels
  # ... standard U-Net architecture ...
  (output_layer): Conv2d(64→3, kernel=1x1)  # Output: 3×512×512
)
```

---

## 🔄 Workflow

1. **Image Preprocessing**
   - Load and resize images to 3×512×512.

2. **Feature Extraction**
   - Pass the image through the autoencoder encoder to obtain a latent vector of shape 1×512×32×32.

3. **Latent Space Reshaping**
   - Upsample and reshape the latent vector to 1×2×512×512.

4. **Multi-Channel U-Net Processing**
   - Concatenate the reshaped latent vector (2 channels) with the original image (3 channels) to form a 5-channel input.
   - Pass this input through the U-Net to obtain a reconstructed 3-channel image.

5. **Latent Decoder and Cycle Consistency**
   - Use a latent decoder to extract the latent vector from the U-Net output.
   - Reshape and resize this latent vector to 1×512×32×32, and pass it through the decoder part of the autoencoder to reconstruct the image.

6. **Loss Calculation**
   - Compute the L1 loss and a perceptual LPIPS loss (weighted by a factor of 5) between the U-Net output and the ground truth.
   - Ensure all images and intermediate results maintain appropriate dimensions throughout the process.

---

## 📉 Loss Function

The total loss combines pixel-level accuracy and perceptual similarity:
```
total_loss = (l1_loss(reconstructed, target) + 
             5 * lpips_loss(reconstructed, target) +
             l1_loss(cycle_output, original_img))
```

---

## ✨ Features

- **Multi-Stage Processing:** Integrates autoencoding, U-Net refinement, and latent validation
- **Dimension Consistency:** Maintains 512×512 resolution throughout
- **Hybrid Loss:** Balances L1 (MAE) and LPIPS perceptual losses
- **Cycle Consistency:** Ensures latent space interpretability via reconstruction

---

## 🚀 Getting Started

```
# Clone the repository
git clone https://github.com/Himanshu-singh040604/self-watermark-recovery-scheme.git

# Create and activate a conda environment
conda create -n watermark_recovery python=3.9
conda activate watermark_recovery

# Install dependencies
pip install -r requirements.txt

# Start training
python train.py --batch_size 8 --lr 1e-4
```

---

## 📄 Citation

```
@misc{SelfWatermarkRecovery2024,
  author = {Himanshu Singh},
  title = {Self Watermark Recovery Scheme},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Himanshu-singh040604/self-watermark-recovery-scheme}}
}
```

---

**Note:**  
Replace any placeholder code or architecture details with your actual implementation if they differ.  
```

---
Answer from Perplexity: pplx.ai/share
