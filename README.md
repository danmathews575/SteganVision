# SteganVision 🔐
### A Multimodal AI-Driven Steganography System

![Architecture Diagram](docs/steganvision_architecture.png)

## 📖 Overview
**SteganVision** is a comprehensive steganography framework that leverages advanced AI and deep learning techniques to hide secrets across multiple modalities: **Image**, **Text**, and **Audio**. Unlike traditional methods, it uses:
- **GANs & CNNs** for Image Steganography (High capacity & imperceptibility)
- **AI-Guided LSB** for Text Steganography (Intelligent embedding placement)
- **Psychoacoustic Masking** for Audio Steganography (Inaudible modifications)

## ✨ Features

### 🖼️ Image → Image Steganography
- **Models**: Includes a baseline CNN and a proposed GAN-based architecture.
- **Performance**: High PSNR/SSIM with resistance to statistical steganalysis.
- **Capabilities**: Hides full-size images within other images.

### 📝 Text → Image Steganography
- **Method**: Adaptive LSB guided by edge detection (Sobel/Laplacian).
- **Security**: Embeds data only in complex regions to avoid visual artifacts.
- **Recovery**: 100% exact text recovery guarantee.

### 🎵 Audio → Audio Steganography
- **Method**: Psychoacoustic model-based LSB embedding.
- **Compression**: Integrated LZMA compression for maximized payload.
- **Format**: Supports WAV audio with exact recovery.

## 🚀 Setup & Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/steganvision.git
    cd steganvision
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **⚠️ Large Model Files (Important)**
    Due to GitHub's file size limits, the pre-trained model checkpoints (>100MB) are **excluded** from this repository.
    
    To run the **Image Steganography** module with pre-trained weights, you must:
    - Download the weights from [External Link Placeholder] (or train your own).
    - Place them in:
        - `image_steganography/checkpoints/best_model.pth`
        - `image_steganography/checkpoints/gan/gan_checkpoint_epoch_0023.pth`

## 🖥️ Usage

Run the unified Streamlit dashboard:
```bash
streamlit run app.py
```
This will launch a web interface where you can explore all three modules interactively.

## 📂 Project Structure
```
project-root/
├── app.py                 # Main Streamlit Dashboard entry point
├── audio_steganography/   # Audio module source & assets
├── image_steganography/   # Image GAN/CNN module source & assets
├── text_steganography/    # Text Adaptive LSB module source & assets
├── docs/                  # Documentation & Diagrams
├── scripts/               # Utility scripts
├── requirements.txt       # Unified dependencies
└── README.md              # This file
```

## 📄 Documentation
For more detailed information, check the `docs/` directory:
- [System Architecture](docs/steganvision_architecture.png)
- [Project Structure](docs/image_project_structure.md)
