# Quick Start Guide

## Project Overview

This is a clean, modular PyTorch project for **GAN-based Image Steganography** inspired by the HiDDeN paper. The project supports both CNN baseline and GAN extension approaches.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Project Structure Summary

### 📁 Core Directories

- **`src/`** - All source code (models, data, losses, training, utils)
- **`configs/`** - YAML configuration files for training
- **`scripts/`** - Executable training and inference scripts
- **`data/`** - Dataset storage (raw, processed, outputs)
- **`tests/`** - Unit tests for all components
- **`checkpoints/`** - Saved model checkpoints
- **`notebooks/`** - Jupyter notebooks for exploration

### 🧠 Models

Located in `src/models/`:
- `encoder.py` - Embeds secret messages into cover images
- `decoder.py` - Extracts messages from stego images
- `discriminator.py` - GAN discriminator (optional)
- `noise_layers.py` - Robustness layers (JPEG, noise, etc.)

### 📊 Training Approaches

**CNN Baseline** (Simple):
```bash
python scripts/train_cnn.py --config configs/cnn_baseline.yaml
```

**GAN-based** (Advanced):
```bash
python scripts/train_gan.py --config configs/gan.yaml
```

### 🎯 Key Features

✅ **Modular Architecture** - Easy to extend and modify  
✅ **Two Training Modes** - CNN baseline and GAN extension  
✅ **Comprehensive Losses** - Image quality + Message accuracy + Adversarial  
✅ **Evaluation Metrics** - PSNR, SSIM, message accuracy, bit error rate  
✅ **Flexible Configuration** - YAML-based config system  
✅ **Multiple Loggers** - TensorBoard and Weights & Biases support  
✅ **Unit Tests** - Test coverage for all components  

### 📝 Implementation Status

**Current State**: ✨ **Clean skeleton ready for implementation**

All files are created with:
- ✅ Proper class structures
- ✅ Method signatures
- ✅ Comprehensive docstrings
- ✅ TODO markers for implementation
- ✅ Type hints

**Next Steps**:
1. Implement model architectures (encoder, decoder, discriminator)
2. Implement data loading and preprocessing
3. Implement loss functions and metrics
4. Implement training loops
5. Add dataset and start training

### 🔧 Development Workflow

1. **Add your dataset** to `data/raw/train/` and `data/raw/val/`
2. **Implement models** starting with encoder and decoder
3. **Implement data pipeline** in `src/data/`
4. **Implement losses** in `src/losses/`
5. **Implement training** in `src/training/`
6. **Run tests** with `pytest tests/`
7. **Start training** with the scripts

### 📚 File Count

- **Python files**: 30+
- **Config files**: 2
- **Test files**: 4
- **Documentation**: 4

### 🎓 Inspired By

**HiDDeN: Hiding Data With Deep Networks** (Zhu et al., 2018)

---

**Ready to implement!** All structure is in place. Start by implementing the encoder and decoder networks, then move to the data pipeline and training logic.
