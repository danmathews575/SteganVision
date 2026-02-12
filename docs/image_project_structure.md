# GAN-based Image Steganography - Project Structure

## Complete Folder Tree

```
MajorP/
│
├── README.md                      # Project overview and documentation
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation script
├── .gitignore                     # Git ignore rules
│
├── configs/                       # Configuration files
│   ├── cnn_baseline.yaml         # Config for CNN baseline training
│   └── gan.yaml                  # Config for GAN training
│
├── data/                         # Data directory
│   ├── README.md                 # Data directory documentation
│   ├── raw/                      # Raw datasets
│   │   └── .gitkeep
│   ├── processed/                # Processed datasets
│   │   └── .gitkeep
│   └── outputs/                  # Generated outputs
│       └── .gitkeep
│
├── src/                          # Source code
│   ├── __init__.py
│   │
│   ├── models/                   # Model architectures
│   │   ├── __init__.py
│   │   ├── encoder.py           # Encoder network (embeds message)
│   │   ├── decoder.py           # Decoder network (extracts message)
│   │   ├── discriminator.py     # Discriminator for GAN training
│   │   └── noise_layers.py      # Noise layers for robustness
│   │
│   ├── data/                    # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py           # Dataset classes
│   │   ├── transforms.py        # Data transformations
│   │   └── dataloader.py        # DataLoader utilities
│   │
│   ├── losses/                  # Loss functions and metrics
│   │   ├── __init__.py
│   │   ├── losses.py            # Loss functions (image, message, adversarial)
│   │   └── metrics.py           # Evaluation metrics (PSNR, SSIM, accuracy)
│   │
│   ├── training/                # Training logic
│   │   ├── __init__.py
│   │   ├── base_trainer.py      # Base trainer class
│   │   ├── cnn_trainer.py       # CNN baseline trainer
│   │   └── gan_trainer.py       # GAN trainer
│   │
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       ├── logger.py            # Logging utilities (TensorBoard, W&B)
│       ├── visualization.py     # Visualization utilities
│       └── helpers.py           # General helper functions
│
├── scripts/                     # Training and evaluation scripts
│   ├── train_cnn.py            # Train CNN baseline
│   ├── train_gan.py            # Train GAN model
│   ├── evaluate.py             # Evaluate trained models
│   └── inference.py            # Inference script (embed/extract)
│
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── conftest.py             # PyTest configuration
│   ├── test_models.py          # Model tests
│   ├── test_data.py            # Data loading tests
│   └── test_losses.py          # Loss function tests
│
├── notebooks/                   # Jupyter notebooks
│   └── README.md               # Notebooks documentation
│
└── checkpoints/                 # Model checkpoints
    └── .gitkeep
```

## Key Components

### Models (`src/models/`)
- **Encoder**: Embeds secret messages into cover images
- **Decoder**: Extracts secret messages from stego images
- **Discriminator**: Distinguishes real from stego images (GAN training)
- **Noise Layers**: Simulates real-world distortions (JPEG, Gaussian noise, etc.)

### Data Pipeline (`src/data/`)
- **Dataset**: Loads cover images and generates random messages
- **Transforms**: Data augmentation and preprocessing
- **DataLoader**: Efficient batch loading

### Loss Functions (`src/losses/`)
- **Image Reconstruction**: MSE, L1, or perceptual loss
- **Message Recovery**: Binary cross-entropy for message accuracy
- **Adversarial Loss**: GAN objectives (vanilla, LSGAN, WGAN)
- **Combined Loss**: Weighted combination of all losses
- **Metrics**: PSNR, SSIM, message accuracy, bit error rate

### Training (`src/training/`)
- **BaseTrainer**: Common training functionality
- **CNNTrainer**: Simple encoder-decoder training
- **GANTrainer**: Adversarial training with discriminator

### Utilities (`src/utils/`)
- **Config**: YAML configuration management
- **Logger**: TensorBoard and Weights & Biases integration
- **Visualization**: Plot results and training curves
- **Helpers**: Seed setting, device management, model saving/loading

### Scripts (`scripts/`)
- **train_cnn.py**: Train CNN baseline model
- **train_gan.py**: Train GAN-based model
- **evaluate.py**: Evaluate model performance
- **inference.py**: Embed or extract messages from images

### Configuration (`configs/`)
- **cnn_baseline.yaml**: CNN training configuration
- **gan.yaml**: GAN training configuration

## Next Steps

1. **Implement Model Architectures**
   - Fill in encoder, decoder, and discriminator networks
   - Implement noise layers for robustness

2. **Implement Data Pipeline**
   - Complete dataset classes
   - Add data transformations
   - Implement message generation

3. **Implement Loss Functions**
   - Add all loss components
   - Implement evaluation metrics

4. **Implement Training Logic**
   - Complete CNN trainer
   - Complete GAN trainer
   - Add logging and checkpointing

5. **Test and Validate**
   - Write unit tests
   - Test on sample data
   - Validate model convergence

## Usage Examples

### Training CNN Baseline
```bash
python scripts/train_cnn.py --config configs/cnn_baseline.yaml
```

### Training GAN Model
```bash
python scripts/train_gan.py --config configs/gan.yaml
```

### Evaluation
```bash
python scripts/evaluate.py --checkpoint checkpoints/model.pth --data data/test
```

### Inference
```bash
# Embed message
python scripts/inference.py --mode embed --checkpoint model.pth --image cover.png --message "secret" --output stego.png

# Extract message
python scripts/inference.py --mode extract --checkpoint model.pth --image stego.png
```

## References

- **HiDDeN Paper**: "Hiding Data With Deep Networks" (Zhu et al., 2018)
- Inspired by deep learning approaches to steganography
- Modular design for easy experimentation and extension
