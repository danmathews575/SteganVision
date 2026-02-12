# Dataset Implementation

## Overview

The `SteganographyDataset` class loads pairs of cover and secret images for steganography training.

- **Cover Images**: CelebA face images (RGB, 256x256)
- **Secret Images**: MNIST handwritten digits (Grayscale, 256x256)
- **Normalization**: Both normalized to [-1, 1]

## Setup

### 1. Download CelebA Dataset

Place CelebA images in:
```
data/celeba/img_align_celeba/
```

Download from: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

### 2. MNIST Dataset

MNIST will auto-download on first use to `data/mnist/`

## Usage

### Basic Usage

```python
from src.data.dataset import SteganographyDataset

# Create dataset
dataset = SteganographyDataset(
    celeba_dir='data/celeba/img_align_celeba',
    mnist_root='data/mnist',
    image_size=256,
    split='train'
)

# Get a sample
cover_image, secret_image = dataset[0]

# Shapes:
# cover_image: [3, 256, 256], range [-1, 1]
# secret_image: [1, 256, 256], range [-1, 1]
```

### With DataLoader

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

for cover_batch, secret_batch in dataloader:
    # cover_batch: [32, 3, 256, 256]
    # secret_batch: [32, 1, 256, 256]
    pass
```

## Testing

Run the test script:

```bash
python scripts/test_dataset.py
```

## Parameters

- `celeba_dir`: Path to CelebA images directory (default: `'data/celeba/img_align_celeba'`)
- `mnist_root`: Path to MNIST root directory (default: `'data/mnist'`)
- `image_size`: Image size for both datasets (default: `256`)
- `split`: Dataset split - `'train'` or `'test'` (default: `'train'`)

## Data Structure

```
data/
├── celeba/
│   └── img_align_celeba/
│       ├── 000001.jpg
│       ├── 000002.jpg
│       └── ...
│
└── mnist/
    └── MNIST/
        ├── raw/
        └── processed/
```

## Implementation Details

### Preprocessing

**CelebA (Cover Images)**:
1. Resize to 256x256
2. Center crop to 256x256
3. Convert to tensor [0, 1]
4. Normalize to [-1, 1]: `(x - 0.5) / 0.5`

**MNIST (Secret Images)**:
1. Resize from 28x28 to 256x256
2. Convert to tensor [0, 1]
3. Normalize to [-1, 1]: `(x - 0.5) / 0.5`

### Dataset Length

The dataset length is the minimum of CelebA and MNIST sizes to ensure all samples have valid pairs.

## Denormalization

To convert back to [0, 1] for visualization:

```python
def denormalize(tensor):
    """Convert from [-1, 1] to [0, 1]"""
    return (tensor + 1) / 2
```

## Example Output

```
Dataset initialized:
  - CelebA images: 202599
  - MNIST images: 60000
  - Total pairs: 60000
```
