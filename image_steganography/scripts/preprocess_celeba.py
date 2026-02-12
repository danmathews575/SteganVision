"""
Offline Preprocessing Script for CelebA Images

Resizes CelebA images to 256x256 and saves them to a new directory.
This speeds up training by avoiding the resize operation during data loading.

Usage:
    python scripts/preprocess_celeba.py

    # With custom paths
    python scripts/preprocess_celeba.py --input data/celeba --output data/celeba_256

    # Limit number of images (for testing)
    python scripts/preprocess_celeba.py --max_images 1000

After preprocessing, update your training command:
    python src/train/train_cnn.py --celeba_dir data/celeba_256
"""

import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess CelebA images: resize to 256x256',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (uses default paths)
    python scripts/preprocess_celeba.py

    # Custom input/output directories
    python scripts/preprocess_celeba.py --input path/to/celeba --output path/to/output

    # Process only first 1000 images (for testing)
    python scripts/preprocess_celeba.py --max_images 1000

    # Use 8 worker threads
    python scripts/preprocess_celeba.py --workers 8

After preprocessing, train with:
    python src/train/train_cnn.py --celeba_dir data/celeba_256
        """
    )
    
    parser.add_argument(
        '--input', type=str,
        default='data/celeba',
        help='Input directory containing CelebA images (default: data/celeba)'
    )
    parser.add_argument(
        '--output', type=str,
        default='data/celeba_256',
        help='Output directory for resized images (default: data/celeba_256)'
    )
    parser.add_argument(
        '--size', type=int, default=256,
        help='Target image size (default: 256)'
    )
    parser.add_argument(
        '--quality', type=int, default=95,
        help='JPEG quality for saved images (default: 95)'
    )
    parser.add_argument(
        '--max_images', type=int, default=None,
        help='Maximum number of images to process (default: None, process all)'
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Number of worker threads (default: CPU count)'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Overwrite existing files (default: skip existing)'
    )
    
    return parser.parse_args()


def resize_and_save(input_path: Path, output_path: Path, size: int, quality: int, force: bool) -> tuple:
    """
    Resize a single image and save it.
    
    Args:
        input_path: Path to input image
        output_path: Path to save resized image
        size: Target size (width and height)
        quality: JPEG quality (1-100)
        force: Whether to overwrite existing files
        
    Returns:
        tuple: (filename, status) where status is 'processed', 'skipped', or 'error: message'
    """
    filename = input_path.name
    
    try:
        # Skip if output exists and not forcing overwrite (idempotent)
        if output_path.exists() and not force:
            return (filename, 'skipped')
        
        # Open and resize image
        with Image.open(input_path) as img:
            # Convert to RGB (in case of RGBA or other modes)
            img = img.convert('RGB')
            
            # Resize with high-quality resampling
            # Use LANCZOS for best quality when downscaling
            img_resized = img.resize((size, size), Image.LANCZOS)
            
            # Save with specified quality
            img_resized.save(output_path, 'JPEG', quality=quality)
        
        return (filename, 'processed')
        
    except Exception as e:
        return (filename, f'error: {str(e)}')


def main():
    args = parse_args()
    
    # Setup paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    print("=" * 60)
    print("CelebA Image Preprocessing")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Input directory:  {input_dir.resolve()}")
    print(f"  Output directory: {output_dir.resolve()}")
    print(f"  Target size:      {args.size}x{args.size}")
    print(f"  JPEG quality:     {args.quality}")
    print(f"  Overwrite mode:   {'enabled' if args.force else 'disabled (skip existing)'}")
    
    # Validate input directory
    if not input_dir.exists():
        print(f"\n❌ Error: Input directory not found: {input_dir}")
        print(f"   Please ensure CelebA images are in: {input_dir}")
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Output directory ready: {output_dir}")
    
    # Get list of images to process
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ])
    
    total_images = len(image_paths)
    print(f"✓ Found {total_images:,} images in input directory")
    
    if total_images == 0:
        print("\n❌ No images found to process!")
        return 1
    
    # Apply max_images limit if specified
    if args.max_images is not None and args.max_images < total_images:
        image_paths = image_paths[:args.max_images]
        print(f"  → Processing first {args.max_images:,} images (--max_images)")
    
    # Determine number of workers
    num_workers = args.workers or min(multiprocessing.cpu_count(), 8)
    print(f"✓ Using {num_workers} worker threads")
    
    # Process images with progress bar
    print(f"\n{'=' * 60}")
    print("Processing Images")
    print("=" * 60)
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    errors = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                resize_and_save,
                input_path,
                output_dir / input_path.name,
                args.size,
                args.quality,
                args.force
            ): input_path
            for input_path in image_paths
        }
        
        # Process results with progress bar
        with tqdm(total=len(futures), desc="Resizing", unit="img") as pbar:
            for future in as_completed(futures):
                filename, status = future.result()
                
                if status == 'processed':
                    processed_count += 1
                elif status == 'skipped':
                    skipped_count += 1
                else:  # error
                    error_count += 1
                    errors.append((filename, status))
                
                pbar.update(1)
                pbar.set_postfix({
                    'done': processed_count,
                    'skip': skipped_count,
                    'err': error_count
                })
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    print(f"  ✓ Processed:  {processed_count:,} images")
    print(f"  → Skipped:    {skipped_count:,} images (already exist)")
    if error_count > 0:
        print(f"  ✗ Errors:     {error_count:,} images")
        print("\n  Error details:")
        for filename, err in errors[:10]:  # Show first 10 errors
            print(f"    - {filename}: {err}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more errors")
    
    print(f"\n{'=' * 60}")
    print("Next Steps")
    print("=" * 60)
    print(f"\nTo use the preprocessed images, run training with:")
    print(f"  python src/train/train_cnn.py --celeba_dir {output_dir}")
    print(f"\nOr update your dataset initialization:")
    print(f"  dataset = SteganographyDataset(celeba_dir='{output_dir}')")
    print("=" * 60)
    
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    exit(main())
