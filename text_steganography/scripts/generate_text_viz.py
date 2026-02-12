
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ai_guided_lsb.importance_model import compute_importance_map, visualize_importance_map
from ai_guided_lsb.encoder import encode

def generate_text_visualization(cover_path, secret_text, output_image_path):
    print(f"Generating visualization for: {cover_path}")
    
    # 1. Run Encoding
    stego_path = os.path.join(os.path.dirname(output_image_path), "temp_stego_viz.png")
    
    print("Running encoding...")
    success, message = encode(
        cover_image_path=cover_path,
        text=secret_text,
        output_path=stego_path,
        bits_per_channel=1,
        save_importance_map=False
    )
    
    if not success:
        print(f"Encoding failed: {message}")
        return

    # 2. Load Images
    print("Loading image data...")
    cover_img = Image.open(cover_path).convert('RGB')
    cover_array = np.array(cover_img)
    
    stego_img = Image.open(stego_path).convert('RGB')
    stego_array = np.array(stego_img)
    
    # 3. Compute Importance Map
    print("Computing importance map...")
    importance_map = compute_importance_map(cover_array)
    heatmap = visualize_importance_map(importance_map)
    
    # 4. Compute Differences
    diff = np.abs(stego_array.astype(int) - cover_array.astype(int))
    diff_mask = np.sum(diff, axis=2) > 0 # Where any channel changed
    
    # 5. Create Visualization
    print("Creating plots...")
    plt.style.use('dark_background')
    
    fig = plt.figure(figsize=(16, 12)) 
    gs = gridspec.GridSpec(2, 3)
    
    # --- Plot A: Original Image ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cover_array)
    ax1.set_title("1. Cover Image", fontsize=14, color='white')
    ax1.axis('off')
    
    # --- Plot B: Importance Map ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(heatmap)
    ax2.set_title("2. AI Importance Map\n(Bright = Safe to Embed)", fontsize=14, color='white')
    ax2.axis('off')

    # --- Plot C: Difference Map (Amplified) ---
    ax3 = fig.add_subplot(gs[0, 2])
    # Create an RGB image where changed pixels are bright pink
    diff_viz = np.zeros_like(cover_array)
    diff_viz[diff_mask] = [255, 0, 255] # Magenta for changes
    
    # Dilate slightly to make individual pixels visible if they are sparse
    kernel = np.ones((3,3), np.uint8)
    diff_viz = cv2.dilate(diff_viz, kernel, iterations=1)
    
    ax3.imshow(diff_viz)
    ax3.set_title("3. Embedding Locations\n(Amplified for visibility)", fontsize=14, color='white')
    ax3.axis('off')
    
    # --- Plot D: Zoomed LSB Details ---
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Find a region with changes to zoom into
    rows, cols = np.where(diff_mask)
    if len(rows) > 0:
        # Pick a center point
        center_idx = len(rows) // 2 
        cy, cx = rows[center_idx], cols[center_idx]
        
        # Define window (20x20 pixels)
        window = 10
        y1, y2 = max(0, cy-window), min(cover_array.shape[0], cy+window)
        x1, x2 = max(0, cx-window), min(cover_array.shape[1], cx+window)
        
        zoom_patch = cover_array[y1:y2, x1:x2].copy()
        
        # Draw boxes around modified pixels
        ax4.imshow(zoom_patch)
        
        # Overlay highlights
        patch_diff = diff_mask[y1:y2, x1:x2]
        py, px = np.where(patch_diff)
        
        ax4.scatter(px, py, s=100, facecolors='none', edgecolors='lime', linewidth=2, label='Modified Pixel')
        
        ax4.set_title(f"4. Micro-view: Zoomed Region\n(Green Box = Modified Pixel)", fontsize=14)
        ax4.legend(loc='lower left', fontsize=10)
        # Add grid lines
        ax4.set_xticks(np.arange(-.5, (x2-x1), 1), minor=True)
        ax4.set_yticks(np.arange(-.5, (y2-y1), 1), minor=True)
        ax4.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.3)
        ax4.tick_params(which='minor', bottom=False, left=False) # Hide minor ticks
        ax4.set_xticks([]) # Hide major ticks
        ax4.set_yticks([])
    else:
        ax4.text(0.5, 0.5, "No pixels modified", ha='center', va='center')

    # --- Plot E: Histogram / Metrics ---
    ax5 = fig.add_subplot(gs[1, 1:]) # Span 2 columns
    ax5.axis('off')
    
    # Calculate Metrics
    mse = np.mean((cover_array.astype(float) - stego_array.astype(float)) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 100
    pixels_modified = np.sum(diff_mask)
    total_pixels = cover_array.shape[0] * cover_array.shape[1]
    
    text_str = (
        f"RESULTS SUMMARY\n"
        f"---------------------------------------------\n"
        f"Status:             ✅ Success\n\n"
        f"Imperceptibility Metrics:\n"
        f"  • PSNR:             {psnr:.2f} dB  (> 40 dB is invisible)\n"
        f"  • MSE:              {mse:.6f}\n\n"
        f"Embedding Stats:\n"
        f"  • Secret Text:      \"{secret_text[:20]}...\" ({len(secret_text)} chars)\n"
        f"  • Pixels Modified:  {pixels_modified:,} / {total_pixels:,}\n"
        f"  • Modification %:   {(pixels_modified/total_pixels)*100:.4f}%\n\n"
        f"AI Strategy:\n"
        f"  • Edges/Textures prioritized\n"
        f"  • Smooth regions avoided\n"
    )
    
    ax5.text(0.1, 0.5, text_str, fontsize=16, family='monospace', color='white',
             bbox=dict(facecolor='black', alpha=0.5, edgecolor='magenta', boxstyle='round,pad=1'),
             transform=ax5.transAxes, va='center')
    ax5.set_title("5. Performance Metrics", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=100, bbox_inches='tight')
    print(f"Visualization saved to: {output_image_path}")
    
    # Cleanup
    if os.path.exists(stego_path):
        os.remove(stego_path)

if __name__ == "__main__":
    # Define paths
    base_dir = r"c:\MajorP\text_steganography"
    cover_file = os.path.join(base_dir, "sample1.png")
    
    if not os.path.exists(cover_file):
        # Fallback if sample1 doesn't exist, try looking in test_data or creating a dummy
        print(f"Warning: {cover_file} not found.")
        # Try to find any png
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".png") and "output" not in file:
                    cover_file = os.path.join(root, file)
                    break 
            if os.path.exists(cover_file): break
            
    if not os.path.exists(cover_file):
        print("Error: No suitable PNG cover image found.")
        sys.exit(1)
        
    # Longer secret text to ensure we use enough pixels to be visible
    secret_text = (
        "This is a secret message demonstrating AI-Guided LSB Steganography. "
        "The system selects high-texture regions to hide this text, "
        "ensuring it remains invisible to the naked eye. " * 5
    )
    
    output_img = os.path.join(base_dir, "scripts", "text_stego_viz_result.png")
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    
    generate_text_visualization(cover_file, secret_text, output_img)
