
import os
import sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal

# Add the parent directory to sys.path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ai_guided_lsb.importance_model import compute_importance_map, get_embedding_order
from ai_guided_lsb.encoder import encode
from ai_guided_lsb.utils import flatten_audio, unflatten_audio

def generate_visualization(cover_path, secret_path, output_image_path):
    print(f"Generating visualization for: {cover_path}")
    
    # 1. Run actual encoding to get stego audio and stats
    # Create a temp output path
    stego_path = os.path.join(os.path.dirname(output_image_path), "temp_stego_viz.wav")
    
    print("Running encoding...")
    stats = encode(
        cover_path=cover_path,
        secret_path=secret_path,
        output_path=stego_path,
        bits_per_sample=1,
        use_compression=False # Disable compression to make bit visualization easier if needed
    )
    
    # 2. Load Audio Data
    print("Loading audio data...")
    cover_audio, sr = sf.read(cover_path, dtype='int16')
    stego_audio, _ = sf.read(stego_path, dtype='int16')
    
    # Flatten
    cover_flat, _ = flatten_audio(cover_audio)
    stego_flat, _ = flatten_audio(stego_audio)
    
    # 3. Compute Importance Map (mocking internal process)
    print("Computing importance map...")
    cover_float = cover_flat.astype(np.float64)
    importance_map = compute_importance_map(cover_float, sr, bits_per_sample=1)
    
    # 4. Create Visualization
    print("Creating plots...") # Use a dark style for "cool" look
    plt.style.use('dark_background')
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
    
    # --- Plot A: Waveform with Importance Overlay ---
    ax1 = fig.add_subplot(gs[0, :])
    
    # Downsample for plotting if too long
    factor = max(1, len(cover_flat) // 10000)
    time_axis = np.arange(len(cover_flat)) / sr
    
    ax1.plot(time_axis[::factor], cover_flat[::factor], color='cyan', alpha=0.6, label='Audio (Waveform)')
    
    # Overlay importance (normalized to audio scale for visibility)
    imp_viz = importance_map[::factor]
    # Scale importance to fit in the plot nicely (e.g. 80% of max amp)
    max_amp = np.max(np.abs(cover_flat))
    ax1.fill_between(time_axis[::factor], -max_amp, max_amp, 
                     where=(imp_viz > 0.7), color='lime', alpha=0.1, label='High Importance (Safe to Embed)')
    ax1.fill_between(time_axis[::factor], -max_amp, max_amp, 
                     where=(imp_viz < 0.3), color='red', alpha=0.1, label='Low Importance (Unsafe)')

    ax1.set_title("1. AI-Guided Analysis: Identifying Safe Embedding Regions", fontsize=14, color='white')
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, time_axis[-1])
    
    # --- Plot B: Spectrogram Comparison ---
    # We'll show Cover Spectrogram and Difference Spectrogram
    
    ax2 = fig.add_subplot(gs[1, 0])
    f, t, Sxx = signal.spectrogram(cover_flat, sr)
    im2 = ax2.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    ax2.set_title("2a. Cover Audio Spectrogram", fontsize=12)
    ax2.set_ylabel("Frequency [Hz]")
    ax2.set_xlabel("Time [sec]")
    fig.colorbar(im2, ax=ax2, label='dB')

    # Difference Spectrogram (Stego - Cover)
    ax3 = fig.add_subplot(gs[1, 1])
    diff_signal = stego_flat.astype(np.float64) - cover_flat.astype(np.float64)
    f_d, t_d, Sxx_d = signal.spectrogram(diff_signal, sr)
    # Use a different colormap for noise
    im3 = ax3.pcolormesh(t_d, f_d, 10 * np.log10(Sxx_d + 1e-10), shading='gouraud', cmap='cool')
    ax3.set_title("2b. Residual Noise (Stego - Cover)", fontsize=12)
    ax3.set_ylabel("Frequency [Hz]")
    ax3.set_xlabel("Time [sec]")
    fig.colorbar(im3, ax=ax3, label='dB')
    
    # --- Plot C: Zoomed In LSB View ---
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Find a region where embedding happened
    diff_indices = np.where(cover_flat != stego_flat)[0]
    if len(diff_indices) > 0:
        center_idx = diff_indices[0] # Take the first modification
        window = 50
        start = max(0, center_idx - window)
        end = min(len(cover_flat), center_idx + window)
        
        x_zoom = np.arange(start, end)
        
        # Plot with markers
        ax4.plot(x_zoom, cover_flat[start:end], 'o-', label='Original', color='gray', alpha=0.5, markersize=4)
        ax4.plot(x_zoom, stego_flat[start:end], 'x--', label='Stego (Modified)', color='lime', alpha=0.8, markersize=6)
        
        # Highlight differences
        for idx in range(start, end):
            if cover_flat[idx] != stego_flat[idx]:
                ax4.annotate('', xy=(idx, stego_flat[idx]), xytext=(idx, cover_flat[idx]),
                             arrowprops=dict(arrowstyle='->', color='yellow', lw=2))
        
        ax4.set_title(f"3. Micro-view: LSB Modification (Zoomed at sample {center_idx})", fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "No differences found (payload might be too small or empty)", 
                 ha='center', va='center', transform=ax4.transAxes)
    
    # --- Plot D: Results & Metrics ---
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    text_str = (
        f"RESULTS SUMMARY\n"
        f"----------------------------------------\n"
        f"Encoding Status:    {stats['status']}\n\n"
        f"Imperceptibility Metrics:\n"
        f"  • SNR:            {stats['snr_db']:.2f} dB  (> 80 dB is excellent)\n"
        f"  • MSE:            {stats['mse']:.6f}\n\n"
        f"Capacity & Efficiency:\n"
        f"  • Payload Size:   {stats['payload_bytes']} bytes\n"
        f"  • Utilization:    {stats['capacity_utilization']:.2f}%\n"
        f"  • Bits/Sample:    {stats['bits_per_sample']}\n\n"
        f"AI Guidance Stats:\n"
        f"  • Mean Importance: {stats['importance_stats']['mean']:.4f}\n"
        f"  • Std Dev:        {stats['importance_stats']['std']:.4f}\n"
    )
    
    ax5.text(0.1, 0.5, text_str, fontsize=14, family='monospace', color='white',
             bbox=dict(facecolor='black', alpha=0.5, edgecolor='lime', boxstyle='round,pad=1'),
             transform=ax5.transAxes, va='center')
    ax5.set_title("4. Validation Metrics", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=100, bbox_inches='tight')
    print(f"Visualization saved to: {output_image_path}")
    
    # Clean up temp file
    if os.path.exists(stego_path):
        os.remove(stego_path)

if __name__ == "__main__":
    # Define paths
    base_dir = r"c:\MajorP\audio_steganography"
    # Use 'short_cover.wav' if available for better viz, else '2bit_cover.wav'
    # 'test_data/short_cover.wav' is in ai_guided_lsb/test_data
    
    # Let's try to find a good cover file
    # We prefer a longer one like 'short_cover.wav' to show more signal variety
    cover_file = os.path.join(base_dir, "ai_guided_lsb", "test_data", "short_cover.wav")
    secret_file = os.path.join(base_dir, "ai_guided_lsb", "test_data", "short_secret.wav") # Using audio as secret for fun
    
    if not os.path.exists(cover_file):
        print(f"Cover file not found: {cover_file}, falling back to 2bit_cover.wav")
        cover_file = os.path.join(base_dir, "ai_guided_lsb", "test_data", "2bit_cover.wav")
        secret_file = os.path.join(base_dir, "ai_guided_lsb", "test_data", "2bit_secret.bin")
        
    output_img = os.path.join(base_dir, "scripts", "steganography_viz_result.png")
    
    generate_visualization(cover_file, secret_file, output_img)
