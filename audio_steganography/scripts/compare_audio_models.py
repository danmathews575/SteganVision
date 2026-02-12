import os
import sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import io
from scipy.signal import spectrogram

# --- SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# Add AI-Guided LSB path
ai_guided_path = os.path.join(project_root, 'audio_steganography')
if ai_guided_path not in sys.path:
    sys.path.insert(0, ai_guided_path)

# Add Traditional LSB path (trickier due to relative imports in its src)
traditional_lsb_root = os.path.join(project_root, 'audio_steganography', 'LSB_', 'audio_steganography')
traditional_lsb_src = os.path.join(traditional_lsb_root, 'src')
if traditional_lsb_src not in sys.path:
    sys.path.insert(0, traditional_lsb_src)

# --- IMPORTS ---
try:
    from ai_guided_lsb.encoder import encode as ai_encode
    # Mock traditional utils if needed or ensure path correctness
    # The traditional LSB code does 'from utils.lsb_utils ...', so 'utils' must be importable
    # We added 'src' to path, so 'modes' and 'utils' should be top-level importable
    from modes.lsb.encoder import encode_file as trad_encode
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure you are running this script with the correct PYTHONPATH or directory structure.")
    sys.exit(1)

# --- METRIC FUNCTIONS ---
def calculate_mse(original, stego):
    return np.mean((original - stego) ** 2)

def calculate_snr(original, stego):
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - stego) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def plot_spectrogram(audio, sr, title, output_path):
    plt.figure(figsize=(10, 4))
    f, t, Sxx = spectrogram(audio, sr)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title)
    plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_waveform_diff(original, stego, title, output_path):
    diff = original - stego
    plt.figure(figsize=(10, 4))
    plt.plot(diff, label='Difference (Noise)', alpha=0.7, color='red', linewidth=0.5)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude Difference')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# --- MAIN COMPARISON ROUTINE ---
def run_comparison():
    print("🚀 Starting Audio Steganography Model Comparison...")
    
    # Configuration
    cover_path = os.path.join(ai_guided_path, 'ai_guided_lsb', 'test_data', 'short_cover.wav')
    output_dir = os.path.join(ai_guided_path, 'scripts', 'output', 'comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate Secret (Binary)
    secret_text = "This is a secret message for evaluating the performance of AI-guided vs Traditional LSB steganography." * 50
    secret_path = os.path.join(output_dir, 'secret.txt')
    with open(secret_path, 'w') as f:
        f.write(secret_text)
        
    print(f"📂 Cover Audio: {cover_path}")
    print(f"📄 Secret Size: {len(secret_text)} bytes")

    # 1. AI-Guided Encoding
    print("\n---------------------------------------------------")
    print("🤖 Running AI-Guided LSB Model...")
    ai_output_path = os.path.join(output_dir, 'stego_ai.wav')
    
    try:
        ai_result = ai_encode(
            cover_path=cover_path,
            secret_path=secret_path,
            output_path=ai_output_path,
            bits_per_sample=1,
            use_compression=True  # AI model supports compression
        )
        print("✅ AI-Guided Encoding Complete")
    except Exception as e:
        print(f"❌ AI-Guided Failed: {e}")
        return

    # 2. Traditional LSB Encoding
    print("\n---------------------------------------------------")
    print("🔨 Running Traditional LSB Model...")
    trad_output_path = os.path.join(output_dir, 'stego_traditional.wav')
    
    try:
        trad_result = trad_encode(
            cover_path=cover_path,
            secret_path=secret_path,
            output_path=trad_output_path,
            bits_per_sample=1
        )
        print("✅ Traditional Encoding Complete")
    except Exception as e:
        print(f"❌ Traditional Failed: {e}")
        # Continue if possible, or exit? Let's try to analyze AI at least.
        trad_result = None

    # 3. Load Audio for Analysis
    original, sr = sf.read(cover_path, dtype='float32')
    original_flat = original.flatten()
    
    stego_ai, _ = sf.read(ai_output_path, dtype='float32')
    stego_ai_flat = stego_ai.flatten()
    
    if trad_result:
        stego_trad, _ = sf.read(trad_output_path, dtype='float32')
        stego_trad_flat = stego_trad.flatten()
    else:
        stego_trad_flat = None

    # 4. Calculate Final Independent Metrics
    # (Re-calculating specifically on float32 representation for fairness)
    print("\n---------------------------------------------------")
    print("📊 PERFORMANCE RESULTS")
    print("---------------------------------------------------")
    
    # AI Metrics
    ai_mse = calculate_mse(original_flat, stego_ai_flat)
    ai_snr = calculate_snr(original_flat, stego_ai_flat)
    
    print(f"🤖 AI-Guided LSB:")
    print(f"   • MSE: {ai_mse:.2e}")
    print(f"   • SNR: {ai_snr:.2f} dB")
    print(f"   • Capacity Used: {ai_result.get('capacity_utilization', 0):.2f}%")
    
    # Traditional Metrics
    if stego_trad_flat is not None:
        trad_mse = calculate_mse(original_flat, stego_trad_flat)
        trad_snr = calculate_snr(original_flat, stego_trad_flat)
        
        print(f"🔨 Traditional LSB:")
        print(f"   • MSE: {trad_mse:.2e}")
        print(f"   • SNR: {trad_snr:.2f} dB")
        # Traditional result dict usage might depend on exact return structure
        print(f"   • Capacity Used: {trad_result.get('capacity_utilization', 0):.2f}%")
        
        # Improvement
        snr_imp = ai_snr - trad_snr
        print(f"\n📈 Improvement (AI vs Traditional):")
        print(f"   • SNR Gain: {snr_imp:+.2f} dB")
        if ai_mse < trad_mse:
            print("   • MSE: BETTER (Lower)")
        else:
            print("   • MSE: WORSE (Higher)")
    else:
        print("⚠️ Traditional metrics skipped due to failure.")

    # 5. Generate Visualizations
    print("\n🎨 Generating Visualizations...")
    
    # Spectrograms
    plot_spectrogram(original_flat[:sr*5], sr, "Original Spectral Content (0-5s)", os.path.join(output_dir, 'spec_original.png'))
    plot_spectrogram(stego_ai_flat[:sr*5], sr, "AI-Stego Spectral Content (0-5s)", os.path.join(output_dir, 'spec_ai.png'))
    if stego_trad_flat is not None:
        plot_spectrogram(stego_trad_flat[:sr*5], sr, "Traditional Stego Spectral Content (0-5s)", os.path.join(output_dir, 'spec_traditional.png'))
        
    # Diff Plots
    plot_waveform_diff(original_flat[:1000], stego_ai_flat[:1000], "AI Noise Residual (First 1000 samples)", os.path.join(output_dir, 'diff_ai.png'))
    if stego_trad_flat is not None:
        plot_waveform_diff(original_flat[:1000], stego_trad_flat[:1000], "Traditional Noise Residual (First 1000 samples)", os.path.join(output_dir, 'diff_traditional.png'))

    print(f"\n✅ Analysis Complete. Results saved to: {output_dir}")

if __name__ == "__main__":
    run_comparison()
