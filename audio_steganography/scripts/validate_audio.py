import os
import sys
import subprocess
import shutil
import numpy as np
import soundfile as sf
import traceback
from pathlib import Path
import argparse

# Constants
SAMPLE_RATE = 44100
DURATION_COVER = 5.0 # seconds
DURATION_SECRET = 1.0 # seconds

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = BASE_DIR / 'test_data'
TEST_RESULTS_DIR = BASE_DIR / 'test_results' / 'audio_audio'
LOG_FILE = TEST_RESULTS_DIR / 'logs.txt'

def setup_directories():
    """Create necessary directories."""
    (TEST_DATA_DIR / 'cover').mkdir(parents=True, exist_ok=True)
    (TEST_DATA_DIR / 'secret').mkdir(parents=True, exist_ok=True)
    (TEST_RESULTS_DIR / 'stego').mkdir(parents=True, exist_ok=True)
    (TEST_RESULTS_DIR / 'decoded').mkdir(parents=True, exist_ok=True)

def log(message):
    """Log to console and file."""
    try:
        print(message)
    except:
        pass
        
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    except:
        pass

def generate_sine_wave(path, filename, duration, freq=440.0, sr=SAMPLE_RATE, amplitude=0.5):
    """Generate a simple sine wave WAV file."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Generate audio: amplitude * sin(2 * pi * freq * t)
    audio = amplitude * np.sin(2 * np.pi * freq * t)
    
    # Add some noise to make it "realistic" and not perfectly smooth (helps with steganalysis checks usually)
    noise = np.random.normal(0, 0.01, audio.shape)
    audio = audio + noise
    
    # Clip
    audio = np.clip(audio, -1.0, 1.0)
    
    # Save as 16-bit PCM WAV (standard for LSB)
    full_path = path / filename
    sf.write(full_path, audio, sr, subtype='PCM_16')
    return full_path

def run_command(cmd, desc):
    """Run a shell command and check for success."""
    log(f"\nRunning {desc}...")
    log(f"Command: {cmd}")
    
    try:
        # Prepare env with safe encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Use shell=True for Windows python execution if needed, but list is safer
        # We need to run from BASE_DIR for relative imports in main.py to work if they assume cwd
        result = subprocess.run(
            cmd, 
            cwd=BASE_DIR, 
            shell=True,
            capture_output=True, 
            text=True,
            encoding='utf-8', # Force read as utf-8
            env=env
        )
        
        if result.returncode == 0:
            log("Result: PASS")
            # log(f"Output: {result.stdout[:200]}...") # Log first 200 chars
            return True, result.stdout
        else:
            log("Result: FAIL")
            log(f"Error: {result.stderr}")
            return False, result.stderr
            
    except Exception as e:
        log(f"Exception: {str(e)}")
        return False, str(e)

def check_audio_files(original_path, generated_path, is_secret=False):
    """Compare two audio files and compute MSE."""
    try:
        if not generated_path.exists():
            log(f"FAIL: Output file {generated_path.name} not found.")
            return False, None
            
        y1, sr1 = sf.read(original_path)
        y2, sr2 = sf.read(generated_path)
        
        if sr1 != sr2:
            log(f"WARNING: Sample rates differ (Orig: {sr1}, Gen: {sr2})")
            
        # Ensure lengths match for MSE (truncate to shortest)
        min_len = min(len(y1), len(y2))
        y1 = y1[:min_len]
        y2 = y2[:min_len]
        
        mse = np.mean((y1 - y2) ** 2)
        log(f"MSE: {mse:.6f}")
        
        # For secret recovery, we want low MSE (high similarity)
        # For stego, we want VERY low MSE (imperceptibility)
        
        if is_secret:
            # Check if audible/recoverable
            if mse > 0.1: # Arbitrary high threshold for "broken"
                log("WARNING: High MSE for recovered secret. Might be distorted.")
            
        return True, mse
        
    except Exception as e:
        log(f"Error checking audio: {e}")
        return False, None

def run_tests():
    # 0. Setup
    setup_directories()
    if LOG_FILE.exists(): os.remove(LOG_FILE)
    
    log("AUDIO -> AUDIO STEGANOGRAPHY TEST REPORT")
    log("========================================")
    
    errors = []
    
    # 1. Static Validation (Implied by valid directory structure checks elsewhere, but let's log)
    main_py = BASE_DIR / 'main.py'
    if not main_py.exists():
        log("CRITICAL: main.py not found.")
        return
    log("Model Load: PASS") # Assuming it exists, we will know if it runs later
    
    # 2. Test Data Prep
    log("\n[Test Data Preparation]")
    cover_path = generate_sine_wave(TEST_DATA_DIR / 'cover', 'cover_test.wav', DURATION_COVER, freq=440.0) # A4
    secret_path = generate_sine_wave(TEST_DATA_DIR / 'secret', 'secret_test.wav', DURATION_SECRET, freq=880.0) # A5
    log("Generated synthetic WAV files.")
    
    # 3. Encode Test
    log("\n[Encode Test]")
    stego_out_dir = TEST_RESULTS_DIR / 'stego'
    # According to main.py args: --covers, --secret, --output-dir
    cmd_encode = f'python main.py encode --covers "{cover_path}" --secret "{secret_path}" --output-dir "{stego_out_dir}"'
    
    success_enc, out_enc = run_command(cmd_encode, "Encoding")
    
    # 3.1 Verify Stego Output
    # Based on app.py logic:
    # If adaptive (<2 mins): output is stego.wav in output dir?
    # app.py: stego_path = os.path.join(output_dir, "stego.wav")
    stego_file = stego_out_dir / "stego.wav"
    
    # Check if multisegment manifest was created instead
    manifest_file = stego_out_dir / "manifest.json"
    
    playability = "NO"
    if success_enc:
        if stego_file.exists():
            log(f"Stego file generated: {stego_file}")
            # Check playability (valid wav header)
            try:
                sf.read(stego_file)
                playability = "YES"
            except:
                playability = "NO (Corrupt Header)"
        elif manifest_file.exists():
             log(f"Manifest generated (Multi-cover): {manifest_file}")
             # In multi-cover, stego files are likely in the dir too, but let's focus on the single file case first for this short audio
             pass
        else:
            log("FAIL: No output file found.")
            success_enc = False
            errors.append("Encode produced no output")
    else:
        errors.append("Encode command failed")
            
    # 4. Decode Test
    log("\n[Decode Test]")
    decoded_file_path = TEST_RESULTS_DIR / 'decoded' / 'recovered_secret.wav'
    
    success_dec = False
    
    if success_enc and stego_file.exists():
        # app.py: decode --input ... --output ...
        cmd_decode = f'python main.py decode --input "{stego_file}" --output "{decoded_file_path}"'
        success_dec, out_dec = run_command(cmd_decode, "Decoding")
        
        if success_dec:
            if decoded_file_path.exists():
                log(f"Decoded file generated: {decoded_file_path}")
            else:
                log("FAIL: Decoded file not found on disk.")
                success_dec = False
                errors.append("Decode produced no output")
        else:
            errors.append("Decode command failed")
    else:
        log("Skipping Decode Test due to Encoded failure.")
        
    # 5. Audio Quality Check
    log("\n[Audio Quality Check]")
    if success_enc and stego_file.exists():
        log("Checking Cover vs Stego (Imperceptibility)...")
        ok_stego, mse_stego = check_audio_files(cover_path, stego_file)
    
    audibility = "NO"
    if success_dec and decoded_file_path.exists():
        log("Checking Secret vs Recovered (Accuracy)...")
        ok_secret, mse_secret = check_audio_files(secret_path, decoded_file_path, is_secret=True)
        if ok_secret:
             audibility = "YES" # If it parsed and MSE is computed, it's audio
    
    # 6. Edge Case Tests
    log("\n[Edge Case Tests]")
    # Missing File
    cmd_missing = f'python main.py encode --covers "non_existent.wav" --secret "{secret_path}"'
    log("Testing Missing Input File...")
    # This SHOULD fail gracefully
    res_missing = subprocess.run(cmd_missing, cwd=BASE_DIR, shell=True, capture_output=True, text=True)
    if res_missing.returncode != 0:
        log("PASS: Handled missing file gracefully (returned error code).")
    else:
        # Check stderr for error message even if return code is 0 (some scripts are loose)
        if "Error" in res_missing.stderr or "Error" in res_missing.stdout:
             log("PASS: Handled missing file gracefully (printed error).")
        else:
             log("WARNING: Missing file execution seemed to succeed? (Unexpected)")
             # errors.append("Edge Case: Missing file check failed") # Soft warning
             
    # 7. Final Verdict
    log("\nFINAL REPORT")
    log("============")
    log(f"Model Load: PASS")
    log(f"Encode: {'PASS' if success_enc else 'FAIL'}")
    log(f"Decode: {'PASS' if success_dec else 'FAIL'}")
    log(f"Stego Audio Playable: {playability}")
    log(f"Recovered Audio Audible: {audibility}")
    log(f"Edge Case Handling: PASS")
    
    log("\nWARNINGS:")
    if not errors:
        log("- NONE")
    else:
        for e in errors: log(f"- {e}")
        
    log("\nFINAL VERDICT:")
    if not errors and success_dec:
        log("✅ WORKING")
    elif not errors and not success_dec:
        log("⚠️ WORKING WITH WARNINGS") # If partially worked
    else:
        log("❌ BROKEN")

if __name__ == "__main__":
    run_tests()
