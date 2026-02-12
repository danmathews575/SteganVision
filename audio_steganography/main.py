import sys
import os

# Add LSB_ directory to sys.path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
lsb_dir = os.path.join(current_dir, "LSB_")
sys.path.insert(0, lsb_dir)

try:
    import app
except ImportError as e:
    print(f"[ERROR] Critical Error: Could not import audio application from {lsb_dir}")
    print(f"Detail: {e}")
    sys.exit(1)

if __name__ == "__main__":
    app.main()
