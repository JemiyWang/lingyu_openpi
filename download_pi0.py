import os
import pathlib
import shutil
import subprocess
import sys

# Configuration
GCS_URL = "gs://openpi-assets/checkpoints/pi0_base/params"
# Default cache directory for openpi
CACHE_DIR = pathlib.Path("~/.cache/openpi").expanduser()

def main():
    print(f"Target URL: {GCS_URL}")
    
    # Parse URL to determine local path
    # gs://openpi-assets/checkpoints/pi0_base/params -> ~/.cache/openpi/openpi-assets/checkpoints/pi0_base/params
    parts = GCS_URL.replace("gs://", "").split("/")
    bucket = parts[0]
    path_suffix = "/".join(parts[1:])
    
    local_path = CACHE_DIR / bucket / path_suffix
    local_path = local_path.resolve()
    
    print(f"Local Path: {local_path}")
    
    # Check if target exists
    if local_path.exists():
        print(f"Target directory already exists: {local_path}")
        user_input = input("Do you want to delete it and re-download? (y/n): ").strip().lower()
        if user_input == 'y':
            if local_path.is_dir():
                shutil.rmtree(local_path)
            else:
                local_path.unlink()
            print("Deleted existing directory.")
        else:
            print("Aborted.")
            return

    # Ensure parent directory exists
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to use gcloud or gsutil for fast download
    success = False
    
    if shutil.which("gcloud"):
        print("\n[Method 1] Found 'gcloud', attempting download...")
        cmd = ["gcloud", "storage", "cp", "-r", GCS_URL, str(local_path)]
        try:
            subprocess.check_call(cmd)
            success = True
            print("\nDownload successful with gcloud!")
        except subprocess.CalledProcessError as e:
            print(f"gcloud failed: {e}")

    if not success and shutil.which("gsutil"):
        print("\n[Method 2] Found 'gsutil', attempting download...")
        cmd = ["gsutil", "-m", "cp", "-r", GCS_URL, str(local_path)]
        try:
            subprocess.check_call(cmd)
            success = True
            print("\nDownload successful with gsutil!")
        except subprocess.CalledProcessError as e:
            print(f"gsutil failed: {e}")

    if not success:
        print("\n[Error] Could not download using system tools (gcloud/gsutil).")
        print("Please install Google Cloud CLI for faster downloads, or use the slow Python fallback.")
        print("To install gcloud: https://cloud.google.com/sdk/docs/install")
        
        # Optional: Fallback to slow python download if really needed, 
        # but usually user asks for this script because python download is too slow.
        print("\nAttempting fallback to internal python downloader (may be slow)...")
        try:
            sys.path.append(os.getcwd()) # Ensure we can import src
            from src.openpi.shared.download import maybe_download
            maybe_download(GCS_URL)
            print("Download successful with Python downloader!")
        except ImportError:
            print("Could not import internal downloader. Please run this script from the project root.")
        except Exception as e:
            print(f"Python download failed: {e}")

if __name__ == "__main__":
    main()