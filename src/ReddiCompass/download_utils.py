from pathlib import Path
import sys, subprocess
from shutil import copyfile

def __import_huggingface__():
    try:
        import huggingface_hub
        return
    except ImportError:
        print("[INFO] huggingface_hub not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])

    import huggingface_hub

# --- STEP 2: Main download function ---
def download_from_huggingface(huggingface_repo: str, filename: str, saving_path: str) -> bool:
    """
    Downloads a file from a public Hugging Face model repo.
    
    Args:
        huggingface_repo (str):
        filename (str): name of the file inside the repo
        saving_path (str): full local path where to store the file

    Returns:
        bool: True if file downloaded/copied; False if already existed
    """
    __import_huggingface__()
    from huggingface_hub import hf_hub_download
    saving_path = Path(saving_path)
    if saving_path.exists():
        print(f"[INFO] File already exists at: {saving_path}")
        return False

    try:
        print("Downloading word2vec model. It may take some time, it's a 4.6GB file.")
        cached_file = hf_hub_download(
            repo_id=huggingface_repo,
            filename=filename,
            repo_type="model"
        )
        saving_path.parent.mkdir(parents=True, exist_ok=True)
        copyfile(cached_file, saving_path)
        print(f"[INFO] Downloaded and saved to: {saving_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to download from Hugging Face: {e}")
        return False
