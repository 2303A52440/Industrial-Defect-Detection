"""
Utility script to create a clean submission ZIP file for the challenge.
Packages source code, technical report, README, and tests while excluding temporary cache files.
"""

import os
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = BASE_DIR / "edge_industrial_qc_submission.zip"

def create_submission_zip():
    print(f"[ZIP] Creating submission ZIP archive at: {OUTPUT_ZIP}")
    
    exclude_dirs = {"__pycache__", ".pytest_cache", ".git", "venv", ".venv"}
    exclude_extensions = {".pyc", ".pyo", ".pyd"}

    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(BASE_DIR):
            # Exclude unwanted directories in-place
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                file_path = Path(root) / file
                if file_path == OUTPUT_ZIP:
                    continue
                if file_path.suffix in exclude_extensions:
                    continue
                
                arcname = file_path.relative_to(BASE_DIR)
                zipf.write(file_path, arcname)
                print(f"  +-- Added: {arcname}")

    print(f"\n[SUCCESS] Submission ZIP ready for upload: {OUTPUT_ZIP}")

if __name__ == "__main__":
    create_submission_zip()
