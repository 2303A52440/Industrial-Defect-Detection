"""
Dataset Download & Sample Preparation Utility Script.
Fetches NEU Surface Defect sample images and structures dataset folders.
"""

import os
import urllib.request
import numpy as np
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = BASE_DIR / "data" / "samples"

def setup_sample_dataset():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Preparing sample surface dataset in: {SAMPLE_DIR}")

    # Generate reference sample images for each NEU Surface Defect class
    defect_classes = ["scratches", "crazing", "patches", "pitted_surface", "inclusion", "clean"]
    
    for d_type in defect_classes:
        img = np.full((640, 640, 3), 180, dtype=np.uint8)
        noise = np.random.normal(0, 10, (640, 640)).astype(np.uint8)
        img = cv2.add(img, cv2.merge([noise, noise, noise]))

        if d_type == "scratches":
            cv2.line(img, (150, 150), (480, 480), (30, 30, 30), 4)
        elif d_type == "crazing":
            for _ in range(6):
                cv2.line(img, (200, 200), (250, 230), (25, 25, 25), 2)
        elif d_type == "patches":
            cv2.ellipse(img, (320, 300), (120, 70), 20, 0, 360, (75, 75, 75), -1)
        elif d_type == "pitted_surface":
            for _ in range(12):
                cv2.circle(img, (np.random.randint(150, 500), np.random.randint(150, 500)), 8, (10, 10, 10), -1)
        elif d_type == "inclusion":
            pts = np.array([[200, 200], [280, 180], [310, 260], [230, 280]], np.int32)
            cv2.fillPoly(img, [pts], (15, 15, 15))

        out_path = SAMPLE_DIR / f"sample_{d_type}.png"
        cv2.imwrite(str(out_path), img)
        print(f"  └─ Generated sample: {out_path.name}")

    print("✅ Sample surface defect dataset ready!")

if __name__ == "__main__":
    setup_sample_dataset()
