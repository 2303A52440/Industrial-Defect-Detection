"""
CLI Demo Simulator Script for Edge AI Quality Control System.
Simulates real-time inspection of manufactured metal components on Arm SoC / Raspberry Pi 5.
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pipeline import EdgeQCPipeline

def generate_demo_surface(defect_type: str = "random") -> np.ndarray:
    """Generates synthetic metal surface texture with controlled defect patterns."""
    img = np.full((640, 640, 3), 180, dtype=np.uint8)
    noise = np.random.normal(0, 12, (640, 640)).astype(np.uint8)
    img = cv2.add(img, cv2.merge([noise, noise, noise]))

    if defect_type == "scratches":
        cv2.line(img, (120, 200), (450, 480), (30, 30, 30), 4)
        cv2.line(img, (140, 190), (470, 470), (40, 40, 40), 2)
    elif defect_type == "crazing":
        for _ in range(8):
            pt1 = (np.random.randint(200, 400), np.random.randint(200, 400))
            pt2 = (pt1[0] + np.random.randint(-40, 40), pt1[1] + np.random.randint(-40, 40))
            cv2.line(img, pt1, pt2, (20, 20, 20), 2)
    elif defect_type == "patches":
        cv2.ellipse(img, (320, 300), (100, 60), 30, 0, 360, (70, 70, 70), -1)
    elif defect_type == "pitted_surface":
        for _ in range(15):
            center = (np.random.randint(150, 500), np.random.randint(150, 500))
            cv2.circle(img, center, np.random.randint(5, 12), (10, 10, 10), -1)
    elif defect_type == "inclusion":
        pts = np.array([[250, 250], [320, 230], [350, 310], [280, 340]], np.int32)
        cv2.fillPoly(img, [pts], (15, 15, 15))

    return img

def run_inspection_demo(num_components: int = 5):
    print("=" * 75)
    print(" EDGE-AI INDUSTRIAL QUALITY CONTROL ASSISTANT - CLI DEMO")
    print(" Architecture: Dual Pipeline (Hailo/OpenCV Vision + Llama ExecuTorch)")
    print(" Target Hardware: Raspberry Pi 5 / Arm SoC")
    print("=" * 75)
    print()

    pipeline = EdgeQCPipeline()
    defect_types = ["scratches", "crazing", "patches", "pitted_surface", "clean"]

    for i in range(num_components):
        dtype = defect_types[i % len(defect_types)]
        part_id = f"CMP-DEMO-{1001 + i}"
        
        img = generate_demo_surface(dtype)
        
        start_t = time.perf_counter()
        detection, advisory, log_record = pipeline.process_component_inspection(img, component_id=part_id)
        total_time_ms = (time.perf_counter() - start_t) * 1000.0

        print(f"---------------------------------------------------------------------------")
        print(f" Component ID       : {part_id}")
        print(f" QC Decision        : {log_record['qc_status']} (Pass Check: {detection.pass_quality_check})")
        print(f" Vision Engine      : {detection.backend_used} | Latency: {detection.inference_time_ms:.1f} ms")
        print(f" Defects Detected   : {len(detection.defects)} (Type: {log_record['primary_defect']}, Severity: {log_record['max_severity']})")
        print(f" Llama Advisory     : [{advisory.action_type}]")
        print(f" Recommendation     : {advisory.primary_recommendation}")
        print(f" Technical Action   : {advisory.technical_details[:100]}...")
        if advisory.safety_warning:
            print(f" Safety Warning     : {advisory.safety_warning}")
        print(f" Total Dual Latency : {total_time_ms:.1f} ms")
        print(f"---------------------------------------------------------------------------")
        print()
        time.sleep(0.1)

    print("[SUCCESS] Demo Simulation Completed Successfully!")
    print(f"[INFO] Log records saved to SQLite DB at: {pipeline.logger.db_path}")

if __name__ == "__main__":
    run_inspection_demo()
