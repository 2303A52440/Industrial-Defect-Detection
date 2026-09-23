"""
Configuration module for Edge-AI Industrial Quality Control System.
Defines system parameters, hardware backends, thresholds, and data paths.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
SAMPLE_IMAGES_DIR = DATA_DIR / "samples"

# Ensure directories exist
for folder in [DATA_DIR, MODELS_DIR, LOGS_DIR, SAMPLE_IMAGES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Hardware Runtime Flags
DEFAULT_VISION_BACKEND = os.getenv("VISION_BACKEND", "auto")  # "hailo", "onnx", "auto"
ENABLE_LLAMA_EXECUTORCH = os.getenv("ENABLE_LLAMA_EXECUTORCH", "true").lower() == "true"

# Vision Model Configuration
DEFAULT_VISION_MODEL_PATH = MODELS_DIR / "neu_surface_defect.onnx"
DEFAULT_HAILO_HEF_PATH = MODELS_DIR / "neu_surface_defect.hef"
IMAGE_SIZE = (640, 640)
CONFIDENCE_THRESHOLD = 0.40
NMS_THRESHOLD = 0.45

# Defect Classes (NEU Surface Defect Standard)
DEFECT_CLASSES = [
    "crazing",           # Fine micro-cracks
    "inclusion",         # Foreign material embedded
    "patches",           # Discolored / rough areas
    "pitted_surface",    # Small cavities or holes
    "rolled_in_scale",   # Oxides rolled into metal
    "scratches"          # Linear surface abrasions
]

# Defect Severity Criteria (Area percentage or confidence score mapping)
SEVERITY_THRESHOLDS = {
    "CRITICAL": 0.75,   # Requires immediate component rejection / line pause
    "MEDIUM": 0.50,     # Requires operator rework or sorting
    "LOW": 0.25         # Minor cosmetic flaw, pass with warning
}

# Llama ExecuTorch Advisor Configuration
LLAMA_MODEL_PTE_PATH = MODELS_DIR / "llama3_2_1b_instruct_arm64.pte"
MAX_ADVISORY_TOKENS = 150
LLAMA_TEMPERATURE = 0.2

# Logging Configuration
DATABASE_PATH = LOGS_DIR / "inspection_history.db"
JSON_LOG_PATH = LOGS_DIR / "inspection_log.json"
