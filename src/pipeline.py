"""
Core System Pipeline Coordinator.
Connects Vision Engine (Hailo/CPU) and Advisory Engine (Llama ExecuTorch) with SQLite Logging.
"""

import cv2
import numpy as np
import time
from typing import Dict, Any, Tuple

from .vision.vision_engine import VisionEngine, DetectionResult
from .advisory.executorch_advisor import ExecuTorchLlamaAdvisor, OperatorAdvisory
from .logging.inspection_logger import InspectionLogger

class EdgeQCPipeline:
    def __init__(self, vision_backend: str = "auto", model_onnx: str = None, llama_pte: str = None):
        self.vision_engine = VisionEngine(model_path=model_onnx, backend=vision_backend)
        self.advisor_engine = ExecuTorchLlamaAdvisor(model_pte_path=llama_pte)
        self.logger = InspectionLogger()
        self.inspection_counter = 1000

    def process_component_inspection(self, frame: np.ndarray, component_id: str = None) -> Tuple[DetectionResult, OperatorAdvisory, Dict[str, Any]]:
        """
        Executes end-to-end dual pipeline inspection for a single manufacturing frame.
        """
        if component_id is None:
            self.inspection_counter += 1
            component_id = f"CMP-{self.inspection_counter:05d}"

        # 1. Pipeline 1: Real-time Vision Inspection
        detection = self.vision_engine.inspect_frame(frame)

        # 2. Pipeline 2: Llama ExecuTorch Advisory Generation
        advisory = self.advisor_engine.generate_advisory(detection)

        # 3. Pipeline 3: Logging & Database Telemetry
        log_record = self.logger.log_inspection(component_id, detection, advisory)

        return detection, advisory, log_record
