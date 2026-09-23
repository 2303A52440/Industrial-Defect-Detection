"""
Edge Inspection Logging System.
Stores inspection results, vision telemetry, and Llama operator advisory logs in SQLite and JSON format.
"""

import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from ..config import DATABASE_PATH, JSON_LOG_PATH
from ..vision.vision_engine import DetectionResult
from ..advisory.executorch_advisor import OperatorAdvisory

class InspectionLogger:
    def __init__(self, db_path: Optional[Path] = None, json_path: Optional[Path] = None):
        self.db_path = str(db_path or DATABASE_PATH)
        self.json_path = str(json_path or JSON_LOG_PATH)
        self._init_db()

    def _init_db(self):
        """Initializes SQLite inspection history schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inspection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                component_id TEXT NOT NULL,
                qc_status TEXT NOT NULL,
                defect_count INTEGER NOT NULL,
                primary_defect TEXT,
                max_severity TEXT,
                vision_backend TEXT NOT NULL,
                vision_latency_ms REAL NOT NULL,
                llama_backend TEXT NOT NULL,
                llama_latency_ms REAL NOT NULL,
                recommendation TEXT NOT NULL,
                action_type TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def log_inspection(
        self,
        component_id: str,
        detection: DetectionResult,
        advisory: OperatorAdvisory
    ) -> Dict[str, Any]:
        """Logs a single inspection record to both SQLite and JSON file."""
        now_str = datetime.now().isoformat()
        primary_defect = detection.defects[0].class_name if detection.defects else "NONE"
        
        severities = [d.severity for d in detection.defects]
        max_severity = "CRITICAL" if "CRITICAL" in severities else "MEDIUM" if "MEDIUM" in severities else "LOW" if "LOW" in severities else "NONE"

        record = {
            "timestamp": now_str,
            "component_id": component_id,
            "qc_status": "PASSED" if detection.pass_quality_check else "REJECTED",
            "defect_count": len(detection.defects),
            "primary_defect": primary_defect,
            "max_severity": max_severity,
            "vision_backend": detection.backend_used,
            "vision_latency_ms": round(detection.inference_time_ms, 2),
            "llama_backend": advisory.runtime_engine,
            "llama_latency_ms": round(advisory.llama_inference_time_ms, 2),
            "recommendation": advisory.primary_recommendation,
            "action_type": advisory.action_type
        }

        # Write to SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO inspection_history (
                timestamp, component_id, qc_status, defect_count, primary_defect,
                max_severity, vision_backend, vision_latency_ms, llama_backend,
                llama_latency_ms, recommendation, action_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["timestamp"], record["component_id"], record["qc_status"], record["defect_count"],
            record["primary_defect"], record["max_severity"], record["vision_backend"],
            record["vision_latency_ms"], record["llama_backend"], record["llama_latency_ms"],
            record["recommendation"], record["action_type"]
        ))
        conn.commit()
        conn.close()

        # Append to JSON Log
        self._append_json(record)

        return record

    def _append_json(self, record: Dict[str, Any]):
        """Appends record to JSON file array."""
        records = []
        if Path(self.json_path).exists():
            try:
                with open(self.json_path, 'r') as f:
                    records = json.load(f)
            except json.JSONDecodeError:
                records = []
        
        records.append(record)
        with open(self.json_path, 'w') as f:
            json.dump(records, f, indent=2)

    def fetch_history_df(self, limit: int = 100) -> pd.DataFrame:
        """Fetches historic inspection records as a Pandas DataFrame for dashboard display."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            f"SELECT * FROM inspection_history ORDER BY id DESC LIMIT {limit}",
            conn
        )
        conn.close()
        return df

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates aggregated QC metrics for factory manager view."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*), SUM(CASE WHEN qc_status='PASSED' THEN 1 ELSE 0 END) FROM inspection_history")
        total, passed = cursor.fetchone()
        
        if not total or total == 0:
            conn.close()
            return {"total_inspections": 0, "pass_rate": 100.0, "avg_vision_ms": 0.0, "avg_llama_ms": 0.0}

        cursor.execute("SELECT AVG(vision_latency_ms), AVG(llama_latency_ms) FROM inspection_history")
        avg_v, avg_l = cursor.fetchone()
        conn.close()

        return {
            "total_inspections": total,
            "pass_rate": round((passed / total) * 100.0, 1),
            "avg_vision_ms": round(avg_v or 0.0, 1),
            "avg_llama_ms": round(avg_l or 0.0, 1)
        }
