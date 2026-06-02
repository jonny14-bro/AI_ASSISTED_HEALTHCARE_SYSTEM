# tools/xray_tool.py

import json
from fracture_detection_engine.inference.pipeline import predict
import os
from datetime import datetime


def analyze_xray(image_path, patient_id):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    scan_dir = os.path.join("patients", patient_id, "xray", timestamp)
    os.makedirs(scan_dir, exist_ok=True)

    predict(image_path, scan_dir, patient_id)

    result_json = os.path.join(scan_dir, "result.json")

    with open(result_json) as f:
        result = json.load(f)

    confidence = result["probability"] * 100
    prediction = result["prediction"]

    if prediction == "FRACTURE":
        findings = "Fracture detected with cortical disruption."
    else:
        findings = "No fracture detected."

    return findings, confidence, scan_dir, timestamp
