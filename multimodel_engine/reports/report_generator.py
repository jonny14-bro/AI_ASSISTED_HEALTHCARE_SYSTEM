# report_generator.py

import json
import os


def save_report(
    report_id,
    patient_id,
    modality,
    findings,
    confidence,
    output_image=None
):
    """
    Save a static medical report as JSON.

    Parameters:
    ----------
    report_id : str
        Unique ID for this scan/report
    patient_id : str
        Patient identifier from backend
    modality : str
        'xray' or 'ultrasound'
    findings : str
        Clinical findings text
    confidence : float
        Model confidence percentage
    output_image : str or None
        Path to annotated image (optional)
    """

    # -------- Create patient report folder --------
    report_dir = os.path.join("patients", patient_id, "reports")
    os.makedirs(report_dir, exist_ok=True)

    # -------- Build report object --------
    report = {
        "report_id": report_id,
        "patient_id": patient_id,
        "modality": modality,
        "findings": findings,
        "confidence": round(float(confidence), 2),
        "output_image": output_image
    }

    # -------- Save JSON file --------
    report_path = os.path.join(
        report_dir,
        f"report_{report_id}.json"
    )

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    return report_path
