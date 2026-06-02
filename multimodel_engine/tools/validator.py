# tools/validator.py

import os


def validate_input(patient_id, message, image_path):

    if not patient_id:
        raise ValueError("Patient ID is required")

    if not message:
        raise ValueError("Message cannot be empty")

    if image_path and not os.path.exists(image_path):
        raise FileNotFoundError("Image file not found")
