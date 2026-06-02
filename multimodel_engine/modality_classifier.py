# modality_classifier.py


def classify_modality_from_text(message):
    """
    Determine scan modality from patient message.

    Returns:
        "xray"
        "ultrasound"
        None  -> if unclear
    """

    if not message:
        return None

    msg = message.lower()

    # -------- X-ray keywords --------
    xray_terms = [
        "xray",
        "x-ray",
        "radiograph",
        "xr",
        "chest xray",
        "bone xray"
    ]

    # -------- Ultrasound keywords --------
    ultrasound_terms = [
        "ultrasound",
        "usg",
        "sonography",
        "doppler",
        "scan report"
    ]

    if any(term in msg for term in xray_terms):
        return "xray"

    if any(term in msg for term in ultrasound_terms):
        return "ultrasound"

    return None
