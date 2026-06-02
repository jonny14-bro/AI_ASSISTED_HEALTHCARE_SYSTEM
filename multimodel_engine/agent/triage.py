def assess_risk(message, findings):

    msg = message.lower()

    if any(w in msg for w in ["severe pain", "can't move", "bleeding"]):
        return "HIGH"

    if "fracture" in findings.lower():
        return "MEDIUM"

    return "LOW"
