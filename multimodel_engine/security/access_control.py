# security/access_control.py

def check_access(role):

    if role not in ["patient", "doctor", "admin"]:
        raise PermissionError("Unauthorized role")
