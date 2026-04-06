def ratio_to_severity_class(ratio):
    """
    Opsional: mapping berbasis rasio bercak.
    Ini hanya fallback kalau kamu butuh rule-based severity.
    """
    if ratio < 0.10:
        return "low"
    elif ratio < 0.25:
        return "medium"
    return "high"


def class_to_numeric(label):
    mapping = {"low": 0, "medium": 1, "high": 2}
    return mapping[label]


def numeric_to_class(value):
    mapping = {0: "low", 1: "medium", 2: "high"}
    return mapping[value]