"""
agent/ml_analyzer.py

ML inference as optional evidence source.
- Loads models/password_classifier.joblib (RandomForest C_security_pattern, 19 features)
- Extracts ONLY required features, runs inference in memory, discards password
- Never logs, prints, or saves raw password
- Returns structured ML evidence with explicit limitation warning
- Heavily length-driven dataset warning is always included
"""

from typing import Dict, Optional
import os
import re

# Global model cache (stores model only, never passwords)
_MODEL_CACHE = None
_ARTIFACT_CACHE = None

WARNING_TEXT = (
    "This model predicts labels learned from the supplied dataset. "
    "Dataset analysis showed that the labels are heavily length-driven "
    "(length-only model ≈99.98% test accuracy, full model ≈99.98%), "
    "so this prediction is not treated as an independent measure of real-world password security."
)

def _load_artifact():
    global _MODEL_CACHE, _ARTIFACT_CACHE
    if _ARTIFACT_CACHE is not None:
        return _ARTIFACT_CACHE
    artifact_path = os.path.join("models", "password_classifier.joblib")
    if not os.path.exists(artifact_path):
        # Also try absolute relative to project root
        artifact_path = os.path.join(os.path.dirname(__file__), "..", "models", "password_classifier.joblib")
        artifact_path = os.path.normpath(artifact_path)
        if not os.path.exists(artifact_path):
            return None
    try:
        import joblib
        artifact = joblib.load(artifact_path)
        _ARTIFACT_CACHE = artifact
        _MODEL_CACHE = artifact.get("model")
        return artifact
    except Exception:
        return None

def _extract_for_model(password: str) -> Dict[str, float]:
    """
    Extract exactly the 19 features expected by the trained model.
    Matches training logic in create_artifact.py / ml_evaluation.py.
    No password stored.
    """
    n = len(password)
    lower = sum(1 for c in password if c.islower())
    upper = sum(1 for c in password if c.isupper())
    digit = sum(1 for c in password if c.isdigit())
    # special = non-alnum (includes whitespace/punctuation/Unicode)
    special = n - sum(1 for c in password if c.isalnum())
    alpha = sum(1 for c in password if c.isalpha())
    unique = len(set(password)) if n else 0

    digit_ratio = digit / n if n else 0.0
    upper_ratio = upper / n if n else 0.0
    lower_ratio = lower / n if n else 0.0
    special_ratio = special / n if n else 0.0

    repeated = n - unique if n else 0

    # maximum repeated run
    if n == 0:
        max_run = 0
    elif n == 1:
        max_run = 1
    else:
        cur = 1
        max_run = 1
        for i in range(1, n):
            if password[i] == password[i-1]:
                cur += 1
                if cur > max_run:
                    max_run = cur
            else:
                cur = 1

    # number_of_character_type_changes
    def _char_type(c: str) -> str:
        if c.islower():
            return "L"
        if c.isupper():
            return "U"
        if c.isdigit():
            return "D"
        return "S"
    changes = 0
    if n > 1:
        prev = _char_type(password[0])
        for ch in password[1:]:
            cur_t = _char_type(ch)
            if cur_t != prev:
                changes += 1
            prev = cur_t

    # has_sequential_digits
    has_seq_digits = 0
    for i in range(n - 2):
        if password[i].isdigit() and password[i+1].isdigit() and password[i+2].isdigit():
            a, b, c_ = ord(password[i]), ord(password[i+1]), ord(password[i+2])
            if (b == a + 1 and c_ == b + 1) or (b == a - 1 and c_ == b - 1):
                has_seq_digits = 1
                break

    # has_sequential_letters (case-insensitive)
    has_seq_letters = 0
    low = password.lower()
    for i in range(n - 2):
        if low[i].isalpha() and low[i+1].isalpha() and low[i+2].isalpha():
            a, b, c_ = ord(low[i]), ord(low[i+1]), ord(low[i+2])
            if (b == a + 1 and c_ == b + 1) or (b == a - 1 and c_ == b - 1):
                has_seq_letters = 1
                break

    # has_repeated_block
    has_block = 1 if (n >= 4 and re.search(r"(.{2,})\1", password)) else 0

    diversity = unique / n if n else 0.0
    num_types = sum([lower > 0, upper > 0, digit > 0, special > 0])

    return {
        "password_length": float(n),
        "lowercase_count": float(lower),
        "uppercase_count": float(upper),
        "digit_count": float(digit),
        "special_count": float(special),
        "alphabetic_count": float(alpha),
        "unique_character_count": float(unique),
        "digit_ratio": float(digit_ratio),
        "uppercase_ratio": float(upper_ratio),
        "lowercase_ratio": float(lower_ratio),
        "special_ratio": float(special_ratio),
        "repeated_character_count": float(repeated),
        "maximum_repeated_run": float(max_run),
        "number_of_character_type_changes": float(changes),
        "has_sequential_digits": float(has_seq_digits),
        "has_sequential_letters": float(has_seq_letters),
        "has_repeated_block": float(has_block),
        "character_diversity_ratio": float(diversity),
        "number_of_unique_character_types": float(num_types),
    }

def analyze_with_ml(password: str) -> Dict:
    """
    Run ML model inference on password (in-memory).

    Args:
        password: synthetic password string (never logged or saved)

    Returns:
        {
            "available": bool,
            "predicted_class": 0/1/2 (if available),
            "confidence": float (max class probability, if predict_proba available),
            "class_probabilities": {"0": p0, "1": p1, "2": p2} (if available),
            "model_name": str,
            "feature_set": str,
            "warning": str (limitation explicit),
            "reason": str (if not available)
        }
        Never includes raw password.
    """
    if not isinstance(password, str):
        return {
            "available": False,
            "reason": "password must be a string",
            "warning": WARNING_TEXT,
        }

    artifact = _load_artifact()
    if artifact is None:
        return {
            "available": False,
            "reason": "Model artifact not found at models/password_classifier.joblib",
            "warning": WARNING_TEXT,
        }

    model = artifact.get("model")
    feature_names = artifact.get("feature_names")
    if model is None or feature_names is None:
        return {
            "available": False,
            "reason": "Model artifact malformed",
            "warning": WARNING_TEXT,
        }

    # Extract ONLY required features (deterministic, explainable)
    feats = _extract_for_model(password)
    # Build vector in expected order
    try:
        import numpy as np
        vec = np.array([[feats[f] for f in feature_names]], dtype=float)
    except KeyError as e:
        return {
            "available": False,
            "reason": f"Missing feature {e}",
            "warning": WARNING_TEXT,
        }

    # Inference
    try:
        pred = model.predict(vec)[0]
        predicted_class = int(pred)
    except Exception as e:
        return {
            "available": False,
            "reason": f"Inference failed: {type(e).__name__}",
            "warning": WARNING_TEXT,
        }

    result: Dict = {
        "available": True,
        "predicted_class": predicted_class,
        "model_name": artifact.get("model_name", "RandomForest"),
        "feature_set": artifact.get("feature_set", "C_security_pattern"),
        "warning": artifact.get("warning", WARNING_TEXT),
    }

    # Confidence handling: use predict_proba if available, call it model class probability
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(vec)[0]
            # proba order corresponds to model.classes_
            classes = getattr(model, "classes_", [0, 1, 2])
            class_probabilities = {str(int(c)): float(p) for c, p in zip(classes, proba)}
            # Ensure 0,1,2 keys exist even if model missing a class (should not happen)
            for k in ["0", "1", "2"]:
                if k not in class_probabilities:
                    class_probabilities[k] = 0.0
            # Confidence is max class probability
            confidence = float(max(proba))
            result["confidence"] = confidence
            result["class_probabilities"] = class_probabilities
            result["model_class_probability_note"] = "model class probability, not probability that password is secure"
        except Exception:
            # If proba fails, still return predicted class
            pass

    return result
