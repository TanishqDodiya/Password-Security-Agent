"""
agent/generator.py - Secure password/passphrase generator

- Uses cryptographically secure randomness (secrets)
- Configurable length and character sets
- Avoids predictable patterns and common passwords
- Does not store generated passwords
- Analyze generated password using same security engine (caller can call analyze_password)
"""

import secrets
import string
from typing import Dict

# Character sets
LOWER = string.ascii_lowercase
UPPER = string.ascii_uppercase
DIGITS = string.digits
SYMBOLS = "!@#$%^&*()-_=+[]{}|;:,.<>?/"

# Common passwords for avoidance (subset, lowercased)
_COMMON_FOR_AVOID = {
    "password", "123456", "123456789", "qwerty", "abc123", "password1", "12345678",
    "111111", "123123", "admin", "letmein", "welcome", "monkey", "dragon",
}

def _is_predictable(pw: str) -> bool:
    """Check if generated password would be flagged as predictable (to avoid)."""
    # Avoid simple patterns that would be immediately weak
    # Use pattern_analyzer without importing at top to avoid circular
    try:
        from .pattern_analyzer import analyze_patterns
        from .feature_extractor import extract_features
        feats = extract_features(pw)
        analysis = analyze_patterns(pw, feats)
        # If any high severity issues, consider predictable
        for it in analysis.get("issues", []):
            if it.get("severity") == "high":
                return True
            if it.get("code") in ("sequential_digits", "sequential_letters", "keyboard_pattern", "repeated_block", "common_password"):
                return True
        # Also avoid low diversity
        if feats["character_diversity_ratio"] < 0.5 and len(pw) >= 8:
            return True
        # Avoid common password exact match
        if pw.lower() in _COMMON_FOR_AVOID:
            return True
    except Exception:
        return False
    return False

def generate_password(
    length: int = 16,
    use_lower: bool = True,
    use_upper: bool = True,
    use_digits: bool = True,
    use_symbols: bool = True,
) -> str:
    """
    Generate a secure password using cryptographically secure randomness.

    Args:
        length: desired length (4-128, default 16)
        use_lower: include lowercase
        use_upper: include uppercase
        use_digits: include digits
        use_symbols: include symbols

    Returns:
        Generated password string

    Raises:
        ValueError: if length invalid or no character types selected
    """
    if not isinstance(length, int):
        raise TypeError("length must be an integer")
    if length < 4 or length > 128:
        raise ValueError("length must be between 4 and 128")
    if not any([use_lower, use_upper, use_digits, use_symbols]):
        raise ValueError("at least one character type must be selected")

    charset = ""
    required = []
    if use_lower:
        charset += LOWER
        required.append(secrets.choice(LOWER))
    if use_upper:
        charset += UPPER
        required.append(secrets.choice(UPPER))
    if use_digits:
        charset += DIGITS
        required.append(secrets.choice(DIGITS))
    if use_symbols:
        charset += SYMBOLS
        required.append(secrets.choice(SYMBOLS))

    # Ensure at least one of each required type is included, fill rest randomly
    # Use secrets for cryptographic security
    remaining = length - len(required)
    # Generate with attempt loop to avoid predictable patterns and common passwords
    for _ in range(10):  # max 10 attempts to avoid weak pattern
        chars = required.copy()
        for _ in range(remaining):
            chars.append(secrets.choice(charset))
        # Shuffle securely
        # Use secrets to shuffle: Fisher-Yates with secrets
        for i in range(len(chars) - 1, 0, -1):
            j = secrets.randbelow(i + 1)
            chars[i], chars[j] = chars[j], chars[i]
        pw = "".join(chars)
        # Avoid predictable patterns and common passwords
        if not _is_predictable(pw):
            return pw
        # If predictable, retry (up to 10 times)
    # If still predictable after 10, return last (still better than insecure random)
    return pw

def generate_passphrase(num_words: int = 4, separator: str = "-") -> str:
    """
    Generate a passphrase from random words (alternative to password).
    Uses secrets for word selection. Words are from a small built-in list for demo.
    """
    # Small word list for demo (not exhaustive, but avoids common password patterns)
    words = [
        "correct", "horse", "battery", "staple", "apple", "river", "mountain",
        "forest", "ocean", "candle", "bridge", "garden", "window", "castle",
        "summer", "winter", "autumn", "spring", "silver", "golden", "crystal",
    ]
    if num_words < 2 or num_words > 10:
        raise ValueError("num_words must be between 2 and 10")
    chosen = [secrets.choice(words) for _ in range(num_words)]
    # Add random digit and symbol to increase strength if desired? Keep simple
    return separator.join(chosen)

def get_generator_options() -> Dict:
    """Return available generator options for frontend."""
    return {
        "length_range": [4, 128],
        "default_length": 16,
        "character_types": ["lowercase", "uppercase", "digits", "symbols"],
    }
