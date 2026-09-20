"""
agent/pattern_analyzer.py

Rule-based predictable pattern detection.
- Works on password in memory, returns structured findings
- Never includes raw password in returned object
- Never logs passwords
- Thresholds are documented engineering choices, not calibrated probabilities
- Does NOT use large password dictionary or internet lookup
"""

import re
import os
from typing import Dict, List, Optional

# Severity levels are rule-engine severities, not experimentally calibrated probabilities
SEVERITY_HIGH = "high"
SEVERITY_MEDIUM = "medium"
SEVERITY_LOW = "low"

def _make_finding(code: str, severity: str, message: str) -> Dict[str, str]:
    return {"code": code, "severity": severity, "message": message}

# --- Common password and dictionary data (loaded once, no raw passwords in code) ---
_COMMON_PASSWORDS = None
_DICTIONARY_WORDS = None
_KEYBOARD_PATTERNS = [
    "qwerty", "qwertz", "qwertyuiop", "asdf", "asdfgh", "asdfghjkl",
    "zxcv", "zxcvbn", "qaz", "wsx", "edc", "rfv", "tgb", "yhn", "ujm",
    "1234", "12345", "123456", "1234567", "12345678", "123456789",
    "1qaz", "2wsx", "3edc", "4rfv", "q1w2", "a1s2",
    "qwer", "wert", "erty",
]

# Small dictionary of common words for detection (lowercase)
_COMMON_WORDS = [
    "password", "admin", "welcome", "monkey", "dragon", "master", "hello",
    "qwerty", "letmein", "login", "princess", "solo", "starwars", "trustno",
    "flower", "football", "baseball", "superman", "batman", "shadow", "sunshine",
    "iloveyou", "whatever", "computer", "internet", "freedom", "charlie", "aa123",
]

def _load_common_passwords() -> set:
    global _COMMON_PASSWORDS
    if _COMMON_PASSWORDS is not None:
        return _COMMON_PASSWORDS
    # Try to load top common passwords from dataset (aggregate, not raw leak)
    # Use data/clean_passwords.csv top 500 by frequency if available, else fallback to hardcoded
    common = set()
    try:
        import pandas as pd
        path = os.path.join(os.path.dirname(__file__), "..", "data", "clean_passwords.csv")
        if not os.path.exists(path):
            path = "data/clean_passwords.csv"
        if os.path.exists(path):
            # Load only password column, value counts, top 500
            df = pd.read_csv(path, usecols=["password"], dtype=str)
            # Drop NaN and get most common
            vc = df["password"].value_counts().head(500)
            for pw in vc.index:
                if isinstance(pw, str) and len(pw) >= 4:
                    common.add(pw.lower())
    except Exception:
        pass
    # Fallback hardcoded common passwords (top from public lists, not dataset copy)
    fallback_common = {
        "password", "123456", "123456789", "qwerty", "abc123", "password1", "12345678",
        "111111", "123123", "admin", "letmein", "welcome", "monkey", "dragon", "1234",
        "qwerty123", "password123", "12345", "football", "iloveyou", "admin123",
    }
    common.update(fallback_common)
    _COMMON_PASSWORDS = common
    return _COMMON_PASSWORDS

def _load_dictionary_words() -> set:
    global _DICTIONARY_WORDS
    if _DICTIONARY_WORDS is not None:
        return _DICTIONARY_WORDS
    # Use hardcoded common words plus knowledge base words
    words = set(w.lower() for w in _COMMON_WORDS)
    # Add some additional common words from knowledge base context
    words.update({"security", "agent", "heuristic", "computer", "internet", "freedom"})
    _DICTIONARY_WORDS = words
    return _DICTIONARY_WORDS

def analyze_patterns(password: str, features: Optional[Dict] = None) -> Dict[str, List[Dict[str, str]]]:
    """
    Detect predictable or weak structural patterns.

    Args:
        password: password string (synthetic test strings only in this phase)
        features: optional precomputed features from extract_features to avoid recomputation

    Returns:
        {
            "issues": [ {code, severity, message}, ... ],
            "warnings": [ ... ],
            "positive_signals": [ ... ]
        }
        No password is included in the returned object.

    Checks:
        A. Sequential digits (threshold: 3 consecutive ascending/descending)
        B. Sequential letters (threshold: 3 consecutive alphabetic, case-insensitive)
        C. Repeated characters (threshold: 3+ consecutive same char; 4+ is high)
        D. Repeated blocks (regex (.{2,})\1 for repeated substring len>=2)
        E. Low character diversity (length>=8 and diversity<0.4 or unique<4)
        F. Single character type (all digits/lower/upper/special)
        G. Predictable structure (letters followed only by digits, e.g., password123)
        H. Common password (exact match to local dataset top list)
        I. Dictionary word (common word substring, len>=4)
        J. Keyboard patterns (qwerty, asdf, etc.)
        K. Predictable substitutions (a->@ etc., not strong protection)
        L. Date/year patterns (1900-2099, date-like)
        M. Possible personal pattern (capitalized word + numbers, cautious)

    Do NOT simply classify based on length; diversity and pattern findings are considered separately.
    """
    if not isinstance(password, str):
        raise TypeError("password must be a string")

    # Avoid recomputation if features provided; otherwise compute minimal needed
    if features is None:
        # Use local import to avoid circular dependency
        from .feature_extractor import extract_features
        features = extract_features(password)

    issues: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []
    positive_signals: List[Dict[str, str]] = []

    n = features["length"]
    unique = features["unique_character_count"]
    diversity = features["character_diversity_ratio"]

    # --- A. Sequential digits ---
    # Detect meaningful sequential digit runs of length >=3
    # Examples: 123, 234, 987, 321
    # Documented threshold: 3
    has_seq_digits = False
    for i in range(n - 2):
        if password[i].isdigit() and password[i+1].isdigit() and password[i+2].isdigit():
            a, b, c = ord(password[i]), ord(password[i+1]), ord(password[i+2])
            if (b == a + 1 and c == b + 1) or (b == a - 1 and c == b - 1):
                has_seq_digits = True
                break
    if has_seq_digits:
        issues.append(_make_finding(
            "sequential_digits", SEVERITY_MEDIUM,
            "Contains a sequential digit pattern (e.g., 123 or 321)."
        ))

    # --- B. Sequential letters ---
    # Case-insensitive, length 3, e.g., abc, xyz, cba
    has_seq_letters = False
    low_pw = password.lower()
    for i in range(n - 2):
        if low_pw[i].isalpha() and low_pw[i+1].isalpha() and low_pw[i+2].isalpha():
            a, b, c = ord(low_pw[i]), ord(low_pw[i+1]), ord(low_pw[i+2])
            if (b == a + 1 and c == b + 1) or (b == a - 1 and c == b - 1):
                has_seq_letters = True
                break
    if has_seq_letters:
        issues.append(_make_finding(
            "sequential_letters", SEVERITY_MEDIUM,
            "Contains a sequential letter pattern (e.g., abc or xyz)."
        ))

    # --- C. Repeated characters ---
    # Threshold documented: 3+ consecutive same character triggers detection; 4+ is high severity
    max_run = features["maximum_repeated_run"]
    if max_run >= 4:
        issues.append(_make_finding(
            "repeated_characters", SEVERITY_HIGH,
            "Contains excessive character repetition (4+ same character consecutively)."
        ))
    elif max_run >= 3:
        warnings.append(_make_finding(
            "repeated_characters", SEVERITY_MEDIUM,
            "Contains repeated characters (3 same character consecutively)."
        ))

    # --- D. Repeated blocks ---
    # Detect repeated short substrings/blocks, e.g., abab, 1212, abcabc
    # Regex: (.{2,})\1  means any substring len>=2 repeated immediately
    # This is a heuristic, not exhaustive block detection
    has_repeated_block = False
    if n >= 4 and re.search(r"(.{2,})\1", password):
        has_repeated_block = True
    if has_repeated_block:
        issues.append(_make_finding(
            "repeated_block", SEVERITY_MEDIUM,
            "Contains a repeated block pattern (e.g., abab or abcabc)."
        ))

    # --- E. Low character diversity ---
    # length high but unique low: length>=8 and (diversity <0.4 or unique <=4)
    # Do NOT simply classify based on length alone; this checks diversity relative to length
    if n >= 8:
        if diversity < 0.4 or unique <= 4:
            issues.append(_make_finding(
                "low_diversity", SEVERITY_MEDIUM,
                "Character diversity is low for its length (many repeated characters)."
            ))
        elif diversity < 0.5:
            warnings.append(_make_finding(
                "low_diversity", SEVERITY_LOW,
                "Character diversity is somewhat low."
            ))

    # --- F. Single character type ---
    # Detect all digits, all lowercase, all uppercase, all special
    # These are findings/signals, not automatically insecure, but indicate limited composition
    type_count = features["unique_character_type_count"]
    if n > 0 and type_count == 1:
        if features["has_digits"] and n >= 1:
            warnings.append(_make_finding(
                "single_type_digits", SEVERITY_MEDIUM,
                "Uses only digits (single character type)."
            ))
        elif features["has_lowercase"] and not features["has_uppercase"]:
            warnings.append(_make_finding(
                "single_type_lowercase", SEVERITY_MEDIUM,
                "Uses only lowercase letters (single character type)."
            ))
        elif features["has_uppercase"] and not features["has_lowercase"]:
            warnings.append(_make_finding(
                "single_type_uppercase", SEVERITY_MEDIUM,
                "Uses only uppercase letters (single character type)."
            ))
        elif features["has_special"]:
            warnings.append(_make_finding(
                "single_type_special", SEVERITY_MEDIUM,
                "Uses only special characters (single character type)."
            ))
    elif type_count >= 3:
        positive_signals.append(_make_finding(
            "multiple_types", SEVERITY_LOW,
            "Uses multiple character types."
        ))
    elif type_count == 2:
        warnings.append(_make_finding(
            "two_types", SEVERITY_LOW,
            "Uses two character types."
        ))

    # --- G. Predictable structure ---
    # Simple structure: letters followed only by digits, e.g., password123
    # This is a common predictable pattern, not a huge dictionary check
    if n >= 4 and re.fullmatch(r"[A-Za-z]+\d+", password):
        warnings.append(_make_finding(
            "predictable_structure", SEVERITY_LOW,
            "Has predictable structure: letters followed by digits."
        ))

    # --- H. Common password detection (using dataset top list, not claiming breach) ---
    # Detect if password exactly matches a common password from dataset/knowledge
    # This is local common-password data, not a breach check
    try:
        common_set = _load_common_passwords()
        if n >= 4 and password.lower() in common_set:
            issues.append(_make_finding(
                "common_password", SEVERITY_HIGH,
                "Matches a commonly used password pattern from local dataset."
            ))
        elif n >= 4:
            # Check if password contains a common password as substring (e.g., password123 contains password)
            low_pw = password.lower()
            for common_pw in common_set:
                if len(common_pw) >= 4 and common_pw in low_pw and len(common_pw) >= n * 0.6:
                    # Only flag if common password is substantial part
                    warnings.append(_make_finding(
                        "common_pattern", SEVERITY_MEDIUM,
                        "Contains a commonly used password pattern."
                    ))
                    break
    except Exception:
        pass

    # --- I. Dictionary / word detection ---
    # Detect dictionary words where feasible; threshold length >=4 to avoid false positives
    try:
        dict_words = _load_dictionary_words()
        low_pw = password.lower()
        for word in dict_words:
            if len(word) >= 4 and word in low_pw:
                # Require word is not just part of longer random string but at least 4 chars and not too short relative to password
                if len(word) >= 4 and (len(word) >= 5 or n <= 12):
                    warnings.append(_make_finding(
                        "dictionary_word", SEVERITY_LOW,
                        f"Contains a dictionary word pattern ({word})."
                    ))
                    break
    except Exception:
        pass

    # --- J. Keyboard patterns ---
    # Detect common keyboard walks (qwerty, asdf, zxcv, etc.)
    try:
        low_pw = password.lower()
        for pattern in _KEYBOARD_PATTERNS:
            if pattern in low_pw:
                issues.append(_make_finding(
                    "keyboard_pattern", SEVERITY_MEDIUM,
                    "Contains a common keyboard pattern (e.g., qwerty/asdf)."
                ))
                break
        # Also check for keyboard adjacent walks like 1qaz, 2wsx already in list
    except Exception:
        pass

    # --- K. Predictable substitutions (leet) ---
    # Detect a->@, s->$, etc., but do not treat as strong protection
    # Only flag if password contains both letters and leet symbols and also looks like a word
    try:
        has_leet = any(c in password for c in "@$0135!7")
        # Map: a->@, s->$, etc.
        # If password contains leet chars and also contains letters, and length reasonable, flag
        if has_leet and n >= 4 and (features["has_lowercase"] or features["has_uppercase"]):
            # Check if replacing leet with original yields a common word
            leet_map = str.maketrans("@$0135!7", "asoeati")
            de_leet = password.translate(leet_map).lower()
            dict_words = _load_dictionary_words()
            for word in dict_words:
                if len(word) >= 4 and word in de_leet:
                    warnings.append(_make_finding(
                        "predictable_substitution", SEVERITY_LOW,
                        "Uses predictable character substitutions (e.g., @ for a) that do not significantly increase security."
                    ))
                    break
    except Exception:
        pass

    # --- L. Date/Year patterns ---
    # Detect obvious years or date-like patterns (1900-2099, 4 consecutive digits that look like year)
    try:
        # Simple year: 19xx or 20xx
        if re.search(r"(19\d{2}|20\d{2})", password):
            warnings.append(_make_finding(
                "year_pattern", SEVERITY_LOW,
                "Contains a year-like pattern (e.g., 19xx or 20xx)."
            ))
        # Date-like: dd/mm/yyyy or mm-dd-yyyy or 8 consecutive digits
        elif re.search(r"\d{2}[/-]\d{2}[/-]\d{2,4}", password):
            warnings.append(_make_finding(
                "date_pattern", SEVERITY_LOW,
                "Contains a date-like pattern."
            ))
    except Exception:
        pass

    # --- M. Personal-information patterns (cautious) ---
    # Where possible, identify obvious predictable structures such as names/usernames
    # Do NOT claim personal info found unless context provided; just note pattern that *could* be personal
    # For now, only note if password contains a capitalized word + year pattern, which often is name+year
    try:
        if re.search(r"[A-Z][a-z]{2,}\d{2,4}", password):
            # e.g., John1985, Alice123
            warnings.append(_make_finding(
                "possible_personal_pattern", SEVERITY_LOW,
                "Contains a structure that could be a name or personal pattern (capitalized word + numbers)."
            ))
    except Exception:
        pass

    # Positive signals not tied to single checks
    if n >= 12 and type_count >= 3 and diversity >= 0.6 and max_run < 3 and not has_seq_digits and not has_seq_letters and not has_repeated_block:
        positive_signals.append(_make_finding(
            "substantial_length", SEVERITY_LOW,
            "Has substantial length with good diversity and no obvious weak patterns."
        ))
    elif n >= 8 and type_count >= 2:
        # General positive for adequate length and mix
        # Avoid duplicating multiple_types above; add length positive
        if n >= 8:
            positive_signals.append(_make_finding(
                "adequate_length", SEVERITY_LOW,
                "Has adequate length."
            ))

    return {
        "issues": issues,
        "warnings": warnings,
        "positive_signals": positive_signals,
    }
