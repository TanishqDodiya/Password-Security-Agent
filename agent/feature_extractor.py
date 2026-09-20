"""
agent/feature_extractor.py

Deterministic security feature extraction.
- Accepts password in memory, returns derived features only
- Never logs, saves, or stores the raw password
- Handles empty, one-char, very long, Unicode, whitespace, punctuation
- Empty password: ratios are 0.0 to avoid division-by-zero
- No global mutable state
- Entropy is calculated as explainable estimates, not proof of security (see docstring)
"""

from typing import Dict
import math

def extract_features(password: str) -> Dict[str, object]:
    """
    Extract deterministic security features from a single password.

    Args:
        password: password string (may be empty, Unicode, whitespace)

    Returns:
        Dictionary with BASIC, RATIOS, STRUCTURAL, BOOLEAN features.
        No password string is included in the returned dict.

    Features:
        BASIC:
            length, lowercase_count, uppercase_count, digit_count,
            special_count, alphabetic_count, unique_character_count
        RATIOS:
            lowercase_ratio, uppercase_ratio, digit_ratio, special_ratio,
            character_diversity_ratio (unique/length)
        STRUCTURAL:
            character_type_changes, repeated_character_count,
            maximum_repeated_run, unique_character_type_count
        BOOLEAN:
            has_lowercase, has_uppercase, has_digits, has_special,
            has_mixed_case, has_multiple_character_types

    Limitations:
        - This is a heuristic feature set, not a cryptographic strength proof
        - Special is defined as any character not alphanumeric (includes whitespace/punctuation/Unicode symbols)
        - Entropy estimates are explainable approximations; high entropy alone does not mean secure if patterns/common words exist
    """
    if not isinstance(password, str):
        raise TypeError("password must be a string")

    n = len(password)

    # BASIC COUNTS - handle empty and Unicode correctly
    lowercase_count = sum(1 for c in password if c.islower())
    uppercase_count = sum(1 for c in password if c.isupper())
    digit_count = sum(1 for c in password if c.isdigit())
    # alphabetic includes Unicode letters (more accurate than lower+upper for some scripts)
    alphabetic_count = sum(1 for c in password if c.isalpha())
    # special: non-alphanumeric (includes whitespace, punctuation, symbols)
    # We compute as total - alnum to include all non-alnum explicitly
    alphanumeric_count = sum(1 for c in password if c.isalnum())
    special_count = n - alphanumeric_count
    unique_character_count = len(set(password)) if n > 0 else 0

    # RATIOS - avoid division-by-zero for empty password
    if n == 0:
        lowercase_ratio = 0.0
        uppercase_ratio = 0.0
        digit_ratio = 0.0
        special_ratio = 0.0
        character_diversity_ratio = 0.0
    else:
        lowercase_ratio = lowercase_count / n
        uppercase_ratio = uppercase_count / n
        digit_ratio = digit_count / n
        special_ratio = special_count / n
        character_diversity_ratio = unique_character_count / n

    # STRUCTURAL
    # character_type_changes: transitions between types L/U/D/S
    def _char_type(c: str) -> str:
        if c.islower():
            return "L"
        if c.isupper():
            return "U"
        if c.isdigit():
            return "D"
        return "S"  # includes whitespace, punctuation, Unicode symbols not covered above

    character_type_changes = 0
    if n > 1:
        prev = _char_type(password[0])
        for ch in password[1:]:
            cur = _char_type(ch)
            if cur != prev:
                character_type_changes += 1
            prev = cur

    repeated_character_count = n - unique_character_count if n > 0 else 0

    # maximum consecutive same character run
    maximum_repeated_run = 0
    if n == 0:
        maximum_repeated_run = 0
    elif n == 1:
        maximum_repeated_run = 1
    else:
        cur_run = 1
        max_run = 1
        for i in range(1, n):
            if password[i] == password[i - 1]:
                cur_run += 1
                if cur_run > max_run:
                    max_run = cur_run
            else:
                cur_run = 1
        maximum_repeated_run = max_run

    # unique character type count: how many of [lower, upper, digit, special] appear
    has_lowercase = lowercase_count > 0
    has_uppercase = uppercase_count > 0
    has_digits = digit_count > 0
    has_special = special_count > 0
    unique_character_type_count = sum([
        has_lowercase, has_uppercase, has_digits, has_special
    ])

    # BOOLEAN
    has_mixed_case = has_lowercase and has_uppercase
    has_multiple_character_types = unique_character_type_count >= 2

    # ENTROPY & GUESSABILITY (explainable estimates)
    # charset_size: estimated size of character set actually used
    charset_size = 0
    if has_lowercase:
        charset_size += 26
    if has_uppercase:
        charset_size += 26
    if has_digits:
        charset_size += 10
    if has_special:
        # Use 32 as common printable symbols estimate; for Unicode, treat as at least 32
        charset_size += 32
    # For pure Unicode beyond ascii, charset_size may be larger, but we keep 32 as baseline
    if charset_size == 0:
        charset_size = 0

    # estimated_entropy: theoretical entropy if password were random from charset
    # = length * log2(charset_size); capped to avoid overflow, explainable
    if n == 0 or charset_size == 0:
        estimated_entropy = 0.0
    else:
        try:
            estimated_entropy = n * math.log2(charset_size)
        except (ValueError, OverflowError):
            estimated_entropy = 0.0

    # shannon_entropy: actual distribution entropy of characters in this password
    # -sum(p * log2(p)) where p is frequency of each unique char
    if n == 0:
        shannon_entropy = 0.0
    else:
        shannon_entropy = 0.0
        for ch in set(password):
            count = password.count(ch)
            p = count / n
            shannon_entropy -= p * math.log2(p) if p > 0 else 0
        # Total Shannon for password = entropy per char * length, but we keep per-password total as shannon * n? Keep per-char for explainability
        # Provide both: per_char and total
        shannon_entropy_total = shannon_entropy * n
        # Use total for reporting
        shannon_entropy = shannon_entropy_total

    # brute_force_combinations: charset_size ** length, but cap to avoid huge ints
    # Provide log2 combinations and human-readable order of magnitude
    if n == 0 or charset_size == 0:
        brute_force_combinations_log2 = 0.0
    else:
        brute_force_combinations_log2 = estimated_entropy  # same as above

    # For display, also calculate approximate combinations as string if not huge
    # Cap at 1e18 for display to avoid false precision
    brute_force_search_space = None
    if charset_size > 0 and n > 0:
        # Use log2 to estimate, then format as 2^log2
        # Store as string like "2^45 (~3.5e13)"
        # Avoid huge pow
        if brute_force_combinations_log2 < 60:
            try:
                combinations = pow(charset_size, n)
                brute_force_search_space = str(combinations)
            except OverflowError:
                brute_force_search_space = f"2^{brute_force_combinations_log2:.1f}"
        else:
            brute_force_search_space = f"2^{brute_force_combinations_log2:.1f}"

    return {
        # BASIC
        "length": n,
        "lowercase_count": lowercase_count,
        "uppercase_count": uppercase_count,
        "digit_count": digit_count,
        "special_count": special_count,
        "alphabetic_count": alphabetic_count,
        "unique_character_count": unique_character_count,
        # RATIOS
        "lowercase_ratio": lowercase_ratio,
        "uppercase_ratio": uppercase_ratio,
        "digit_ratio": digit_ratio,
        "special_ratio": special_ratio,
        "character_diversity_ratio": character_diversity_ratio,
        # STRUCTURAL
        "character_type_changes": character_type_changes,
        "repeated_character_count": repeated_character_count,
        "maximum_repeated_run": maximum_repeated_run,
        "unique_character_type_count": unique_character_type_count,
        # BOOLEAN
        "has_lowercase": has_lowercase,
        "has_uppercase": has_uppercase,
        "has_digits": has_digits,
        "has_special": has_special,
        "has_mixed_case": has_mixed_case,
        "has_multiple_character_types": has_multiple_character_types,
        # ENTROPY & BRUTE-FORCE (explainable, not proof of security)
        "charset_size": charset_size,
        "estimated_entropy": round(estimated_entropy, 2),
        "shannon_entropy": round(shannon_entropy, 2),
        "brute_force_combinations_log2": round(brute_force_combinations_log2, 2),
        "brute_force_search_space": brute_force_search_space,
    }
