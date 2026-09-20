"""
agent/risk_engine.py

Heuristic risk scoring - deterministic engineering assessment.

Distinctions:
- DATASET MODEL: Learns patterns from provided dataset (heavily length-driven, 99.98% length-only accuracy)
- SECURITY RULE ENGINE: Deterministic heuristics (pattern_analyzer) based on composition/diversity
- FINAL HEURISTIC RISK: Combines signals into 0-100 Heuristic Security Risk Score (NOT calibrated probability)

ML result is OPTIONAL and never automatically determines final result.
Long predictable passwords are NOT considered secure just because dataset model predicts class 2.
"""

from typing import Dict, Optional

# Risk level thresholds are documented engineering thresholds, not dataset-derived
# Chosen to give intuitive buckets; can be adjusted with documented reasoning
LEVEL_THRESHOLDS = [
    (80, "Low Risk"),
    (60, "Lower Risk"),
    (30, "Moderate Risk"),
    (0,  "High Risk"),
]

def _level_for_score(score: int) -> str:
    for thresh, label in LEVEL_THRESHOLDS:
        if score >= thresh:
            return label
    return "High Risk"

def evaluate_risk(
    features: Dict,
    pattern_analysis: Dict,
    ml_result: Optional[Dict] = None
) -> Dict:
    """
    Combine length, diversity, composition, pattern findings, and optional ML evidence
    into a transparent heuristic score 0-100.

    Args:
        features: dict from extract_features (must not contain password)
        pattern_analysis: dict from analyze_patterns with issues/warnings/positive_signals
        ml_result: optional {predicted_class: 0/1/2, confidence: 0-1} - displayed separately

    Returns:
        {
            "heuristic_score": 0-100,
            "risk_level": "High Risk" | "Moderate Risk" | "Lower Risk" | "Low Risk",
            "issues": [...],
            "warnings": [...],
            "positive_signals": [...],
            "ml_evidence": {...} or None,
            "explanations": [...],
            "scoring_breakdown": {component: points}
        }
        No password is included in returned object.
        Deterministic: same input always produces same result.
    """
    if not isinstance(features, dict) or not isinstance(pattern_analysis, dict):
        raise TypeError("features and pattern_analysis must be dicts")

    # Validate ml_result if provided (do not hardcode, do not load model here)
    ml_evidence = None
    if ml_result is not None:
        if not isinstance(ml_result, dict):
            raise TypeError("ml_result must be dict or None")
        # Expect predicted_class and confidence if provided, but be tolerant
        if "predicted_class" in ml_result or "confidence" in ml_result:
            ml_evidence = {
                "predicted_class": ml_result.get("predicted_class"),
                "confidence": ml_result.get("confidence"),
                "note": "ML dataset evidence - shows what length-driven dataset model predicts, NOT real-world security verdict"
            }
        else:
            ml_evidence = {"note": "ML evidence provided but missing predicted_class/confidence", "raw": str(ml_result)[:100]}

    # --- Transparent heuristic scoring 0-100 ---
    # Start at 50 (neutral), add positives, subtract negatives, clamp 0-100
    # Documented contributions:

    score = 50
    breakdown: Dict[str, int] = {}

    n = features.get("length", 0)
    diversity = features.get("character_diversity_ratio", 0)
    type_count = features.get("unique_character_type_count", 0)
    max_run = features.get("maximum_repeated_run", 0)

    # Positive signals
    # Adequate length: >=8 gets +10, >=12 gets additional +10, >=16 gets +5
    length_points = 0
    if n >= 8:
        length_points += 10
    if n >= 12:
        length_points += 10
    if n >= 16:
        length_points += 5
    if n >= 20:
        length_points += 5
    score += length_points
    breakdown["length"] = length_points

    # Character diversity: high diversity positive
    diversity_points = 0
    if diversity >= 0.7 and n >= 8:
        diversity_points = 10
    elif diversity >= 0.5 and n >= 8:
        diversity_points = 5
    elif diversity < 0.3 and n >= 8:
        diversity_points = -5  # negative handled below but also here for transparency
    score += diversity_points
    breakdown["diversity"] = diversity_points

    # Multiple character types: 2 types +5, 3 types +10, 4 types +15
    type_points = 0
    if type_count == 2:
        type_points = 5
    elif type_count == 3:
        type_points = 10
    elif type_count >= 4:
        type_points = 15
    score += type_points
    breakdown["character_types"] = type_points

    # Mixed case bonus
    mixed_bonus = 5 if features.get("has_mixed_case") else 0
    score += mixed_bonus
    breakdown["mixed_case"] = mixed_bonus

    # Negative signals
    # Very short length: <6 high penalty, <4 even more
    short_penalty = 0
    if n == 0:
        short_penalty = -30
    elif n < 4:
        short_penalty = -25
    elif n < 6:
        short_penalty = -15
    elif n < 8:
        short_penalty = -5
    score += short_penalty
    breakdown["short_length"] = short_penalty

    # Pattern findings penalties
    issues = pattern_analysis.get("issues", [])
    warnings = pattern_analysis.get("warnings", [])
    positive = pattern_analysis.get("positive_signals", [])

    # Count by code to avoid double-counting and keep deterministic
    issue_codes = [it.get("code") for it in issues]
    warning_codes = [it.get("code") for it in warnings]

    repeated_penalty = 0
    if "repeated_characters" in issue_codes:
        # high severity repeated (4+)
        repeated_penalty -= 15
    elif "repeated_characters" in warning_codes:
        repeated_penalty -= 8
    score += repeated_penalty
    breakdown["repeated"] = repeated_penalty

    seq_penalty = 0
    seq_count = issue_codes.count("sequential_digits") + issue_codes.count("sequential_letters")
    if seq_count >= 2:
        seq_penalty = -20
    elif seq_count == 1:
        seq_penalty = -12
    score += seq_penalty
    breakdown["sequential"] = seq_penalty

    block_penalty = -12 if "repeated_block" in issue_codes else 0
    score += block_penalty
    breakdown["repeated_block"] = block_penalty

    low_div_penalty = 0
    if "low_diversity" in issue_codes:
        low_div_penalty = -12
    elif "low_diversity" in warning_codes:
        low_div_penalty = -6
    score += low_div_penalty
    breakdown["low_diversity"] = low_div_penalty

    single_type_penalty = 0
    if any(c.startswith("single_type") for c in warning_codes):
        single_type_penalty = -10
    score += single_type_penalty
    breakdown["single_type"] = single_type_penalty

    predictable_penalty = -10 if "predictable_structure" in warning_codes else 0
    score += predictable_penalty
    breakdown["predictable_structure"] = predictable_penalty
    # Additional penalty for combination: long but highly predictable (both sequential and predictable)
    # Ensures long predictable is not automatically considered Low Risk despite length
    combo_penalty = 0
    if "predictable_structure" in warning_codes and seq_count >= 1:
        combo_penalty = -8
    score += combo_penalty
    breakdown["predictable_sequential_combo"] = combo_penalty

    # --- New pattern penalties (explainable, deterministic) ---
    common_penalty = 0
    if "common_password" in issue_codes:
        common_penalty = -20
    elif "common_pattern" in warning_codes:
        common_penalty = -12
    score += common_penalty
    breakdown["common_password"] = common_penalty

    dict_penalty = 0
    if "dictionary_word" in warning_codes or "dictionary_word" in issue_codes:
        dict_penalty = -8
    score += dict_penalty
    breakdown["dictionary_word"] = dict_penalty

    keyboard_penalty = 0
    if "keyboard_pattern" in issue_codes:
        keyboard_penalty = -12
    score += keyboard_penalty
    breakdown["keyboard_pattern"] = keyboard_penalty

    substitution_penalty = 0
    if "predictable_substitution" in warning_codes:
        substitution_penalty = -6
    score += substitution_penalty
    breakdown["predictable_substitution"] = substitution_penalty

    year_penalty = 0
    if "year_pattern" in warning_codes or "date_pattern" in warning_codes:
        year_penalty = -6
    score += year_penalty
    breakdown["year_date_pattern"] = year_penalty

    personal_penalty = 0
    if "possible_personal_pattern" in warning_codes:
        personal_penalty = -4
    score += personal_penalty
    breakdown["personal_pattern"] = personal_penalty

    # Clamp 0-100 deterministic
    score = max(0, min(100, score))
    heuristic_score = int(round(score))

    risk_level = _level_for_score(heuristic_score)

    # --- Entropy & Brute-force analysis (explainable, not proof of security) ---
    # Use features if available; do not treat entropy alone as secure
    charset_size = features.get("charset_size", 0)
    estimated_entropy = features.get("estimated_entropy", 0)
    shannon_entropy = features.get("shannon_entropy", 0)
    brute_log2 = features.get("brute_force_combinations_log2", 0)
    entropy_analysis = {
        "charset_size": charset_size,
        "estimated_entropy_bits": estimated_entropy,
        "shannon_entropy_bits": shannon_entropy,
        "note": "High entropy indicates larger theoretical search space, but real-world guessability also depends on common patterns, dictionary words, and predictable structures. High entropy alone does not guarantee security."
    }
    # Brute-force resistance: human-readable search space
    brute_force_analysis = {
        "search_space_log2": brute_log2,
        "search_space_display": features.get("brute_force_search_space", ""),
        "note": "Estimated search space assumes random generation from charset. Predictable patterns significantly reduce effective resistance; avoid false precision."
    }

    # --- Attack pattern analysis (which guessing strategies could be effective) ---
    attack_patterns = []
    # Dictionary attacks
    if "common_password" in issue_codes or "common_pattern" in warning_codes or "dictionary_word" in warning_codes or "dictionary_word" in issue_codes:
        attack_patterns.append({
            "attack": "dictionary",
            "risk": "high" if "common_password" in issue_codes else "medium",
            "explanation": "Dictionary/common-password guessing could be effective due to common word/pattern."
        })
    # Pattern attacks (sequential, keyboard, repetition, predictable structure)
    if any(c in issue_codes for c in ["sequential_digits", "sequential_letters", "keyboard_pattern", "repeated_characters", "repeated_block", "predictable_structure"]) or "predictable_substitution" in warning_codes:
        attack_patterns.append({
            "attack": "pattern",
            "risk": "medium",
            "explanation": "Pattern-based guessing (sequences, repeats, keyboard walks, predictable substitutions) could be effective."
        })
    # Brute-force
    if n < 8 or type_count == 1:
        attack_patterns.append({
            "attack": "brute_force",
            "risk": "high" if n < 6 else "medium",
            "explanation": "Brute-force guessing is more feasible for short or single-type passwords."
        })
    elif n >= 12 and type_count >= 3 and diversity >= 0.6:
        attack_patterns.append({
            "attack": "brute_force",
            "risk": "low",
            "explanation": "Brute-force is less feasible due to length and diversity, but not impossible."
        })
    # Credential stuffing - cannot be determined from password alone
    attack_patterns.append({
        "attack": "credential_stuffing",
        "risk": "unknown",
        "explanation": "Credential stuffing exposure cannot be determined from this password alone; password reuse across services increases this risk."
    })
    # If no specific attack identified, add generic
    if not any(a["attack"] in ["dictionary", "pattern", "brute_force"] for a in attack_patterns):
        attack_patterns.append({
            "attack": "generic",
            "risk": "low",
            "explanation": "No obvious pattern-based attack vector detected, but general guessing still possible."
        })

    # Build human-readable explanations (deterministic, no LLM)
    explanations = []
    if n == 0:
        explanations.append("Password is empty.")
    elif n < 6:
        explanations.append("Password is very short.")
    elif n >= 12:
        explanations.append("Has substantial length.")

    if type_count >= 3:
        explanations.append("Uses multiple character types.")
    elif type_count == 1:
        explanations.append("Uses only a single character type.")

    if diversity < 0.4 and n >= 8:
        explanations.append("Character diversity is low.")
    elif diversity >= 0.7:
        explanations.append("Character diversity is high.")

    # Include pattern messages (already human-readable) without adding password
    for it in issues:
        msg = it.get("message", "")
        if msg and msg not in explanations:
            explanations.append(msg)
    for it in warnings:
        # Avoid duplicate messages already in issues
        msg = it.get("message", "")
        if msg and msg not in explanations:
            # Only add warnings that are not already covered as issues
            if it.get("code") not in issue_codes:
                explanations.append(msg)

    for sig in positive:
        msg = sig.get("message", "")
        if msg and msg not in explanations:
            explanations.append(msg)

    # The ML result is displayed separately and does NOT affect heuristic_score
    # This is critical because dataset is length-driven: long predictable must not be considered secure just because ML predicts 2

    # --- Specific recommendations based on detected problems ---
    recommendations = []
    if n == 0:
        recommendations.append("Enter a password to analyze.")
    if n > 0 and n < 8:
        recommendations.append("Increase the password length and avoid predictable year suffixes.")
    if n >= 8 and n < 12:
        recommendations.append("Consider increasing length to 12+ characters for better resistance.")
    if "common_password" in issue_codes or "common_pattern" in warning_codes:
        recommendations.append("Completely change this password; it matches a commonly used pattern from local dataset.")
    if "dictionary_word" in warning_codes:
        recommendations.append("Avoid dictionary words; use unrelated words or random generation.")
    if "sequential_digits" in issue_codes or "sequential_letters" in issue_codes or "keyboard_pattern" in issue_codes:
        recommendations.append("Remove predictable sequences and keyboard patterns.")
    if "repeated_characters" in issue_codes or "repeated_block" in issue_codes or "low_diversity" in issue_codes:
        recommendations.append("Remove repeated characters and increase character diversity.")
    if "single_type_digits" in warning_codes or "single_type_lowercase" in warning_codes or "single_type_uppercase" in warning_codes:
        recommendations.append("Use multiple character types (lowercase, uppercase, numbers, symbols).")
    if "predictable_substitution" in warning_codes:
        recommendations.append("Don't rely on predictable substitutions like @ for a; they add little security.")
    if "year_pattern" in warning_codes or "date_pattern" in warning_codes:
        recommendations.append("Avoid obvious year or date patterns.")
    if "possible_personal_pattern" in warning_codes:
        recommendations.append("Avoid structures that could be personal information (name + year).")
    if not recommendations and risk_level in ["Low Risk", "Lower Risk"]:
        recommendations.append("Password shows good properties; consider using a password manager for uniqueness and MFA for important accounts.")
    # Ensure at least one recommendation for high risk
    if not recommendations and risk_level == "High Risk":
        recommendations.append("Increase length, add character diversity, and avoid common patterns.")

    return {
        "heuristic_score": heuristic_score,
        "risk_level": risk_level,
        "issues": issues,
        "warnings": warnings,
        "positive_signals": positive,
        "ml_evidence": ml_evidence,
        "explanations": explanations,
        "recommendations": recommendations,
        "attack_analysis": attack_patterns,
        "entropy_analysis": entropy_analysis,
        "brute_force_analysis": brute_force_analysis,
        "scoring_breakdown": breakdown,
        # Transparency note
        "note": "Heuristic Security Risk Score is an engineering assessment, not a calibrated probability or crack-time estimate",
    }
