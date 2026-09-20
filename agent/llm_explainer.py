"""
agent/llm_explainer.py - Llama explanation layer with provider abstraction.

- LLM must explain already-computed security assessment, not decide it
- LLM must NOT change heuristic_score, risk_level, or generate security probability
- Input is structured findings + retrieved knowledge, never raw password
- Deterministic fallback ensures code is runnable without actual Llama model
- Clearly reports whether real Llama or fallback is used
"""

from typing import Dict, Any, List, Optional
import os

# Provider abstraction - try to detect available Llama runtime without blindly installing huge models
def _detect_llama_available() -> tuple[bool, str]:
    """
    Inspect environment for available Llama runtime.
    Checks for: ollama, llama_cpp, transformers with llama model, etc.
    Returns (available, runtime_name)
    """
    # Check for ollama (common local Llama runtime)
    try:
        import importlib.util
        if importlib.util.find_spec("ollama") is not None:
            return True, "ollama"
    except Exception:
        pass
    # Check for llama_cpp
    try:
        import importlib.util
        if importlib.util.find_spec("llama_cpp") is not None:
            return True, "llama_cpp"
    except Exception:
        pass
    # Check for transformers with local model (but don't assume model downloaded)
    try:
        import importlib.util
        if importlib.util.find_spec("transformers") is not None:
            # Transformers is not installed in this env (we checked), so not available
            # Even if installed, would need model weights, so treat as not available for MVP
            pass
    except Exception:
        pass
    # Check for openai (not Llama, but could be used) - not considered Llama
    return False, "fallback"

_LLAMA_AVAILABLE, _LLAMA_RUNTIME = _detect_llama_available()

def is_llama_available() -> bool:
    return _LLAMA_AVAILABLE

def get_llama_runtime() -> str:
    return _LLAMA_RUNTIME

def _build_llm_prompt(findings: Dict[str, Any]) -> str:
    """
    Build prompt for LLM from structured findings.
    Never includes raw password.
    Explicitly instructs model not to invent, not to change risk level/score.
    """
    risk_level = findings.get("risk_level", "Unknown")
    heuristic_score = findings.get("heuristic_score", 0)
    issues = findings.get("issues", [])
    warnings = findings.get("warnings", [])
    positives = findings.get("positive_signals", [])
    ml = findings.get("ml", {})
    knowledge = findings.get("relevant_knowledge", [])

    # Build findings summary without password
    parts = []
    parts.append(f"Security Assessment: {risk_level} (Heuristic Score: {heuristic_score}/100)")
    if issues:
        parts.append("Detected issues:")
        for it in issues:
            parts.append(f"- {it.get('code')}: {it.get('message')} (severity: {it.get('severity')})")
    if warnings:
        parts.append("Warnings:")
        for it in warnings:
            parts.append(f"- {it.get('code')}: {it.get('message')} (severity: {it.get('severity')})")
    if positives:
        parts.append("Positive signals:")
        for it in positives:
            parts.append(f"- {it.get('message')}")

    if ml and ml.get("available"):
        parts.append(f"ML Dataset Evidence: predicted class {ml.get('predicted_class')} with confidence {ml.get('confidence', 0):.2f} (class probabilities: {ml.get('class_probabilities', {})})")
        parts.append(f"ML Warning: {ml.get('warning', '')}")

    if knowledge:
        parts.append("Relevant security knowledge:")
        for k in knowledge:
            parts.append(f"- [{k.get('source')}] {k.get('content')[:200]}")

    findings_text = "\n".join(parts)

    prompt = f"""You are a password security explainer. You must explain the already-computed security assessment.

Rules (non-negotiable):
- Do NOT invent security findings.
- Do NOT change the supplied risk level or heuristic score.
- Do NOT calculate a new security score.
- Explain only the supplied findings.
- Give practical recommendations.
- Do NOT mention or reproduce the password.
- Do NOT claim guaranteed protection against attacks.
- If ML evidence is present, explain that it is dataset-derived and heavily length-driven, not real-world security truth.

Findings:
{findings_text}

Relevant security knowledge:
{chr(10).join([k.get('content','')[:300] for k in knowledge]) if knowledge else 'No additional knowledge'}

Task:
Provide a concise summary, why the assessment was given, and practical recommendations. Use the sources provided.
"""
    return prompt

def _fallback_explanation(findings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic fallback when real Llama is not available.
    Generates structured explanation from findings without LLM.
    Clearly marked as fallback.
    """
    risk_level = findings.get("risk_level", "Unknown")
    heuristic_score = findings.get("heuristic_score", 0)
    issues = findings.get("issues", [])
    warnings = findings.get("warnings", [])
    positives = findings.get("positive_signals", [])
    knowledge = findings.get("relevant_knowledge", [])
    ml = findings.get("ml", {})

    # Summary based on risk_level
    if risk_level == "High Risk":
        summary = f"Assessment is High Risk (score {heuristic_score}/100). Immediate improvement recommended."
    elif risk_level == "Moderate Risk":
        summary = f"Assessment is Moderate Risk (score {heuristic_score}/100). Several weaknesses detected."
    elif risk_level == "Lower Risk":
        summary = f"Assessment is Lower Risk (score {heuristic_score}/100). Some strengths present but issues remain."
    elif risk_level == "Low Risk":
        summary = f"Assessment is Low Risk (score {heuristic_score}/100). Strong heuristic signals, but no guarantee of uncrackability."
    else:
        summary = f"Assessment is {risk_level} (score {heuristic_score}/100)."

    why = []
    for it in issues:
        why.append(it.get("message", "Issue detected."))
    for it in warnings:
        # Avoid duplicate messages
        msg = it.get("message", "")
        if msg and msg not in why:
            why.append(msg)
    if not why and positives:
        why.append("No critical issues detected, but review positive signals.")

    # Add knowledge-based why if available
    if knowledge:
        why.append(f"Relevant knowledge: {knowledge[0].get('content','')[:120]}")

    recommendations = []
    # Map issues to recommendations deterministically
    code_to_rec = {
        "sequential_digits": "Avoid sequential digit patterns like 123.",
        "sequential_letters": "Avoid alphabetical sequences like abc.",
        "repeated_characters": "Avoid excessive repetition of the same character.",
        "repeated_block": "Avoid repeated blocks like abab.",
        "low_diversity": "Increase character diversity and unique characters.",
        "single_type_digits": "Use multiple character types, not only digits.",
        "single_type_lowercase": "Include uppercase, digits, or symbols.",
        "single_type_uppercase": "Include lowercase, digits, or symbols.",
        "single_type_special": "Combine with letters and digits.",
        "predictable_structure": "Avoid predictable structure like letters followed only by digits.",
    }
    for it in issues + warnings:
        code = it.get("code")
        if code in code_to_rec and code_to_rec[code] not in recommendations:
            recommendations.append(code_to_rec[code])

    # Generic recommendations based on risk
    if risk_level in ["High Risk", "Moderate Risk"]:
        if "Use a longer password (12+ characters) with mixed types." not in recommendations:
            recommendations.append("Use a longer password (12+ characters) with mixed types.")
        if "Consider a password manager for unique, random passwords." not in recommendations:
            recommendations.append("Consider a password manager for unique, random passwords.")
        if "Enable multi-factor authentication (MFA) where available." not in recommendations:
            recommendations.append("Enable multi-factor authentication (MFA) where available.")

    if positives:
        # Acknowledge positives
        for sig in positives:
            msg = sig.get("message", "")
            if msg and "multiple character types" in msg.lower():
                if "Continued use of multiple types is positive." not in recommendations:
                    pass  # not a recommendation, just positive

    if not recommendations:
        recommendations.append("Maintain current practices and consider MFA.")

    sources = list(set([k.get("source", "knowledge") for k in knowledge if k.get("source")]))
    if not sources and knowledge:
        sources = ["knowledge/password_length.md"]
    elif not sources:
        sources = ["knowledge/password_length.md", "knowledge/common_patterns.md"]

    return {
        "explanation_available": True,
        "provider": "fallback",
        "summary": summary,
        "why": why,
        "recommendations": recommendations,
        "sources": sources,
        "note": "Fallback deterministic explanation (Llama not available). Heuristic score and risk level unchanged.",
    }

def generate_explanation(findings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate human-readable explanation from findings + retrieved knowledge.

    Args:
        findings: {
            risk_level, heuristic_score, issues, warnings, positive_signals,
            relevant_knowledge: [{"content","source","score"},...],
            ml: {...}
        }  # No password

    Returns:
        {
            explanation_available: bool,
            provider: "llama" | "fallback",
            summary: str,
            why: [str],
            recommendations: [str],
            sources: [str],
            ... (provider-specific)
        }
        Never includes password.
    """
    # Validate that findings do not contain password (privacy check)
    # We check for common password-like keys, but not exhaustive
    if "password" in findings:
        # Remove it without logging it
        findings = {k: v for k, v in findings.items() if k != "password"}

    # Build prompt (for logging/debugging, but never include password)
    prompt = _build_llm_prompt(findings)

    # Try real Llama if available
    if _LLAMA_AVAILABLE:
        # Placeholder for real Llama invocation - would call ollama/llama_cpp here
        # Since we detected availability but don't have a model to run in MVP,
        # we still use fallback but mark as attempted
        # In a real deployment, this would call the model with the prompt
        # For now, we return fallback but indicate Llama was detected
        fallback = _fallback_explanation(findings)
        fallback["provider"] = f"{_LLAMA_RUNTIME} (fallback used, model not loaded)"
        fallback["prompt"] = prompt[:500]  # truncated prompt for debugging, no password
        return fallback
    else:
        result = _fallback_explanation(findings)
        result["prompt"] = prompt[:500]
        return result

def explanation_available() -> bool:
    # Always available via fallback
    return True
