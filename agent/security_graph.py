"""
agent/security_graph.py

LangGraph orchestration around deterministic security engine.
Phase 4 extends with RAG + Llama explanation (orchestration only).

- LangGraph = orchestration only, NOT intelligence
- Feature extractor = deterministic structural analysis
- Pattern analyzer = deterministic pattern detection
- ML analyzer = dataset-derived supporting evidence (heavily length-driven)
- Risk engine = authoritative security assessment (heuristic, not ML)
- RAG = security knowledge retrieval (findings-based, never password)
- Llama = explanation generation (fallback deterministic if no model)

Privacy:
- Password exists only temporarily in in-memory graph state during one invocation
- No persistent checkpointing, no DB/files/logs, no caching of raw passwords
- Graph is stateless between requests
- Never prints/logs/saves password, never includes password in exception messages or output
- RAG query and LLM input are built from findings, never raw password
"""

from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, START, END

from .feature_extractor import extract_features
from .pattern_analyzer import analyze_patterns
from .ml_analyzer import analyze_with_ml
from .risk_engine import evaluate_risk
from .rag import retrieve_knowledge
from .llm_explainer import generate_explanation

# ------------------------------------------------------------
# State: only necessary in-memory fields, no persistence
# ------------------------------------------------------------
class SecurityState(TypedDict, total=False):
    password: str
    features: Dict[str, Any]
    pattern_analysis: Dict[str, Any]
    ml_result: Dict[str, Any]
    risk_result: Dict[str, Any]
    rag_knowledge: Dict[str, Any]
    explanation: Dict[str, Any]

# ------------------------------------------------------------
# Nodes - each calls existing Phase 1/2 functions, no rewriting
# ------------------------------------------------------------
def feature_node(state: SecurityState) -> Dict[str, Any]:
    """Calls existing extract_features() - deterministic, no password stored beyond state."""
    pw = state.get("password", "")
    # Validate type without exposing password in exception
    if not isinstance(pw, str):
        raise TypeError("password must be a string")
    feats = extract_features(pw)
    return {"features": feats}

def pattern_node(state: SecurityState) -> Dict[str, Any]:
    """Calls existing analyze_patterns() with password + features."""
    pw = state.get("password", "")
    feats = state.get("features")
    if feats is None:
        # Fallback if called out of order (should not happen in linear graph)
        from .feature_extractor import extract_features as _ef
        feats = _ef(pw)
    analysis = analyze_patterns(pw, feats)
    return {"pattern_analysis": analysis}

def ml_node(state: SecurityState) -> Dict[str, Any]:
    """Calls existing analyze_with_ml() - optional evidence, never overrides heuristic."""
    pw = state.get("password", "")
    ml = analyze_with_ml(pw)
    return {"ml_result": ml}

def risk_node(state: SecurityState) -> Dict[str, Any]:
    """Calls existing evaluate_risk() with features, patterns, and optional ML."""
    feats = state.get("features")
    patterns = state.get("pattern_analysis")
    ml = state.get("ml_result")
    if feats is None or patterns is None:
        raise ValueError("Missing features or pattern_analysis for risk evaluation")
    # Convert ml_analyzer output to risk_engine ml_result format
    # ml_analyzer returns {available, predicted_class, confidence, ...}
    # risk_engine expects {predicted_class, confidence} or None
    ml_for_risk = None
    if ml is not None and ml.get("available"):
        ml_for_risk = {
            "predicted_class": ml.get("predicted_class"),
            "confidence": ml.get("confidence"),
        }
        # Also pass through class_probabilities if needed for future, but risk_engine currently uses only these two
    risk = evaluate_risk(feats, patterns, ml_result=ml_for_risk)
    # Attach full ml evidence separately for reporting (risk already contains ml_evidence copy)
    # Keep ml evidence distinct
    return {"risk_result": risk}

def rag_node(state: SecurityState) -> Dict[str, Any]:
    """RAG retrieval based on security findings, never raw password."""
    feats = state.get("features", {})
    patterns = state.get("pattern_analysis", {})
    risk = state.get("risk_result", {})
    ml = state.get("ml_result", {})
    # Build findings dict without password
    findings = {
        "risk_level": risk.get("risk_level", "") if risk else "",
        "heuristic_score": risk.get("heuristic_score", 0) if risk else 0,
        "issues": patterns.get("issues", []) if patterns else [],
        "warnings": patterns.get("warnings", []) if patterns else [],
        "positive_signals": patterns.get("positive_signals", []) if patterns else [],
        "features": feats,
        "ml": ml,
    }
    rag_result = retrieve_knowledge(findings, top_k=3)
    return {"rag_knowledge": rag_result}

def llm_explanation_node(state: SecurityState) -> Dict[str, Any]:
    """LLama explanation generation from findings + RAG knowledge, never password."""
    risk = state.get("risk_result", {})
    patterns = state.get("pattern_analysis", {})
    feats = state.get("features", {})
    ml = state.get("ml_result", {})
    rag = state.get("rag_knowledge", {})
    # Build findings for LLM (no password)
    findings = {
        "risk_level": risk.get("risk_level", "Unknown") if risk else "Unknown",
        "heuristic_score": risk.get("heuristic_score", 0) if risk else 0,
        "issues": patterns.get("issues", []) if patterns else [],
        "warnings": patterns.get("warnings", []) if patterns else [],
        "positive_signals": patterns.get("positive_signals", []) if patterns else [],
        "relevant_knowledge": rag.get("results", []) if rag else [],
        "ml": ml,
    }
    explanation = generate_explanation(findings)
    return {"explanation": explanation}

# ------------------------------------------------------------
# Build graph - linear deterministic orchestration, no checkpointing
# Phase 3: START -> feature -> pattern -> ml -> risk -> END
# Phase 4: adds rag -> llm_explanation after risk
# ------------------------------------------------------------
def _build_graph():
    graph = StateGraph(SecurityState)
    graph.add_node("feature_node", feature_node)
    graph.add_node("pattern_node", pattern_node)
    graph.add_node("ml_node", ml_node)
    graph.add_node("risk_node", risk_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("llm_explanation_node", llm_explanation_node)

    graph.add_edge(START, "feature_node")
    graph.add_edge("feature_node", "pattern_node")
    graph.add_edge("pattern_node", "ml_node")
    graph.add_edge("ml_node", "risk_node")
    graph.add_edge("risk_node", "rag_node")
    graph.add_edge("rag_node", "llm_explanation_node")
    graph.add_edge("llm_explanation_node", END)

    # No checkpointer -> stateless between requests, no persistence
    compiled = graph.compile()
    return compiled

# Singleton compiled graph (stateless, no checkpoint)
_graph = None

def _get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph

# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def analyze_password(password: str) -> Dict[str, Any]:
    """
    Invoke LangGraph orchestration for a single password.
    Phase 4 returns extended result with RAG + explanation.

    Args:
        password: password string (synthetic test strings only; never logged)

    Returns:
        {
            "risk": risk_result (heuristic_score, risk_level, issues, warnings, positive_signals, ml_evidence, explanations, ...),
            "features": features dict,
            "patterns": pattern_analysis dict,
            "ml": ml_result dict (with warning, predicted_class, confidence, class_probabilities),
            "rag": rag_knowledge dict (query, results, retrieval_available),
            "explanation": explanation dict (summary, why, recommendations, sources, provider)
        }
        Never includes raw password. Heuristic risk is authoritative; ML is supporting evidence;
        RAG is knowledge retrieval; Llama is explanation only.
    """
    if not isinstance(password, str):
        raise TypeError("password must be a string")

    graph = _get_graph()
    # Invoke graph with password only in initial state; no persistence
    result_state = graph.invoke({"password": password})

    # Build clean public response - never return raw password
    return {
        "risk": result_state.get("risk_result"),
        "features": result_state.get("features"),
        "patterns": result_state.get("pattern_analysis"),
        "ml": result_state.get("ml_result"),
        "rag": result_state.get("rag_knowledge"),
        "explanation": result_state.get("explanation"),
    }

# Expose graph for testing/inspection (without exposing checkpoint)
__all__ = ["analyze_password", "_build_graph", "_get_graph"]
