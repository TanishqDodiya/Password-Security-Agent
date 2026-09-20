"""
agent package - Core Password Security Engine

Components:
- feature_extractor: deterministic feature extraction (no passwords stored)
- pattern_analyzer: rule-based weak pattern detection
- risk_engine: heuristic risk scoring (not calibrated probability)

Dataset model vs Security rule engine distinction documented in risk_engine.py
"""

from .feature_extractor import extract_features
from .pattern_analyzer import analyze_patterns
from .risk_engine import evaluate_risk

__all__ = ["extract_features", "analyze_patterns", "evaluate_risk"]
