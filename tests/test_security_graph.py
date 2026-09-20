"""
tests/test_security_graph.py

LangGraph orchestration tests - Phase 3.
Synthetic passwords only, no real dataset, no passwords printed.
Verifies graph topology, privacy, determinism, and heuristic/ML separation.
"""

import unittest
import os
import glob

from agent.security_graph import analyze_password, _build_graph, _get_graph
from agent.feature_extractor import extract_features
from agent.pattern_analyzer import analyze_patterns
from agent.risk_engine import evaluate_risk
from agent.ml_analyzer import analyze_with_ml

# Synthetic passwords for testing (not real leaked credentials)
SYN_EMPTY = ""
SYN_ONE = "x"
SYN_NORMAL = "zxqpmkva"  # length 8, diverse
SYN_LONG_PREDICTABLE = "syntheticLetters123456"  # letters + sequential digits, long
SYN_LONG_MIXED = "Tr0ub4dor&XyZ!9qLmExtraLong123"  # long mixed
SYN_VERY_SHORT = "ab"

class TestSecurityGraph(unittest.TestCase):

    def test_graph_can_be_constructed(self):
        graph = _build_graph()
        self.assertIsNotNone(graph)
        # Graph should have 4 nodes in linear topology
        # Check via get_graph singleton
        g2 = _get_graph()
        self.assertIsNotNone(g2)

    def test_graph_can_analyze_synthetic(self):
        result = analyze_password(SYN_NORMAL)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)

    def test_output_contains_risk_result(self):
        result = analyze_password(SYN_NORMAL)
        self.assertIn("risk", result)
        risk = result["risk"]
        self.assertIn("heuristic_score", risk)
        self.assertIn("risk_level", risk)
        self.assertIn(risk["risk_level"], ["High Risk", "Moderate Risk", "Lower Risk", "Low Risk"])
        self.assertGreaterEqual(risk["heuristic_score"], 0)
        self.assertLessEqual(risk["heuristic_score"], 100)

    def test_output_contains_pattern_analysis(self):
        result = analyze_password(SYN_NORMAL)
        self.assertIn("patterns", result)
        patterns = result["patterns"]
        self.assertIn("issues", patterns)
        self.assertIn("warnings", patterns)
        self.assertIn("positive_signals", patterns)

    def test_output_contains_ml_evidence(self):
        result = analyze_password(SYN_NORMAL)
        self.assertIn("ml", result)
        ml = result["ml"]
        self.assertTrue(ml.get("available"))
        self.assertIn("predicted_class", ml)
        self.assertIn(ml["predicted_class"], [0, 1, 2])
        self.assertIn("warning", ml)
        self.assertIn("heavily length-driven", ml["warning"])

    def test_features_present(self):
        result = analyze_password(SYN_NORMAL)
        self.assertIn("features", result)
        feats = result["features"]
        for key in ["length", "lowercase_count", "uppercase_count", "digit_count", "special_count",
                    "character_diversity_ratio", "maximum_repeated_run"]:
            self.assertIn(key, feats)

    def test_graph_does_not_return_password(self):
        for pw in [SYN_NORMAL, SYN_LONG_MIXED, SYN_LONG_PREDICTABLE]:
            result = analyze_password(pw)
            # No key should be named password and no value should equal pw
            self.assertNotIn("password", result)
            # Check that none of the top-level values equals pw
            for k, v in result.items():
                if isinstance(v, str):
                    self.assertNotEqual(v, pw)
            # Also check nested risk/patterns/ml don't contain pw as exact value
            blob = str(result)
            if len(pw) >= 4:
                self.assertNotIn(pw, blob, "password leaked in output")

    def test_synthetic_not_in_stringified_output(self):
        pw = SYN_LONG_PREDICTABLE
        result = analyze_password(pw)
        blob = str(result)
        self.assertNotIn(pw, blob)

    def test_long_predictable_independent_of_ml(self):
        # Long predictable ML predicts 2 with high confidence, but heuristic must not be Low Risk automatically
        result = analyze_password(SYN_LONG_PREDICTABLE)
        ml_class = result["ml"].get("predicted_class")
        risk_level = result["risk"]["risk_level"]
        # ML will predict 2 due to length 22
        self.assertEqual(ml_class, 2)
        # Heuristic should penalize sequential+predictable, so not Low Risk (should be Lower/Moderate)
        self.assertNotEqual(risk_level, "Low Risk", "Long predictable must not be auto Low Risk despite ML 2")
        # Verify heuristic_score <80 for this case
        self.assertLess(result["risk"]["heuristic_score"], 80)

    def test_ml_does_not_override_heuristic(self):
        pw = SYN_LONG_PREDICTABLE
        # Direct heuristic without ML
        feats = extract_features(pw)
        patterns = analyze_patterns(pw, feats)
        direct = evaluate_risk(feats, patterns, ml_result=None)
        # Via graph with ML
        graph_result = analyze_password(pw)
        self.assertEqual(direct["heuristic_score"], graph_result["risk"]["heuristic_score"])
        self.assertEqual(direct["risk_level"], graph_result["risk"]["risk_level"])
        # ML evidence present separately
        self.assertIsNotNone(graph_result["risk"]["ml_evidence"])
        self.assertIsNotNone(graph_result["ml"])

    def test_deterministic_repeated_invocation(self):
        pw = SYN_NORMAL
        r1 = analyze_password(pw)
        r2 = analyze_password(pw)
        self.assertEqual(r1["risk"]["heuristic_score"], r2["risk"]["heuristic_score"])
        self.assertEqual(r1["risk"]["risk_level"], r2["risk"]["risk_level"])
        self.assertEqual(r1["ml"]["predicted_class"], r2["ml"]["predicted_class"])
        if "confidence" in r1["ml"] and "confidence" in r2["ml"]:
            self.assertAlmostEqual(r1["ml"]["confidence"], r2["ml"]["confidence"], places=5)

    def test_empty_password_handled(self):
        result = analyze_password(SYN_EMPTY)
        self.assertIn("risk", result)
        self.assertEqual(result["features"]["length"], 0)
        self.assertEqual(result["risk"]["risk_level"], "High Risk")
        # No exception, no password leak
        self.assertNotIn("password", result)

    def test_very_short_handled(self):
        result = analyze_password(SYN_VERY_SHORT)
        self.assertIn("risk", result)
        self.assertEqual(result["features"]["length"], 2)
        self.assertIn(result["risk"]["risk_level"], ["High Risk", "Moderate Risk", "Lower Risk", "Low Risk"])
        # Very short should be High Risk
        self.assertEqual(result["risk"]["risk_level"], "High Risk")

    def test_no_filesystem_checkpoint_artifact(self):
        # Capture file list before
        before_files = set(glob.glob("**/*", recursive=True))
        # Also check for langgraph checkpoint patterns
        before_checkpoints = set(glob.glob(".langgraph*")) | set(glob.glob("*.checkpoint*")) | set(glob.glob("checkpoints/*"))
        result = analyze_password(SYN_NORMAL)
        after_files = set(glob.glob("**/*", recursive=True))
        after_checkpoints = set(glob.glob(".langgraph*")) | set(glob.glob("*.checkpoint*")) | set(glob.glob("checkpoints/*"))
        # New files should not include checkpoint/db
        new_files = after_files - before_files
        # Allow __pycache__ and pyc files, but no checkpoint/db
        forbidden = [f for f in new_files if "checkpoint" in f.lower() or f.endswith(".db") or ".langgraph" in f]
        self.assertEqual(len(forbidden), 0, f"Graph created forbidden artifacts: {forbidden}")
        self.assertEqual(len(after_checkpoints), 0, "No checkpoint artifacts should exist")
        # Also verify result does not contain password
        self.assertNotIn(SYN_NORMAL, str(result) if len(SYN_NORMAL) >=4 else "")

if __name__ == "__main__":
    unittest.main()
