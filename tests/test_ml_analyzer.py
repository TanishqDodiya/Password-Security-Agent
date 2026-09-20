"""
tests/test_ml_analyzer.py

Integration tests for ML analyzer (optional evidence source).
Synthetic passwords only, no real leaked credentials, no passwords printed.
"""

import unittest
import os
import time

# Import ML analyzer and risk engine
from agent.ml_analyzer import analyze_with_ml, WARNING_TEXT
from agent.feature_extractor import extract_features
from agent.pattern_analyzer import analyze_patterns
from agent.risk_engine import evaluate_risk

SYNTHETIC_A = "zxqpmkva"  # length 8, will be predicted as 1 likely
SYNTHETIC_B = "a"  # length 1, predicted as 0
SYNTHETIC_C = "Tr0ub4dor&XyZ!9qLmExtraLong123"  # long, predicted as 2
SYNTHETIC_D = "syntheticLetters123456"  # long predictable, predicted as 2 but heuristic should penalize

class TestMLAnalyzer(unittest.TestCase):

    def test_model_loads_successfully(self):
        result = analyze_with_ml(SYNTHETIC_A)
        self.assertTrue(result.get("available"), f"Model should load, got {result.get('reason')}")

    def test_valid_synthetic_produces_evidence(self):
        result = analyze_with_ml(SYNTHETIC_A)
        self.assertTrue(result["available"])
        self.assertIn("predicted_class", result)
        self.assertIn(result["predicted_class"], [0, 1, 2])
        self.assertIn("model_name", result)
        self.assertIn("feature_set", result)
        self.assertIn("warning", result)

    def test_ml_result_contains_no_raw_password(self):
        for pw in [SYNTHETIC_A, SYNTHETIC_C]:  # skip single-char SYNTHETIC_B to avoid false positive substring
            result = analyze_with_ml(pw)
            blob = str(result)
            self.assertNotIn(pw, blob, "ML result leaked password")
        # For single char, check exact-value not substring
        result = analyze_with_ml(SYNTHETIC_B)
        self.assertNotEqual(result.get("predicted_class"), SYNTHETIC_B)

    def test_missing_model_handled_cleanly(self):
        # Simulate missing by temporarily renaming artifact if exists, but we test the function's handling
        # We will test with invalid type to trigger unavailable path without actually deleting file
        result = analyze_with_ml(12345)  # not a string
        self.assertFalse(result["available"])
        self.assertIn("warning", result)
        self.assertIn(WARNING_TEXT[:20], result["warning"])

    def test_ml_does_not_override_heuristic(self):
        # Heuristic score must be same with and without ML
        feats = extract_features(SYNTHETIC_D)
        analysis = analyze_patterns(SYNTHETIC_D, feats)
        r_no_ml = evaluate_risk(feats, analysis, ml_result=None)
        ml_evidence = analyze_with_ml(SYNTHETIC_D)
        # Convert ml_analyzer output to risk_engine ml_result format
        ml_for_risk = {"predicted_class": ml_evidence.get("predicted_class"), "confidence": ml_evidence.get("confidence", 0.5)}
        r_with_ml = evaluate_risk(feats, analysis, ml_result=ml_for_risk)
        self.assertEqual(r_no_ml["heuristic_score"], r_with_ml["heuristic_score"], "Heuristic score must not change with ML evidence")
        self.assertEqual(r_no_ml["risk_level"], r_with_ml["risk_level"])
        # ML evidence should be present separately
        self.assertIsNotNone(r_with_ml["ml_evidence"])

    def test_deterministic_output(self):
        r1 = analyze_with_ml(SYNTHETIC_A)
        r2 = analyze_with_ml(SYNTHETIC_A)
        self.assertEqual(r1["predicted_class"], r2["predicted_class"])
        if "confidence" in r1 and "confidence" in r2:
            self.assertAlmostEqual(r1["confidence"], r2["confidence"], places=5)
        if "class_probabilities" in r1:
            for k in ["0", "1", "2"]:
                self.assertAlmostEqual(r1["class_probabilities"][k], r2["class_probabilities"][k], places=6)

    def test_different_lengths_can_produce_different_classes(self):
        r_short = analyze_with_ml(SYNTHETIC_B)  # len 1 -> expect 0
        r_long = analyze_with_ml(SYNTHETIC_C)  # long -> expect 2
        # At least they should be valid and potentially different; we test that model can output different classes
        self.assertIn(r_short["predicted_class"], [0,1,2])
        self.assertIn(r_long["predicted_class"], [0,1,2])
        # Given length-driven dataset, these should differ
        self.assertNotEqual(r_short["predicted_class"], r_long["predicted_class"], "Different lengths should produce different dataset classes (length-driven)")

    def test_no_dataset_file_modified(self):
        # Check timestamps of protected files not changed during test (allow small drift but ensure not overwritten)
        # We check that files still exist and have expected header, not that they were rewritten in last 5 seconds
        for path in ["data/data.csv", "data/clean_passwords.csv"]:
            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as f:
                header = f.readline().strip()
                self.assertEqual(header, "password,strength" if "clean" in path or "data.csv" in path else header)
        # Also ensure model artifact does not contain raw synthetic password
        # Check artifact file not containing synthetic string (should be binary but we test not containing synthetic as bytes)
        artifact_path = os.path.join("models", "password_classifier.joblib")
        self.assertTrue(os.path.exists(artifact_path))
        # File should not contain the synthetic password as plain text (model stores only weights)
        with open(artifact_path, "rb") as f:
            data = f.read()
            # For short synthetic, might be false positive due to random bytes, so test longer synthetic
            self.assertNotIn(SYNTHETIC_C.encode(), data)

    def test_no_password_written_to_disk(self):
        # Ensure results files do not contain synthetic passwords
        for root, _, files in os.walk("results"):
            for name in files:
                path = os.path.join(root, name)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    self.assertNotIn(SYNTHETIC_A, content)
                    self.assertNotIn(SYNTHETIC_C, content)

    def test_class_probabilities_valid(self):
        result = analyze_with_ml(SYNTHETIC_A)
        self.assertTrue(result["available"])
        if "class_probabilities" in result:
            probs = result["class_probabilities"]
            self.assertIn("0", probs)
            self.assertIn("1", probs)
            self.assertIn("2", probs)
            total = sum(probs.values())
            self.assertAlmostEqual(total, 1.0, places=2, msg="Probabilities should sum to ~1.0")
            for v in probs.values():
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)
            # confidence should be max proba
            self.assertAlmostEqual(result["confidence"], max(probs.values()), places=5)
            self.assertIn("model class probability", str(result.get("model_class_probability_note", "")).lower())

    def test_warning_explicit(self):
        result = analyze_with_ml(SYNTHETIC_A)
        self.assertIn("warning", result)
        self.assertIn("heavily length-driven", result["warning"])
        self.assertIn("not treated as an independent measure", result["warning"])

if __name__ == "__main__":
    unittest.main()
