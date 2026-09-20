"""
tests/test_security_engine.py

Unit tests for core password security engine.
- Uses synthetic passwords only, no real leaked credentials
- Never prints actual passwords, no passwords in outputs
- Tests properties, not absolute scores unless explicitly defined
- Verifies privacy: returned objects do not contain raw password
"""

import unittest

from agent.feature_extractor import extract_features
from agent.pattern_analyzer import analyze_patterns
from agent.risk_engine import evaluate_risk

# Synthetic passwords created specifically for testing (not real leaked credentials)
# These are deterministic synthetic strings for test purposes only
SYN = {
    "empty": "",
    "one_char": "x",
    "very_short": "ab",
    "digits_only": "48291357",  # digits only, non-sequential
    "lowercase_only": "zxqpmkva",  # lowercase only, non-sequential, high diversity
    "uppercase_only": "ZXQPMKVA",
    "mixed_case": "AbCdEfGh",
    "seq_digits": "test123abc",  # contains 123 sequential digits
    "seq_letters": "xyzTest789",  # contains xyz sequential letters
    "repeated_chars": "aaaBBB111",  # repeated chars run 3
    "repeated_block": "abab1234",  # repeated block abab
    "long_predictable": "syntheticLetters123456",  # letters then digits, long predictable
    "long_mixed": "Tr0ub4dor&XyZ!9qLm",  # long mixed, multiple types
    "unicode": "hélloÜñîΩ≈123",  # Unicode
    "very_long": "a" * 200,  # very long (200 chars) - single type, low diversity
    "whitespace": "a b c 123",  # contains whitespace
    "punctuation": "hello!@#World123",
}

# Required feature keys
REQUIRED_FEATURES = [
    "length", "lowercase_count", "uppercase_count", "digit_count", "special_count",
    "alphabetic_count", "unique_character_count",
    "lowercase_ratio", "uppercase_ratio", "digit_ratio", "special_ratio",
    "character_diversity_ratio",
    "character_type_changes", "repeated_character_count", "maximum_repeated_run",
    "unique_character_type_count",
    "has_lowercase", "has_uppercase", "has_digits", "has_special",
    "has_mixed_case", "has_multiple_character_types",
]

ALLOWED_LEVELS = {"High Risk", "Moderate Risk", "Lower Risk", "Low Risk"}

class TestFeatureExtractor(unittest.TestCase):

    def test_returns_required_fields(self):
        for key, pw in SYN.items():
            with self.subTest(case=key):
                feats = extract_features(pw)
                for field in REQUIRED_FEATURES:
                    self.assertIn(field, feats, f"missing field {field}")

    def test_empty_no_division_by_zero(self):
        feats = extract_features(SYN["empty"])
        self.assertEqual(feats["length"], 0)
        self.assertEqual(feats["lowercase_ratio"], 0.0)
        self.assertEqual(feats["digit_ratio"], 0.0)
        self.assertEqual(feats["character_diversity_ratio"], 0.0)
        self.assertEqual(feats["maximum_repeated_run"], 0)

    def test_one_char(self):
        feats = extract_features(SYN["one_char"])
        self.assertEqual(feats["length"], 1)
        self.assertEqual(feats["maximum_repeated_run"], 1)
        self.assertIn(feats["unique_character_type_count"], [1])

    def test_very_short(self):
        feats = extract_features(SYN["very_short"])
        self.assertEqual(feats["length"], 2)
        self.assertGreaterEqual(feats["character_diversity_ratio"], 0)

    def test_digits_only(self):
        feats = extract_features(SYN["digits_only"])
        self.assertTrue(feats["has_digits"])
        self.assertFalse(feats["has_lowercase"])
        self.assertEqual(feats["special_count"], 0)

    def test_lowercase_only(self):
        feats = extract_features(SYN["lowercase_only"])
        self.assertTrue(feats["has_lowercase"])
        self.assertFalse(feats["has_uppercase"])
        self.assertGreater(feats["lowercase_count"], 0)

    def test_uppercase_only(self):
        feats = extract_features(SYN["uppercase_only"])
        self.assertTrue(feats["has_uppercase"])
        self.assertFalse(feats["has_lowercase"])

    def test_mixed_case(self):
        feats = extract_features(SYN["mixed_case"])
        self.assertTrue(feats["has_mixed_case"])
        self.assertTrue(feats["has_lowercase"] and feats["has_uppercase"])

    def test_unicode_handling(self):
        feats = extract_features(SYN["unicode"])
        # Should not crash, should handle Unicode
        self.assertGreater(feats["length"], 0)
        self.assertIsInstance(feats["lowercase_count"], int)

    def test_very_long(self):
        feats = extract_features(SYN["very_long"])
        self.assertEqual(feats["length"], 200)
        self.assertEqual(feats["maximum_repeated_run"], 200)
        self.assertEqual(feats["unique_character_count"], 1)

    def test_whitespace(self):
        feats = extract_features(SYN["whitespace"])
        # whitespace counted as special (non-alnum)
        self.assertGreater(feats["special_count"], 0)

    def test_ratios_consistency(self):
        for pw in SYN.values():
            feats = extract_features(pw)
            n = feats["length"]
            if n > 0:
                # ratios should sum roughly to 1.0 (lower+upper+digit+special = n)
                total_ratio = feats["lowercase_ratio"] + feats["uppercase_ratio"] + feats["digit_ratio"] + feats["special_ratio"]
                self.assertAlmostEqual(total_ratio, 1.0, places=5)

    def test_no_password_in_output(self):
        for key, pw in SYN.items():
            if pw == "":
                continue
            feats = extract_features(pw)
            # Ensure no value equals the password exactly (no storage of raw password)
            for v in feats.values():
                if isinstance(v, str):
                    self.assertNotEqual(v, pw)
            # Also ensure password not leaked as a whole value in string representation for longer passwords
            if len(pw) >= 4:
                self.assertNotIn(pw, str(feats.values()))

class TestPatternAnalyzer(unittest.TestCase):

    def test_sequential_digits_detected(self):
        feats = extract_features(SYN["seq_digits"])
        analysis = analyze_patterns(SYN["seq_digits"], feats)
        codes = [i["code"] for i in analysis["issues"]]
        self.assertIn("sequential_digits", codes)

    def test_sequential_letters_detected(self):
        feats = extract_features(SYN["seq_letters"])
        analysis = analyze_patterns(SYN["seq_letters"], feats)
        codes = [i["code"] for i in analysis["issues"]]
        self.assertIn("sequential_letters", codes)

    def test_repeated_chars_detected(self):
        feats = extract_features(SYN["repeated_chars"])
        analysis = analyze_patterns(SYN["repeated_chars"], feats)
        all_codes = [i["code"] for i in analysis["issues"] + analysis["warnings"]]
        self.assertIn("repeated_characters", all_codes)

    def test_repeated_block_detected(self):
        feats = extract_features(SYN["repeated_block"])
        analysis = analyze_patterns(SYN["repeated_block"], feats)
        codes = [i["code"] for i in analysis["issues"]]
        self.assertIn("repeated_block", codes)

    def test_single_type_detection(self):
        feats = extract_features(SYN["digits_only"])
        analysis = analyze_patterns(SYN["digits_only"], feats)
        all_codes = [it["code"] for it in analysis["warnings"] + analysis["issues"]]
        # digits only should trigger single_type
        self.assertTrue(any(c.startswith("single_type") for c in all_codes))

    def test_predictable_structure(self):
        feats = extract_features(SYN["long_predictable"])
        analysis = analyze_patterns(SYN["long_predictable"], feats)
        codes = [it["code"] for it in analysis["warnings"] + analysis["issues"]]
        self.assertIn("predictable_structure", codes)

    def test_no_crash_on_empty(self):
        feats = extract_features(SYN["empty"])
        analysis = analyze_patterns(SYN["empty"], feats)
        self.assertIn("issues", analysis)
        self.assertIn("warnings", analysis)
        self.assertIn("positive_signals", analysis)

    def test_no_password_in_analysis(self):
        for pw in SYN.values():
            if pw == "":
                continue
            feats = extract_features(pw)
            analysis = analyze_patterns(pw, feats)
            # Check that password not in returned structure
            blob = str(analysis)
            self.assertNotIn(pw, blob, "password leaked in analysis output")

    def test_finding_format(self):
        analysis = analyze_patterns(SYN["seq_digits"], extract_features(SYN["seq_digits"]))
        for bucket in ["issues", "warnings", "positive_signals"]:
            self.assertIn(bucket, analysis)
            for item in analysis[bucket]:
                self.assertIn("code", item)
                self.assertIn("severity", item)
                self.assertIn("message", item)
                self.assertIn(item["severity"], ["high", "medium", "low"])

class TestRiskEngine(unittest.TestCase):

    def test_score_range(self):
        for pw in SYN.values():
            feats = extract_features(pw)
            analysis = analyze_patterns(pw, feats)
            result = evaluate_risk(feats, analysis)
            self.assertGreaterEqual(result["heuristic_score"], 0)
            self.assertLessEqual(result["heuristic_score"], 100)

    def test_level_valid(self):
        for pw in SYN.values():
            feats = extract_features(pw)
            analysis = analyze_patterns(pw, feats)
            result = evaluate_risk(feats, analysis)
            self.assertIn(result["risk_level"], ALLOWED_LEVELS)

    def test_deterministic(self):
        pw = SYN["long_mixed"]
        feats = extract_features(pw)
        analysis = analyze_patterns(pw, feats)
        r1 = evaluate_risk(feats, analysis)
        r2 = evaluate_risk(feats, analysis)
        self.assertEqual(r1["heuristic_score"], r2["heuristic_score"])
        self.assertEqual(r1["risk_level"], r2["risk_level"])

    def test_password_not_in_result(self):
        for key, pw in SYN.items():
            if pw == "" or len(pw) < 4:
                # Skip very short passwords where substring check causes false positives
                # Privacy is ensured by exact value check instead
                continue
            feats = extract_features(pw)
            analysis = analyze_patterns(pw, feats)
            result = evaluate_risk(feats, analysis)
            blob = str(result)
            self.assertNotIn(pw, blob, "password leaked in risk result")
        # Additional exact-value check for all including short
        for pw in SYN.values():
            if pw == "":
                continue
            feats = extract_features(pw)
            analysis = analyze_patterns(pw, feats)
            result = evaluate_risk(feats, analysis)
            # Ensure no field exactly equals the password
            for bucket in ["issues", "warnings", "positive_signals"]:
                for item in result[bucket]:
                    self.assertNotEqual(item.get("message"), pw)
                    self.assertNotEqual(item.get("code"), pw)

    def test_empty_does_not_crash(self):
        feats = extract_features(SYN["empty"])
        analysis = analyze_patterns(SYN["empty"], feats)
        result = evaluate_risk(feats, analysis)
        self.assertEqual(result["heuristic_score"] >= 0, True)
        self.assertIn(result["risk_level"], ALLOWED_LEVELS)

    def test_ml_optional(self):
        feats = extract_features(SYN["long_mixed"])
        analysis = analyze_patterns(SYN["long_mixed"], feats)
        # Without ML
        r_no_ml = evaluate_risk(feats, analysis)
        self.assertIsNone(r_no_ml["ml_evidence"])
        # With ML
        ml = {"predicted_class": 2, "confidence": 0.99}
        r_ml = evaluate_risk(feats, analysis, ml_result=ml)
        self.assertIsNotNone(r_ml["ml_evidence"])
        self.assertEqual(r_ml["ml_evidence"]["predicted_class"], 2)
        # ML must not automatically set score to high; score should be same heuristic
        self.assertEqual(r_no_ml["heuristic_score"], r_ml["heuristic_score"])

    def test_long_predictable_not_automatically_secure(self):
        # Long predictable should not be automatically Low Risk despite length
        feats = extract_features(SYN["long_predictable"])
        analysis = analyze_patterns(SYN["long_predictable"], feats)
        result = evaluate_risk(feats, analysis)
        # It has sequential and predictable structure, so should not be 80+ Low Risk blindly
        # We test that it is not automatically considered Low Risk just due to length
        # The engine should penalize predictable patterns
        # Note: we don't assert exact score, but check that issues exist and score not maximal
        self.assertLess(result["heuristic_score"], 90)

    def test_long_mixed_has_positive(self):
        feats = extract_features(SYN["long_mixed"])
        analysis = analyze_patterns(SYN["long_mixed"], feats)
        result = evaluate_risk(feats, analysis)
        self.assertGreater(len(result["positive_signals"]) + len(result["explanations"]), 0)

    def test_unicode_risk(self):
        feats = extract_features(SYN["unicode"])
        analysis = analyze_patterns(SYN["unicode"], feats)
        result = evaluate_risk(feats, analysis)
        self.assertIn(result["risk_level"], ALLOWED_LEVELS)

    def test_very_long_risk(self):
        feats = extract_features(SYN["very_long"])
        analysis = analyze_patterns(SYN["very_long"], feats)
        result = evaluate_risk(feats, analysis)
        # Very long but single type, low diversity, repeated -> should not be Low Risk blindly
        self.assertLess(result["heuristic_score"], 80)

    def test_scoring_breakdown_exists(self):
        feats = extract_features(SYN["long_mixed"])
        analysis = analyze_patterns(SYN["long_mixed"], feats)
        result = evaluate_risk(feats, analysis)
        self.assertIn("scoring_breakdown", result)
        self.assertIn("heuristic_score", result)
        self.assertIn("note", result)

    def test_no_exception_leak(self):
        try:
            extract_features(123)  # type error should not include password
        except TypeError as e:
            self.assertNotIn("123", str(e))

class TestPrivacy(unittest.TestCase):

    def test_returns_do_not_contain_password(self):
        # Verify that none of the three layers leak the password
        test_pw = "syntheticPrivacyCheck123!@#"
        feats = extract_features(test_pw)
        analysis = analyze_patterns(test_pw, feats)
        risk = evaluate_risk(feats, analysis, ml_result={"predicted_class": 1, "confidence": 0.5})
        for obj in [feats, analysis, risk]:
            blob = str(obj)
            self.assertNotIn(test_pw, blob, "password leaked in output object")

if __name__ == "__main__":
    unittest.main()
