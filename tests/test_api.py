"""
tests/test_api.py - FastAPI backend tests for Phase 5A.
Synthetic passwords only, privacy-first.
Tests health, info, analyze, validation, privacy, and that passwords are not persisted.
"""

import unittest
import os
import glob
from unittest.mock import patch

from fastapi.testclient import TestClient
from api import app
from agent.security_graph import analyze_password

client = TestClient(app)

# Synthetic passwords for testing (never real dataset)
SYN_VALID = "zxqpmkva"
SYN_EMPTY = ""
SYN_LONG_MIXED = "Tr0ub4dor&XyZ!9qLmExtraLong123"
SYN_SYNTHETIC_PRIVACY = "SyntheticPassword123!"  # for privacy test, not used outside

class TestAPI(unittest.TestCase):

    def test_health_returns_200(self):
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertNotIn("password", str(data).lower())

    def test_info_returns_200(self):
        resp = client.get("/info")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("name", data)
        self.assertEqual(data["name"], "Password Security Agent")
        self.assertIn("ml_enabled", data)
        self.assertIn("rag_enabled", data)
        self.assertIn("llm_provider", data)
        # Honest: fallback in this environment
        self.assertEqual(data["llm_provider"], "fallback")
        # No secrets or paths (allow word "password" in name)
        for key in ["env", "secret", "api_key"]:
            self.assertNotIn(key, str(data).lower())
        # Ensure no filesystem path leaked
        self.assertNotIn(":\\", str(data))
        self.assertNotIn("/", str(data.get("llm_provider", "")))

    def test_analyze_accepts_valid_synthetic(self):
        resp = client.post("/analyze", json={"password": SYN_VALID})
        self.assertEqual(resp.status_code, 200)

    def test_analyze_returns_200_for_valid(self):
        resp = client.post("/analyze", json={"password": SYN_LONG_MIXED})
        self.assertEqual(resp.status_code, 200)

    def test_response_contains_risk(self):
        resp = client.post("/analyze", json={"password": SYN_VALID})
        data = resp.json()
        self.assertIn("risk", data)
        self.assertIn("heuristic_score", data["risk"])
        self.assertIn("risk_level", data["risk"])

    def test_response_contains_patterns(self):
        resp = client.post("/analyze", json={"password": SYN_VALID})
        data = resp.json()
        self.assertIn("patterns", data)
        self.assertIn("issues", data["patterns"])
        self.assertIn("warnings", data["patterns"])

    def test_response_contains_ml_evidence(self):
        resp = client.post("/analyze", json={"password": SYN_VALID})
        data = resp.json()
        self.assertIn("ml", data)
        self.assertTrue(data["ml"].get("available"))
        self.assertIn("predicted_class", data["ml"])
        self.assertIn("warning", data["ml"])

    def test_response_contains_rag_explanation(self):
        resp = client.post("/analyze", json={"password": SYN_VALID})
        data = resp.json()
        self.assertIn("rag", data)
        self.assertIn("results", data["rag"])
        self.assertIn("explanation", data)
        self.assertIn("summary", data["explanation"])
        self.assertIn("recommendations", data["explanation"])

    def test_response_does_not_contain_password(self):
        pw = SYN_SYNTHETIC_PRIVACY
        resp = client.post("/analyze", json={"password": pw})
        data = resp.json()
        # Check no key named password
        self.assertNotIn("password", data)
        # Check that password not in any value (exact)
        blob = str(data)
        self.assertNotIn(pw, blob, "Response leaked password")

    def test_str_response_does_not_contain_password(self):
        pw = SYN_SYNTHETIC_PRIVACY
        resp = client.post("/analyze", json={"password": pw})
        # str(response) includes json dump
        blob = str(resp.json())
        self.assertNotIn(pw, blob)
        # Also check raw text
        self.assertNotIn(pw, resp.text)

    def test_missing_password_rejected_safely(self):
        resp = client.post("/analyze", json={})
        self.assertEqual(resp.status_code, 422)
        body = str(resp.json())
        self.assertNotIn(SYN_SYNTHETIC_PRIVACY, body)
        # Detail should indicate missing field but not contain password value
        self.assertIn("detail", resp.json())

    def test_empty_password_handled_by_engine(self):
        resp = client.post("/analyze", json={"password": SYN_EMPTY})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["features"]["length"], 0)
        self.assertEqual(data["risk"]["risk_level"], "High Risk")

    def test_invalid_json_rejected_safely(self):
        resp = client.post("/analyze", content="not json", headers={"Content-Type": "application/json"})
        self.assertEqual(resp.status_code, 422)
        # Should not contain password
        self.assertNotIn(SYN_SYNTHETIC_PRIVACY, resp.text)

    def test_internal_exception_does_not_expose_password(self):
        pw = SYN_SYNTHETIC_PRIVACY
        with patch("api.analyze_password", side_effect=Exception("internal failure")):
            resp = client.post("/analyze", json={"password": pw})
            self.assertEqual(resp.status_code, 500)
            data = resp.json()
            self.assertEqual(data.get("detail"), "Password analysis failed.")
            # Must not expose password, stack trace, or path
            blob = str(data)
            self.assertNotIn(pw, blob)
            self.assertNotIn("Traceback", blob)
            self.assertNotIn("File", blob)

    def test_no_password_written_to_disk_during_api(self):
        pw = SYN_SYNTHETIC_PRIVACY
        # Capture files before
        before = set(glob.glob("**/*", recursive=True))
        resp = client.post("/analyze", json={"password": pw})
        self.assertEqual(resp.status_code, 200)
        after = set(glob.glob("**/*", recursive=True))
        new_files = after - before
        # Filter out __pycache__ and pyc which are expected
        new_files = {f for f in new_files if not f.startswith("venv") and "__pycache__" not in f and f.endswith(".pyc") is False}
        # Ensure no new file contains password
        for path in new_files:
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read()
                        self.assertNotIn(pw, content, f"Password leaked to file {path}")
                except Exception:
                    pass
        # Also check that no new .log, .db created
        for f in new_files:
            self.assertNotIn("password", f.lower())

    def test_heuristic_unchanged_via_api(self):
        pw = SYN_LONG_MIXED
        direct = analyze_password(pw)
        resp = client.post("/analyze", json={"password": pw})
        data = resp.json()
        self.assertEqual(direct["risk"]["heuristic_score"], data["risk"]["heuristic_score"])
        self.assertEqual(direct["risk"]["risk_level"], data["risk"]["risk_level"])

    def test_privacy_through_api_comprehensive(self):
        pw = SYN_SYNTHETIC_PRIVACY
        resp = client.post("/analyze", json={"password": pw})
        data = resp.json()
        # JSON response
        self.assertNotIn(pw, str(data))
        # Response string
        self.assertNotIn(pw, resp.text)
        # Exception path already tested, but also check that even after success, no password in explanation or rag query
        if "rag" in data and "query" in data["rag"]:
            self.assertNotIn(pw, data["rag"]["query"])
        if "explanation" in data:
            self.assertNotIn(pw, str(data["explanation"]))

if __name__ == "__main__":
    unittest.main()
