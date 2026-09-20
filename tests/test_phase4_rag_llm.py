"""
tests/test_phase4_rag_llm.py - Phase 4 RAG + Llama integration tests.
Synthetic passwords only, no real dataset, privacy-first.
"""

import unittest
import os
import glob

from agent.rag import retrieve_knowledge, knowledge_base_available, get_knowledge_stats, _build_retrieval_query
from agent.llm_explainer import generate_explanation, is_llama_available, get_llama_runtime
from agent.security_graph import analyze_password
from agent.feature_extractor import extract_features
from agent.pattern_analyzer import analyze_patterns
from agent.risk_engine import evaluate_risk
from agent.ml_analyzer import analyze_with_ml

SYN_NORMAL = "zxqpmkva"
SYN_LONG_PREDICTABLE = "syntheticLetters123456"
SYN_LONG_MIXED = "Tr0ub4dor&XyZ!9qLmExtraLong123"
SYN_EMPTY = ""
SYN_ONE = "x"

class TestPhase4(unittest.TestCase):

    def test_knowledge_base_loads(self):
        self.assertTrue(knowledge_base_available())
        stats = get_knowledge_stats()
        self.assertGreater(stats["num_chunks"], 0)
        self.assertGreater(stats["num_docs"], 0)

    def test_documents_can_be_retrieved(self):
        findings = {
            "risk_level": "Moderate Risk",
            "heuristic_score": 58,
            "issues": [{"code": "sequential_digits", "severity": "medium", "message": "Contains sequential digits"}],
            "warnings": [],
            "positive_signals": [],
            "features": {"length": 10, "unique_character_type_count": 2},
            "ml": {"available": True, "predicted_class": 1},
        }
        result = retrieve_knowledge(findings, top_k=3)
        self.assertTrue(result["retrieval_available"])
        self.assertGreater(len(result["results"]), 0)
        for r in result["results"]:
            self.assertIn("content", r)
            self.assertIn("source", r)
            self.assertIn("score", r)

    def test_retrieval_query_no_password(self):
        pw = SYN_LONG_MIXED
        findings = {
            "risk_level": "Lower Risk",
            "heuristic_score": 75,
            "issues": [{"code": "sequential_digits", "message": "seq"}],
            "warnings": [],
            "positive_signals": [],
            "features": extract_features(pw),
            "ml": analyze_with_ml(pw),
        }
        result = retrieve_knowledge(findings)
        query = result["query"]
        if len(pw) >= 4:
            self.assertNotIn(pw, query, "RAG query must not contain raw password")
        # Also test helper
        from agent.rag import _query_contains_password
        self.assertFalse(_query_contains_password(query, pw))

    def test_rag_returns_relevant_knowledge(self):
        # For sequential digits, should retrieve common_patterns doc
        findings = {
            "risk_level": "Moderate Risk",
            "heuristic_score": 58,
            "issues": [{"code": "sequential_digits", "severity": "medium", "message": "Contains sequential digits"}],
            "warnings": [],
            "positive_signals": [],
            "features": {"length": 9, "unique_character_type_count": 2},
            "ml": {},
        }
        result = retrieve_knowledge(findings, top_k=3)
        self.assertGreater(len(result["results"]), 0)
        # At least one result should mention sequential or pattern
        contents = " ".join([r["content"].lower() for r in result["results"]])
        self.assertTrue("sequential" in contents or "pattern" in contents or "digit" in contents)

    def test_llm_input_no_password(self):
        pw = SYN_LONG_MIXED
        feats = extract_features(pw)
        patterns = analyze_patterns(pw, feats)
        risk = evaluate_risk(feats, patterns)
        # Build findings as llm would receive
        findings = {
            "risk_level": risk["risk_level"],
            "heuristic_score": risk["heuristic_score"],
            "issues": risk["issues"],
            "warnings": risk["warnings"],
            "positive_signals": risk["positive_signals"],
            "relevant_knowledge": retrieve_knowledge({"risk_level": risk["risk_level"], "issues": risk["issues"], "warnings": risk["warnings"], "features": feats})["results"],
            "ml": analyze_with_ml(pw),
        }
        # Ensure findings do not contain password
        import json
        blob = str(findings)
        if len(pw) >= 4:
            self.assertNotIn(pw, blob)
        # Generate explanation and ensure its input prompt not containing password
        expl = generate_explanation(findings)
        self.assertNotIn(pw, str(expl) if len(pw) >= 4 else "")

    def test_llm_cannot_change_heuristic_score(self):
        pw = SYN_LONG_PREDICTABLE
        result = analyze_password(pw)
        feats = extract_features(pw)
        patterns = analyze_patterns(pw, feats)
        ml = analyze_with_ml(pw)
        direct = evaluate_risk(feats, patterns, ml_result={"predicted_class": ml.get("predicted_class"), "confidence": ml.get("confidence")} if ml.get("available") else None)
        self.assertEqual(result["risk"]["heuristic_score"], direct["heuristic_score"])
        # Also ensure explanation does not claim different score
        expl = result["explanation"]
        # Summary should mention same score
        self.assertIn(str(direct["heuristic_score"]), expl["summary"])

    def test_llm_cannot_change_risk_level(self):
        pw = SYN_LONG_PREDICTABLE
        result = analyze_password(pw)
        feats = extract_features(pw)
        patterns = analyze_patterns(pw, feats)
        direct = evaluate_risk(feats, patterns, ml_result=None)
        self.assertEqual(result["risk"]["risk_level"], direct["risk_level"])
        # ML predicts 2 but heuristic is not Low Risk
        self.assertEqual(result["ml"]["predicted_class"], 2)
        self.assertNotEqual(result["risk"]["risk_level"], "Low Risk")

    def test_explanation_structure(self):
        result = analyze_password(SYN_NORMAL)
        expl = result["explanation"]
        self.assertIn("explanation_available", expl)
        self.assertTrue(expl["explanation_available"])
        self.assertIn("summary", expl)
        self.assertIn("why", expl)
        self.assertIsInstance(expl["why"], list)
        self.assertIn("recommendations", expl)
        self.assertIsInstance(expl["recommendations"], list)
        self.assertIn("sources", expl)
        self.assertIsInstance(expl["sources"], list)
        self.assertIn("provider", expl)

    def test_missing_llama_handled_cleanly(self):
        # is_llama_available should return bool without exception
        avail = is_llama_available()
        runtime = get_llama_runtime()
        self.assertIsInstance(avail, bool)
        self.assertIsInstance(runtime, str)
        # Even if not available, explanation should still be available via fallback
        result = analyze_password(SYN_NORMAL)
        self.assertTrue(result["explanation"]["explanation_available"])
        self.assertIn(result["explanation"]["provider"], ["fallback", "ollama", "llama_cpp", "ollama (fallback used, model not loaded)", "llama_cpp (fallback used, model not loaded)"])

    def test_fallback_deterministic(self):
        pw = SYN_NORMAL
        r1 = analyze_password(pw)
        r2 = analyze_password(pw)
        e1 = r1["explanation"]
        e2 = r2["explanation"]
        self.assertEqual(e1["summary"], e2["summary"])
        self.assertEqual(e1["why"], e2["why"])
        self.assertEqual(e1["recommendations"], e2["recommendations"])
        self.assertEqual(e1["provider"], e2["provider"])

    def test_full_langgraph_pipeline_works(self):
        for pw in [SYN_EMPTY, SYN_ONE, SYN_NORMAL, SYN_LONG_PREDICTABLE, SYN_LONG_MIXED]:
            result = analyze_password(pw)
            self.assertIn("risk", result)
            self.assertIn("features", result)
            self.assertIn("patterns", result)
            self.assertIn("ml", result)
            self.assertIn("rag", result)
            self.assertIn("explanation", result)
            self.assertIn("heuristic_score", result["risk"])
            self.assertIn("risk_level", result["risk"])
            # Verify rag knowledge present
            self.assertIn("results", result["rag"])
            # Verify explanation present
            self.assertTrue(result["explanation"]["explanation_available"])

    def test_no_password_written_to_disk(self):
        # Ensure knowledge base and results don't contain synthetic passwords
        for pw in [SYN_NORMAL, SYN_LONG_MIXED]:
            # Run graph to ensure no file written
            analyze_password(pw)
            for root, _, files in os.walk("knowledge"):
                for name in files:
                    path = os.path.join(root, name)
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if len(pw) >= 4:
                            self.assertNotIn(pw, content)
            for root, _, files in os.walk("results"):
                for name in files:
                    path = os.path.join(root, name)
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if len(pw) >= 4:
                            self.assertNotIn(pw, content)
            # Check vector store not on disk (we use in-memory)
            self.assertFalse(os.path.exists("vector_store.db"))
            self.assertFalse(os.path.exists("chroma.sqlite3"))

    def test_no_password_in_logs_errors_results(self):
        pw = SYN_LONG_MIXED
        result = analyze_password(pw)
        blob = str(result)
        if len(pw) >= 4:
            self.assertNotIn(pw, blob)
        # Check rag query
        self.assertNotIn(pw, result["rag"]["query"] if len(pw) >=4 else "")
        # Check explanation
        expl_blob = str(result["explanation"])
        if len(pw) >= 4:
            self.assertNotIn(pw, expl_blob)

    def test_no_persistent_checkpoint_stores_password(self):
        pw = SYN_LONG_MIXED
        result = analyze_password(pw)
        # Check no checkpoint files created in project root
        checkpoints = glob.glob("**/*checkpoint*", recursive=True)
        # Filter to only project root non-venv (venv has checkpoint module, ignore)
        project_checkpoints = [c for c in checkpoints if not c.startswith("venv")]
        self.assertEqual(len(project_checkpoints), 0, f"Found checkpoint artifacts: {project_checkpoints}")
        # Check no .db files in project root
        dbs = [p for p in glob.glob("**/*.db", recursive=True) if not p.startswith("venv")]
        self.assertEqual(len(dbs), 0)
        # Check result does not contain password in state
        self.assertNotIn(pw, str(result) if len(pw) >=4 else "")

if __name__ == "__main__":
    unittest.main()
