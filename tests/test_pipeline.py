# tests/test_pipeline.py
"""
Automated Unit Tests for Offline Voice Assistant
- Text Preprocessing & Normalization
- ActionRouter Dispatch & Safe-Mode Guards
- Pipeline Loading & Direct Text Inference
- Two-Tier Confidence Threshold & OOD Handling
"""

import os
import unittest
import joblib
import numpy as np

from actions import ActionRouter
from assistant_text import normalize_text, predict_intent

class TestAssistantPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.router = ActionRouter(safe_mode=True)
        cls.model_path = os.path.join("models", "best_model.joblib")
        cls.assertTrue(os.path.exists(cls.model_path), f"Model missing at {cls.model_path}")
        cls.pipeline = joblib.load(cls.model_path)

    # --------------------------
    # 1. PREPROCESSING TESTS
    # --------------------------
    def test_normalize_text(self):
        self.assertEqual(normalize_text("  Turn ON the Lights!  "), "turn on the lights")
        self.assertEqual(normalize_text("What's the time???"), "what s the time")
        self.assertEqual(normalize_text("Volume   UP   now..."), "volume up now")
        self.assertEqual(normalize_text(""), "")

    # --------------------------
    # 2. ACTION ROUTER TESTS & SAFETY GUARDS
    # --------------------------
    def test_action_router_informational(self):
        res_time = self.router.dispatch("get_time")
        self.assertTrue(res_time["success"])
        self.assertEqual(res_time["action"], "get_time")
        self.assertIn("The current time is", res_time["reply"])

        res_date = self.router.dispatch("get_date")
        self.assertTrue(res_date["success"])
        self.assertIn("Today is", res_date["reply"])

    def test_action_router_safe_mode_destructive(self):
        """CRITICAL: Ensure shutdown and restart are strictly simulated in safe mode!"""
        res_shutdown = self.router.dispatch("shutdown_device")
        self.assertTrue(res_shutdown["success"])
        self.assertTrue(res_shutdown["metadata"].get("simulated"))
        self.assertIn("prevented in demonstration mode", res_shutdown["reply"])

        res_restart = self.router.dispatch("restart_device")
        self.assertTrue(res_restart["success"])
        self.assertTrue(res_restart["metadata"].get("simulated"))
        self.assertIn("prevented in demonstration mode", res_restart["reply"])

    def test_action_router_unknown(self):
        res_unknown = self.router.dispatch("unknown")
        self.assertFalse(res_unknown["success"])
        self.assertEqual(res_unknown["action"], "unknown")
        self.assertIn("not sure how to help", res_unknown["reply"])

    # --------------------------
    # 3. PIPELINE INFERENCE TESTS
    # --------------------------
    def test_pipeline_raw_text_inference(self):
        """Ensure pipeline accepts raw text and outputs valid classes."""
        pred = self.pipeline.predict(["turn on the light"])[0]
        self.assertEqual(pred, "turn_on_light")

        probs = self.pipeline.predict_proba(["what is the time"])[0]
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)
        self.assertGreater(len(probs), 10)

    # --------------------------
    # 4. TWO-TIER REJECTION & PREDICTION FLOW
    # --------------------------
    def test_predict_intent_flow(self):
        # 1. In-domain confident intent
        res = predict_intent("what is the time")
        self.assertEqual(res["final_intent"], "get_time")
        self.assertGreaterEqual(res["confidence"], 0.35)
        self.assertTrue(res["action_result"]["success"])

        # 2. Tier 1: Out-of-domain known unknown
        res_ood = predict_intent("what is the weather on mars")
        self.assertEqual(res_ood["final_intent"], "unknown")
        self.assertFalse(res_ood["action_result"]["success"])

        # 3. Tier 2: Low-confidence arbitrary gibberish
        res_gibberish = predict_intent("qwerty asdfgh zxcvbnm foo bar")
        self.assertEqual(res_gibberish["action_result"]["action"], "clarification_fallback")
        self.assertFalse(res_gibberish["action_result"]["success"])

if __name__ == "__main__":
    unittest.main()
