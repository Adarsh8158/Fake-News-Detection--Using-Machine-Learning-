"""
Unit and Integration Tests for Fake News Detection Pipeline
"""

import sys
import os
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from preprocess import clean_text, extract_stylometric_features
from explain import predict_article, get_model


class TestPipeline(unittest.TestCase):

    def test_clean_text_removes_punctuation_and_urls(self):
        raw = "BREAKING NEWS! Check https://example.com for more info #123."
        cleaned = clean_text(raw)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertNotIn("#", cleaned)
        self.assertNotIn("123", cleaned)
        self.assertTrue(cleaned.islower())

    def test_stylometric_signals(self):
        raw = "SHOCKING ALERT!!! Miracle Cure Discovered?!"
        style = extract_stylometric_features(raw)
        self.assertGreaterEqual(style["exclamation_count"], 3)
        self.assertGreaterEqual(style["uppercase_words"], 2)
        self.assertGreater(style["sensationalism_score"], 20.0)

    def test_model_loaded(self):
        model = get_model()
        self.assertIsNotNone(model)
        self.assertIn("tfidf", model.named_steps)
        self.assertIn("classifier", model.named_steps)

    def test_prediction_output_format(self):
        sample = "WASHINGTON (Reuters) - The Senate approved the bipartisan spending bill on Tuesday."
        res = predict_article(sample)
        self.assertIn("label", res)
        self.assertIn("confidence", res)
        self.assertIn("probabilities", res)
        self.assertIn("highlighted_html", res)
        self.assertEqual(res["label"], "REAL")


if __name__ == "__main__":
    unittest.main()
