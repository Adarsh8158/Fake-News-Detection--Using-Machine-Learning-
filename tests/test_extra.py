"""
Extra tests: edge cases, label mapping, leakage-check helpers, and a small
training smoke test that only needs data/sample_news.csv (no large CSVs).
"""

import os
import sys
import unittest

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from preprocess import clean_text, extract_stylometric_features  # noqa: E402
from explain import predict_article, get_model  # noqa: E402
import leakage_check  # noqa: E402


class TestEdgeCases(unittest.TestCase):
    def test_empty_and_whitespace_input(self):
        self.assertIn("error", predict_article(""))
        self.assertIn("error", predict_article("   \n "))

    def test_non_string_input_to_clean_text(self):
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(123), "")

    def test_numbers_and_symbols_only(self):
        self.assertIn("error", predict_article("!!! 12345 ???"))

    def test_non_english_text_does_not_crash(self):
        res = predict_article("यह एक परीक्षण समाचार है")
        self.assertTrue("error" in res or "label" in res)

    def test_very_long_input(self):
        res = predict_article("The government announced a new policy today. " * 5000)
        self.assertIn(res["label"], ("REAL", "FAKE"))

    def test_stylometry_on_empty_text(self):
        style = extract_stylometric_features("")
        self.assertEqual(style["word_count"], 0)
        self.assertEqual(style["sensationalism_score"], 0.0)


class TestLabelMapping(unittest.TestCase):
    def test_probabilities_and_label_agree(self):
        res = predict_article(
            "ALERT: Miracle lemon water cures cancer overnight! Doctors are terrified of this secret!"
        )
        self.assertAlmostEqual(res["probabilities"]["Fake"] + res["probabilities"]["Real"], 100.0, places=1)
        expected = "FAKE" if res["probabilities"]["Fake"] > res["probabilities"]["Real"] else "REAL"
        self.assertEqual(res["label"], expected)
        self.assertEqual(res["is_fake"], res["label"] == "FAKE")

    def test_class_order_is_fake_then_real(self):
        # explain.py assumes classes_ == [0, 1] with 0 = Fake and 1 = Real
        self.assertEqual(list(get_model().named_steps["classifier"].classes_), [0, 1])


class TestLeakageHelpers(unittest.TestCase):
    def test_reuters_dateline_removed(self):
        raw = "WASHINGTON (Reuters) - The Senate voted on Tuesday."
        out = leakage_check.strip_artifacts(raw, 1)
        self.assertNotIn("reuters", out.lower())
        self.assertIn("Senate", out)

    def test_baseline_leaves_text_unchanged(self):
        raw = "WASHINGTON (Reuters) - The Senate voted."
        self.assertEqual(leakage_check.strip_artifacts(raw, 0), raw)

    def test_other_artifacts_removed_only_at_level_two(self):
        raw = "Some claim. Featured image via Getty Images pic.twitter.com/abc123"
        self.assertIn("Getty", leakage_check.strip_artifacts(raw, 1))
        lvl2 = leakage_check.strip_artifacts(raw, 2).lower()
        self.assertNotIn("getty", lvl2)
        self.assertNotIn("pic.twitter.com", lvl2)

    def test_prepare_returns_clean_lowercase_text(self):
        out = leakage_check.prepare("BREAKING (Reuters) - Hello, World 2020!", 1)
        self.assertEqual(out, out.lower())
        self.assertNotIn("reuters", out)


class TestTrainingSmoke(unittest.TestCase):
    def test_pipeline_fits_on_sample_csv(self):
        df = pd.read_csv(os.path.join(ROOT_DIR, "data", "sample_news.csv"))
        text = (df["title"].fillna("") + " " + df["text"].fillna("")).map(clean_text)
        pipe = leakage_check.make_pipeline()
        pipe.set_params(tfidf__min_df=1)  # the sample is tiny
        pipe.fit(text, df["label"])
        preds = pipe.predict(text)
        self.assertEqual(len(preds), len(df))
        self.assertTrue(set(preds) <= {0, 1})


if __name__ == "__main__":
    unittest.main()
