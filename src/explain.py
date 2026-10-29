"""
Inference & Explainability Engine for Fake News Detection
Provides real-time model inference, feature attribution, and text highlighting.
"""

import os
import re
import html
import joblib
import numpy as np

try:
    from .preprocess import clean_text, extract_stylometric_features
except ImportError:
    from preprocess import clean_text, extract_stylometric_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fake_news_model.pkl")

_pipeline = None


def get_model():
    """Lazy loader for trained pipeline artifact."""
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def explain_news(pipeline, text: str, top_k: int = 10) -> dict:
    """
    Extracts the highest-impact words from the input text that swayed
    the Logistic Regression classifier toward Fake (Class 0) or Real (Class 1).
    """
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["classifier"]

    vec = tfidf.transform([text])
    doc_weights = vec.toarray()[0]
    active_indices = np.where(doc_weights > 0)[0]

    if len(active_indices) == 0:
        return {"fake_cues": [], "real_cues": []}

    feature_names = np.array(tfidf.get_feature_names_out())
    active_words = feature_names[active_indices]
    coefficients = clf.coef_[0][active_indices]

    # Impact score = TF-IDF weight * Logistic Regression coefficient
    impact_scores = doc_weights[active_indices] * coefficients

    fake_mask = impact_scores < 0
    real_mask = impact_scores > 0

    fake_items = []
    if np.any(fake_mask):
        fake_words = active_words[fake_mask]
        fake_vals = np.abs(impact_scores[fake_mask])
        sorted_fake = np.argsort(fake_vals)[::-1]
        for idx in sorted_fake[:top_k]:
            fake_items.append({
                "word": str(fake_words[idx]),
                "weight": round(float(fake_vals[idx]), 3)
            })

    real_items = []
    if np.any(real_mask):
        real_words = active_words[real_mask]
        real_vals = impact_scores[real_mask]
        sorted_real = np.argsort(real_vals)[::-1]
        for idx in sorted_real[:top_k]:
            real_items.append({
                "word": str(real_words[idx]),
                "weight": round(float(real_vals[idx]), 3)
            })

    return {
        "fake_cues": fake_items,
        "real_cues": real_items
    }


def generate_highlighted_html(raw_text: str, fake_words: list, real_words: list) -> str:
    """
    Generates a professional, theme-adaptive text highlight block.
    Uses refined translucency and semantic colors that look clean on dark & light modes.
    """
    if not raw_text:
        return ""

    escaped_text = html.escape(raw_text)

    # Sort tokens by length descending so longer n-grams match first
    fake_set = sorted([item["word"] for item in fake_words], key=len, reverse=True)
    real_set = sorted([item["word"] for item in real_words], key=len, reverse=True)

    for word in fake_set:
        pattern = re.compile(rf"\b({re.escape(word)})\b", re.IGNORECASE)
        escaped_text = pattern.sub(
            r'<mark style="background-color: rgba(239, 68, 68, 0.18); color: #ef4444; padding: 2px 6px; border-radius: 4px; font-weight: 600; border-bottom: 2px solid #ef4444;" title="Suspicious Indicator">\1</mark>',
            escaped_text
        )

    for word in real_set:
        pattern = re.compile(rf"\b({re.escape(word)})\b", re.IGNORECASE)
        escaped_text = pattern.sub(
            r'<mark style="background-color: rgba(34, 197, 94, 0.18); color: #10b981; padding: 2px 6px; border-radius: 4px; font-weight: 600; border-bottom: 2px solid #10b981;" title="Credible Indicator">\1</mark>',
            escaped_text
        )

    return (
        f'<div style="line-height: 1.85; font-size: 1.02rem; padding: 1.25rem; '
        f'background: rgba(128, 128, 128, 0.05); border: 1px solid rgba(128, 128, 128, 0.15); '
        f'border-radius: 10px; font-family: inherit;">'
        f'{escaped_text}'
        f'</div>'
    )


def predict_article(raw_text: str) -> dict:
    """
    Full end-to-end inference for a news article with HTML word highlighting.
    """
    if not raw_text or not raw_text.strip():
        return {"error": "Please enter or paste a news article headline or text."}

    model = get_model()

    # 1. Stylometric metadata
    style = extract_stylometric_features(raw_text)

    # 2. Text preprocessing
    cleaned = clean_text(raw_text)
    if not cleaned:
        return {"error": "Input text did not contain recognizable English words."}

    # 3. Model Inference (0 = Fake, 1 = Real)
    pred_label = int(model.predict([cleaned])[0])
    probabilities = model.predict_proba([cleaned])[0]

    prob_fake = float(probabilities[0])
    prob_real = float(probabilities[1])

    is_fake = (pred_label == 0)
    label_str = "FAKE" if is_fake else "REAL"
    confidence = (prob_fake if is_fake else prob_real) * 100.0

    # 4. Explainability Layer
    explanations = explain_news(model, cleaned)

    # 5. Highlighted HTML
    highlighted_html = generate_highlighted_html(
        raw_text,
        explanations.get("fake_cues", []),
        explanations.get("real_cues", [])
    )

    return {
        "label": label_str,
        "is_fake": is_fake,
        "confidence": round(confidence, 2),
        "probabilities": {
            "Fake": round(prob_fake * 100.0, 2),
            "Real": round(prob_real * 100.0, 2)
        },
        "explainability": explanations,
        "highlighted_html": highlighted_html,
        "stylometry": style,
        "cleaned_preview": cleaned[:160] + ("..." if len(cleaned) > 160 else "")
    }


if __name__ == "__main__":
    sample = "WASHINGTON (Reuters) - Senate leaders agreed on a bipartisan spending measure Tuesday."
    res = predict_article(sample)
    print("Label:", res["label"])
