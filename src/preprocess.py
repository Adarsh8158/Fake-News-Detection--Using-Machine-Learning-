"""
Text Preprocessing and Stylometric Signal Extraction
Handles cleaning, normalization, and stylistic analysis for news text.
"""

import re
import string


def clean_text(text: str) -> str:
    """
    Cleans raw news headline or article body text.
    1. Converts text to lowercase.
    2. Strips URL links (http, https, www).
    3. Strips HTML tags.
    4. Strips special symbols and brackets.
    5. Normalizes whitespace.

    Args:
        text (str): Raw news text.

    Returns:
        str: Normalized clean text ready for TF-IDF.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove HTML markup
    text = re.sub(r"<.*?>", " ", text)

    # 4. Remove punctuation marks
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 5. Remove numbers/digits
    text = re.sub(r"\d+", " ", text)

    # 6. Normalize multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_stylometric_features(raw_text: str) -> dict:
    """
    Extracts stylistic signals from raw, uncleaned text.
    Fake news and clickbait frequently rely on psychological cues like
    excessive ALL CAPS words, multiple exclamation marks, and urgent phrasing.

    Args:
        raw_text (str): Original uncleaned news text.

    Returns:
        dict: Stylometric measurements and sensationalism index.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {
            "char_count": 0,
            "word_count": 0,
            "uppercase_words": 0,
            "exclamation_count": 0,
            "question_count": 0,
            "sensationalism_score": 0.0
        }

    words = raw_text.split()
    total_words = len(words)
    exclamations = raw_text.count("!")
    questions = raw_text.count("?")

    # Strip punctuation from tokens to accurately detect words in ALL CAPS
    caps_words = []
    for w in words:
        clean_w = w.strip(string.punctuation)
        if clean_w.isupper() and len(clean_w) > 1 and clean_w.isalpha():
            caps_words.append(clean_w)

    caps_count = len(caps_words)

    # Heuristic Sensationalism Score (0.0 to 1.0)
    score = 0.0
    if exclamations > 0:
        score += min(0.35, exclamations * 0.1)
    if questions > 1:
        score += min(0.15, (questions - 1) * 0.05)
    if caps_count > 0:
        ratio = caps_count / max(total_words, 1)
        score += min(0.5, ratio * 2.0)

    return {
        "char_count": len(raw_text),
        "word_count": total_words,
        "uppercase_words": caps_count,
        "exclamation_count": exclamations,
        "question_count": questions,
        "sensationalism_score": round(min(score, 1.0) * 100, 1)
    }


if __name__ == "__main__":
    test_str = "SHOCKING ALERT: Miracle Lemon Water Cures Cancer In 48 Hours!! Doctors Terrified!"
    print("Cleaned:", clean_text(test_str))
    print("Stylometry:", extract_stylometric_features(test_str))
