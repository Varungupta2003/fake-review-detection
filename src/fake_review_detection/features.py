from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd
import textstat

try:
    import spacy
except ImportError:
    spacy = None


class StylometricFeatureExtractor:
    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self.nlp = None
        self.spacy_model = spacy_model

    def _load_nlp(self):
        if self.nlp is None:
            if spacy is None:
                raise ImportError("spaCy is required for POS-based features.")
            self.nlp = spacy.load(self.spacy_model, disable=["ner"])

    @staticmethod
    def _safe_div(num: float, den: float) -> float:
        return float(num) / float(den) if den else 0.0

    def extract_one(self, text: str) -> dict[str, float]:
        self._load_nlp()
        doc = self.nlp(text)

        tokens = [t.text for t in doc if not t.is_space]
        alpha_tokens = [t.text.lower() for t in doc if t.is_alpha]
        sentences = list(doc.sents)

        token_count = len(tokens)
        word_count = len(alpha_tokens)
        sentence_count = len(sentences)
        char_count = len(text)
        unique_words = len(set(alpha_tokens))
        word_lengths = [len(w) for w in alpha_tokens]
        sentence_lengths = [sum(1 for t in sent if t.is_alpha) for sent in sentences]

        counts = Counter(alpha_tokens)
        hapax_count = sum(1 for _, c in counts.items() if c == 1)

        pos_counts = Counter(t.pos_ for t in doc if t.is_alpha)
        punct_counts = Counter(ch for ch in text if ch in "!?.,;:")

        polarity_proxy_positive = sum(1 for w in alpha_tokens if w in {"great", "good", "excellent", "amazing", "perfect", "love"})
        polarity_proxy_negative = sum(1 for w in alpha_tokens if w in {"bad", "poor", "terrible", "awful", "worst", "hate"})

        features = {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": float(np.mean(word_lengths)) if word_lengths else 0.0,
            "avg_sentence_length": float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
            "type_token_ratio": self._safe_div(unique_words, word_count),
            "hapax_ratio": self._safe_div(hapax_count, word_count),
            "uppercase_ratio": self._safe_div(sum(1 for ch in text if ch.isupper()), max(char_count, 1)),
            "digit_ratio": self._safe_div(sum(1 for ch in text if ch.isdigit()), max(char_count, 1)),
            "exclamation_count": punct_counts["!"],
            "question_count": punct_counts["?"],
            "comma_count": punct_counts[","],
            "period_count": punct_counts["."],
            "semicolon_count": punct_counts[";"],
            "colon_count": punct_counts[":"],
            "noun_ratio": self._safe_div(pos_counts["NOUN"], word_count),
            "verb_ratio": self._safe_div(pos_counts["VERB"], word_count),
            "adj_ratio": self._safe_div(pos_counts["ADJ"], word_count),
            "adv_ratio": self._safe_div(pos_counts["ADV"], word_count),
            "pron_ratio": self._safe_div(pos_counts["PRON"], word_count),
            "stopword_ratio": self._safe_div(sum(1 for t in doc if t.is_stop), token_count),
            "repeated_word_ratio": self._safe_div(word_count - unique_words, word_count),
            "flesch_reading_ease": float(textstat.flesch_reading_ease(text)) if word_count else 0.0,
            "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(text)) if word_count else 0.0,
            "gunning_fog": float(textstat.gunning_fog(text)) if word_count else 0.0,
            "sentiment_positive_proxy": polarity_proxy_positive,
            "sentiment_negative_proxy": polarity_proxy_negative,
        }
        return features

    def transform(self, texts: Iterable[str]) -> pd.DataFrame:
        rows = [self.extract_one(text) for text in texts]
        return pd.DataFrame(rows)
