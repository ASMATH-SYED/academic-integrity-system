
import re
import numpy as np
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def extract_stylometric_features(text):

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    words = re.findall(r'\b\w+\b', text.lower())

    # Basic Metrics
    avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    vocab_richness = len(set(words)) / len(words) if words else 0
    stopword_ratio = sum(1 for w in words if w in stop_words) / len(words) if words else 0

    # Sentence Length Variance (Burstiness indicator)
    sentence_lengths = [len(s.split()) for s in sentences]
    sentence_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0

    # Repetition Ratio
    repetition_ratio = 1 - (len(set(sentences)) / len(sentences)) if sentences else 0

    # Structural Regularity Score
    structural_score = 1 / (1 + sentence_variance)

    return [
        avg_sentence_length,
        vocab_richness,
        stopword_ratio,
        sentence_variance,
        repetition_ratio,
        structural_score
    ]