import numpy as np
from src.model_loader import load_ml_model
from src.perplexity import calculate_perplexity
from src.stylometric_features import extract_stylometric_features
from src.embedding_similarity import calculate_embedding_similarity


def predict_ai_content(text):

    # Load trained ML model & vectorizer
    model, vectorizer = load_ml_model()

    # Transform input text
    text_vector = vectorizer.transform([text])

    # ML probability (AI class assumed as 1)
    ml_prob = model.predict_proba(text_vector)[0][1]

    # -------- Perplexity --------
    perplexity = calculate_perplexity(text)

    # Normalize perplexity score
    if perplexity < 30:
        perplexity_score = 1.0
    elif perplexity < 50:
        perplexity_score = 0.8
    elif perplexity < 80:
        perplexity_score = 0.5
    else:
        perplexity_score = 0.2

    # -------- Stylometric Features --------
    features = extract_stylometric_features(text)
    avg_sentence_length = features[0]

    if avg_sentence_length > 20:
        stylometric_score = 0.8
    elif avg_sentence_length > 15:
        stylometric_score = 0.6
    else:
        stylometric_score = 0.4

    # -------- Embedding Similarity --------
    embedding_score_norm = calculate_embedding_similarity(text)

    # -------- Final Hybrid Score --------
    final_ai_score = (
        0.55 * ml_prob +
        0.25 * perplexity_score +
        0.10 * stylometric_score +
        0.10 * embedding_score_norm
    )

    # -------- Final Classification --------
    if final_ai_score > 0.60:
        classification = "AI-Generated Writing Detected"
    elif final_ai_score > 0.50:
        classification = "Mixed / AI-Assisted Writing Detected"
    else:
        classification = "Human Writing Pattern Detected"

    confidence = f"{round(final_ai_score * 100, 2)}%"

    return classification, confidence