import pickle
import os


def load_ml_model():

    model_path = os.path.join("model", "model.pkl")
    vectorizer_path = os.path.join("model", "vectorizer.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer