import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


def train_model():
    print("Loading dataset...")

    # Load dataset
    data = pd.read_csv("data/final_dataset.csv")

    if "text" not in data.columns or "label" not in data.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    X = data["text"].astype(str)
    y = data["label"]

    print(f"Total samples: {len(data)}")
    print(f"Human samples: {len(data[data['label'] == 0])}")
    print(f"AI samples: {len(data[data['label'] == 1])}")

    # TF-IDF Vectorizer (Controlled Power)
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 1),   # Unigrams only
        stop_words="english",
        lowercase=True
    )

    X_vectorized = vectorizer.fit_transform(X)

    # Train-Test Split (for model saving)
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    print("Training model...")

    model = LogisticRegression(
        max_iter=1000,
        C=0.5
    )

    model.fit(X_train, y_train)

    print("\nEvaluating with Train/Test Split...")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Test Accuracy:", round(accuracy, 4))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # 🔥 5-Fold Cross Validation (Research-grade evaluation)
    print("\nPerforming 5-Fold Cross Validation...")

    cv_scores = cross_val_score(model, X_vectorized, y, cv=5)

    print("Cross Validation Scores:", cv_scores)
    print("Average CV Accuracy:", round(cv_scores.mean(), 4))

    # Save model & vectorizer
    pickle.dump(model, open("model/model.pkl", "wb"))
    pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

    print("\nModel and vectorizer saved successfully.")


if __name__ == "__main__":
    train_model()