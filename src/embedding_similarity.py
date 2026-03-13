from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import numpy as np

@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def calculate_embedding_similarity(text):
    model = load_embedding_model()

    sentences = text.split(".")
    shuffled = sentences.copy()
    np.random.shuffle(shuffled)

    original_embedding = model.encode([" ".join(sentences)])
    shuffled_embedding = model.encode([" ".join(shuffled)])

    similarity = cosine_similarity(original_embedding, shuffled_embedding)[0][0]

    return similarity