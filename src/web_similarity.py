import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from serpapi import GoogleSearch

def check_web_similarity(user_text):

    api_key = st.secrets["SERPAPI_KEY"]

    # ---- SERP API Search ----
    params = {
        "engine": "google",
        "q": user_text[:200],  # Limit query size
        "api_key": api_key,
        "num": 5
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    web_matches = []

    if "organic_results" not in results:
        return []

    for result in results["organic_results"][:3]:

        title = result.get("title", "")
        link = result.get("link", "")
        snippet = result.get("snippet", "")

        if not snippet:
            continue

        # ---- Cosine Similarity ----
        vectorizer = TfidfVectorizer().fit_transform([user_text, snippet])
        similarity_matrix = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
        similarity_score = similarity_matrix[0][0] * 100

        web_matches.append({
            "title": title,
            "url": link,
            "similarity": round(similarity_score, 2)
        })

    # Sort by similarity
    web_matches = sorted(web_matches, key=lambda x: x["similarity"], reverse=True)

    return web_matches[:3]