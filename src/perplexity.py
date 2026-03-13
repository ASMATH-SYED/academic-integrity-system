import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st

MODEL_NAME = "distilgpt2"

# Cache model so it loads only once
@st.cache_resource
def load_perplexity_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True  # 🔥 Force offline usage
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        local_files_only=True  # 🔥 Force offline usage
    )

    model.eval()
    return tokenizer, model


def calculate_perplexity(text):
    tokenizer, model = load_perplexity_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    perplexity = torch.exp(loss).item()
    return perplexity
