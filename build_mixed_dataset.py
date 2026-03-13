import pandas as pd
import random

print("Loading datasets...")

human = pd.read_csv("data/human_dataset.csv")
ai = pd.read_csv("data/ai_dataset.csv")

mixed_texts = []

min_samples = min(len(human), len(ai))

for i in range(min_samples):
    human_text = str(human.iloc[i]["text"])
    ai_text = str(ai.iloc[i]["text"])

    h_sent = human_text.split(".")
    a_sent = ai_text.split(".")

    combined = []

    for j in range(min(len(h_sent), len(a_sent))):
        if j % 2 == 0:
            combined.append(h_sent[j].strip())
        else:
            combined.append(a_sent[j].strip())

    mixed_text = ". ".join(combined)
    mixed_texts.append(mixed_text)

mixed_df = pd.DataFrame({
    "text": mixed_texts,
    "label": 1   # treat as AI-like (hard class)
})

mixed_df.to_csv("data/mixed_dataset.csv", index=False)

print("Mixed dataset created successfully.")
print("Total mixed samples:", len(mixed_df))