import pandas as pd

def normalize_text(text, max_words=120):
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text

print("Loading datasets...")

human = pd.read_csv("data/human_dataset.csv")
ai = pd.read_csv("data/ai_dataset.csv")
mixed = pd.read_csv("data/mixed_dataset.csv")
real_chatgpt = pd.read_csv("data/real_chatgpt_dataset.csv")

# Normalize text length
human["text"] = human["text"].apply(lambda x: normalize_text(str(x)))
ai["text"] = ai["text"].apply(lambda x: normalize_text(str(x)))
mixed["text"] = mixed["text"].apply(lambda x: normalize_text(str(x)))
real_chatgpt["text"] = real_chatgpt["text"].apply(lambda x: normalize_text(str(x)))

# Combine datasets
combined = pd.concat(
    [human, ai, mixed, real_chatgpt],
    ignore_index=True
)

combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

combined.to_csv("data/final_dataset.csv", index=False)

print("Merged dataset created successfully.")
print("Total samples:", len(combined))
print("Human samples:", len(human))
print("AI samples:", len(ai) + len(mixed) + len(real_chatgpt))