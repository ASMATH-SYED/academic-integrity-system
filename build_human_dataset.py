import wikipedia
import csv
import random
import re

topics = [
    "Data Science",
    "Machine Learning",
    "Artificial Intelligence",
    "Cybersecurity",
    "Deep Learning",
    "Cloud Computing",
    "Software Engineering",
    "Organizational Behavior",
    "Research Methodology",
    "Intellectual Property"
]

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

dataset = []

for topic in topics:
    try:
        page = wikipedia.page(topic)
        content = clean_text(page.content)

        paragraphs = content.split('. ')
        chunk_size = 8  # 8 sentences per sample (~150-250 words)

        for i in range(0, len(paragraphs), chunk_size):
            chunk = '. '.join(paragraphs[i:i+chunk_size])
            if len(chunk.split()) > 120:
                dataset.append([chunk, 0])  # 0 = Human

    except Exception as e:
        print(f"Skipping {topic}: {e}")

print("Total Human Samples:", len(dataset))

with open("data/human_dataset.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(dataset)

print("Human dataset saved successfully.")