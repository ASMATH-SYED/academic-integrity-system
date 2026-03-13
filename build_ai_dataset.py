import random
import pandas as pd

topics = [
    "machine learning", "artificial intelligence", "blockchain technology",
    "cybersecurity", "data mining", "cloud computing",
    "neural networks", "organizational behavior",
    "intellectual property rights", "research methodology",
    "big data analytics", "deep learning", "ethical AI",
    "data science in healthcare", "motivation theories"
]

openers = [
    "In recent years, {} has gained considerable attention.",
    "{} represents a transformative area in modern research.",
    "The study of {} has evolved significantly over the last decade.",
    "{} is widely regarded as a foundational discipline in technological advancement.",
    "Scholars increasingly focus on {} due to its interdisciplinary relevance."
]

bodies = [
    "It integrates theoretical principles with practical applications across industries.",
    "Researchers continue to explore its scalability, efficiency, and ethical implications.",
    "The framework underlying this domain combines statistical reasoning and computational modeling.",
    "Its implementation often involves algorithmic optimization and data-driven decision making.",
    "The conceptual foundations emphasize systematic analysis and structured methodologies."
]

expansions = [
    "Furthermore, ongoing developments are reshaping academic and industrial landscapes.",
    "Moreover, emerging trends highlight both opportunities and limitations.",
    "However, critical challenges remain in ensuring transparency and reliability.",
    "Consequently, researchers advocate for responsible innovation strategies.",
    "In addition, interdisciplinary collaboration enhances its practical impact."
]

conclusions = [
    "Overall, its influence is expected to expand in future research domains.",
    "Thus, it continues to redefine traditional academic paradigms.",
    "Therefore, sustained investigation remains essential for long-term advancement.",
    "As a result, institutions increasingly incorporate it into curricula.",
    "In conclusion, its relevance is unlikely to diminish in the coming years."
]

def generate_ai_text():
    topic = random.choice(topics)
    text = " ".join([
        random.choice(openers).format(topic.title()),
        random.choice(bodies),
        random.choice(expansions),
        random.choice(conclusions)
    ])
    return text

samples = []

for _ in range(400):  # generate more diversity
    samples.append({"text": generate_ai_text(), "label": 1})

df = pd.DataFrame(samples)
df.to_csv("data/ai_dataset.csv", index=False)

print("AI dataset rebuilt successfully.")
print("Total AI samples:", len(df))