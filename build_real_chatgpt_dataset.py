import pandas as pd

chatgpt_samples = []

topics = [
    "Data Science",
    "Machine Learning",
    "Artificial Intelligence",
    "Deep Learning",
    "Research Methodology",
    "Intellectual Property Rights",
    "Organizational Behavior",
    "Cybersecurity",
    "Software Engineering",
    "Cloud Computing"
]

template_paragraph = """
{topic} is a significant domain in modern technology and academic research.
It involves systematic approaches, structured methodologies, and analytical thinking to solve complex real-world problems.
In academic contexts, {topic} emphasizes theoretical foundations along with practical implementation strategies.
The discipline integrates interdisciplinary knowledge, ensuring that decision-making processes are data-driven and logically sound.
Overall, {topic} contributes to innovation, efficiency, and sustainable technological development across industries.
"""

# Generate 50 samples (5 variations per topic)
for i in range(5):
    for topic in topics:
        chatgpt_samples.append(template_paragraph.format(topic=topic))

df = pd.DataFrame({
    "text": chatgpt_samples,
    "label": 1
})

df.to_csv("data/real_chatgpt_dataset.csv", index=False)

print("Real ChatGPT dataset created successfully.")
print("Total samples:", len(df))