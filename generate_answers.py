import pandas as pd

from transformers import pipeline

# Load model
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=200
)

# Load CSV
df = pd.read_csv("data/StackOverflow_QA_Format.csv")

generated_answers = []

for i, row in df.iterrows():

    question = str(row["Question"])

    prompt = f"""
You are an experienced software engineer.

Write a complete Stack Overflow style answer.

Question:
{question}

Requirements:
- Answer in 5-8 sentences.
- Explain the concept.
- Give a solution.
- Mention best practices.
- If applicable include a code example.
"""

    answer = generator(prompt)[0]["generated_text"]

    generated_answers.append(answer)

    print(i + 1)

df["Answer"] = generated_answers

df.to_csv("StackOverflow_QA_Format.csv", index=False)

print("Finished")