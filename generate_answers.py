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
    retrieved_answer = str(row["Answer"])

    prompt = f"""
    You are an experienced software engineer and Stack Overflow expert.

    Your task is to improve the existing Stack Overflow answer using ONLY the information provided below.

    Question:
    {question}

    Retrieved Answer from Dataset:
    {retrieved_answer}

    Instructions:
    1. Keep the meaning of the retrieved answer exactly the same.
    2. Rewrite it in clear, professional English.
    3. Explain the concept step by step.
    4. Add missing details only if they logically follow from the retrieved answer.
    5. Include best practices whenever applicable.
    6. If the retrieved answer contains code, preserve and format it correctly.
    7. Do NOT invent facts or add information unrelated to the retrieved answer.
    8. If the retrieved answer is already complete, simply improve its readability.

    Return only the final improved answer.
    """