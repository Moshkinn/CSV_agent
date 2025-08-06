import openai
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_question_about_csv(df, question):
    # Convert dataframe to string (limited size)
    context = df.head(100).to_csv(index=False)
    prompt = f"""You are a data assistant. Use this data:\n\n{context}\n\nNow answer this question: {question}"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message["content"]

# Example
df = pd.read_csv("your_file.csv")
answer = ask_question_about_csv(df, "What are the most common categories?")
print(answer)
