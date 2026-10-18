import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Step 1: Load our API key
project_folder = Path(__file__).resolve().parent
load_dotenv(project_folder / ".env")

api_key = os.getenv("NEBIUS_API_KEY")

if not api_key:
    raise ValueError("Nebius API key not found!")

# Step 2: Connect to Nebius
client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=api_key
)

# Step 3: Send a question to DeepSeek
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V4-Flash-0731",
    messages=[
        {
            "role": "user",
            "content": "Explain Retrieval-Augmented Generation in one sentence."
        }
    ],
    max_completion_tokens=200
)

# Step 4: Extract and display the answer
answer = response.choices[0].message.content

print("\nAI-generated answer:")
print(answer)