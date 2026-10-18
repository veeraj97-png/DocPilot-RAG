import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Step 1: Load the Nebius API key
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

# Step 3: Define our embedding model
model_name = "Qwen/Qwen3-Embedding-8B"

# Step 4: Define the sentence we want to convert
sentence = "DocPilot retrieves information from documents."

# Step 5: Generate the embedding
response = client.embeddings.create(
    model=model_name,
    input=sentence
)

# Step 6: Extract the numerical vector
embedding = response.data[0].embedding

# Step 7: Display the results
print("Embedding generated successfully!")
print("Original sentence:", sentence)
print("Vector dimensions:", len(embedding))
print("First 10 vector values:", embedding[:10])