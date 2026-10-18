import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Step 1: Load our API keys
project_folder = Path(__file__).resolve().parent
load_dotenv(project_folder / ".env")

nebius_key = os.getenv("NEBIUS_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

if not nebius_key or not pinecone_key:
    raise ValueError("Missing API keys. Check your .env file.")

# Step 2: Connect to Nebius
nebius = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=nebius_key
)

# Step 3: Connect to Pinecone
pc = Pinecone(api_key=pinecone_key)
index = pc.Index("docpilot-index")

# Step 4: Define our question
question = "What happens when DocPilot cannot find sufficient information?"

# Step 5: Convert the question into an embedding
response = nebius.embeddings.create(
    model="Qwen/Qwen3-Embedding-8B",
    input=question
)

question_embedding = response.data[0].embedding

# Step 6: Search Pinecone for similar vectors
results = index.query(
    vector=question_embedding,
    top_k=3,
    namespace="docpilot",
    include_metadata=True
)

# Step 7: Display the retrieved chunks
print("\nQUESTION:", question)
print("\n--- RETRIEVED DOCUMENT CHUNKS ---")

for match in results.matches:
    print(f"\nChunk ID: {match.id}")
    print(f"Similarity score: {match.score:.4f}")
    print(f"Source: {match.metadata.get('source')}")
    print(f"Text: {match.metadata.get('text')}")