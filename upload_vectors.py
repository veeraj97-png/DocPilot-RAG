import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load API credentials
project_folder = Path(__file__).resolve().parent
load_dotenv(project_folder / ".env")

nebius_key = os.getenv("NEBIUS_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

if not nebius_key or not pinecone_key:
    raise ValueError("Missing Nebius or Pinecone API key.")

# Step 2: Connect to Nebius
nebius = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=nebius_key
)

# Step 3: Connect to Pinecone
pc = Pinecone(api_key=pinecone_key)
index = pc.Index("docpilot-index")

# Step 4: Read our knowledge document
file_path = project_folder / "data" / "sample.txt"
document = file_path.read_text(encoding="utf-8")

# Step 5: Split the document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_text(document)
print(f"Created {len(chunks)} chunks.")

# Step 6: Generate embeddings
response = nebius.embeddings.create(
    model="Qwen/Qwen3-Embedding-8B",
    input=chunks
)

# Step 7: Prepare records for Pinecone
vectors = []

for i, item in enumerate(response.data):
    vectors.append({
        "id": f"docpilot-chunk-{i + 1}",
        "values": item.embedding,
        "metadata": {
            "source": "sample.txt",
            "chunk_number": i + 1,
            "text": chunks[i]
        }
    })

# Step 8: Upload records to Pinecone
index.upsert(vectors=vectors, namespace="docpilot")

print(f"Successfully uploaded {len(vectors)} vectors to Pinecone!")