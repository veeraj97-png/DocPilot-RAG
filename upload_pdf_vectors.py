import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

project_folder = Path(__file__).resolve().parent

load_dotenv(project_folder / ".env")

NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not NEBIUS_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing API keys. Check your .env file.")

PDF_PATH = project_folder / "data" / "triumphspeed400.pdf"

INDEX_NAME = "docpilot-index"
NAMESPACE = "triumph-speed400"

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

BATCH_SIZE = 20

# --------------------------------------------------
# 2. CONNECT TO NEBIUS AND PINECONE
# --------------------------------------------------

nebius = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=NEBIUS_API_KEY,
    timeout=60.0
)

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

# --------------------------------------------------
# 3. READ THE PDF
# --------------------------------------------------

print("Reading Triumph Owner's Handbook...")

reader = PdfReader(PDF_PATH)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

all_chunks = []

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text()

    if not text or not text.strip():
        continue

    page_chunks = splitter.split_text(text)

    for chunk_number, chunk in enumerate(page_chunks, start=1):

        all_chunks.append({
            "id": f"triumph-page-{page_number}-chunk-{chunk_number}",
            "text": chunk,
            "page": page_number,
            "chunk_number": chunk_number,
            "source": "triumphspeed400.pdf"
        })

print(f"Total PDF pages: {len(reader.pages)}")
print(f"Total chunks created: {len(all_chunks)}")

# --------------------------------------------------
# 4. GENERATE EMBEDDINGS AND UPLOAD IN BATCHES
# --------------------------------------------------

total_uploaded = 0

for start in range(0, len(all_chunks), BATCH_SIZE):

    batch = all_chunks[start:start + BATCH_SIZE]

    batch_number = start // BATCH_SIZE + 1

    print(f"\nProcessing batch {batch_number}...")

    texts = [item["text"] for item in batch]

    # Retry the complete batch if a request fails.
    for attempt in range(1, 4):

        try:

            response = nebius.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts
            )

            # Match embeddings to their input positions.
            embeddings = {
                item.index: item.embedding
                for item in response.data
            }

            vectors = []

            for i, item in enumerate(batch):

                embedding = embeddings[i]

                if len(embedding) != 4096:
                    raise ValueError(
                        f"Unexpected embedding dimension: {len(embedding)}"
                    )

                vectors.append({
                    "id": item["id"],
                    "values": embedding,
                    "metadata": {
                        "source": item["source"],
                        "page": item["page"],
                        "chunk_number": item["chunk_number"],
                        "text": item["text"]
                    }
                })

            index.upsert(
                vectors=vectors,
                namespace=NAMESPACE
            )

            total_uploaded += len(vectors)

            print(
                f"Batch {batch_number} uploaded: "
                f"{len(vectors)} vectors"
            )

            break

        except Exception as error:

            print(
                f"Batch {batch_number}, attempt {attempt} "
                f"failed: {type(error).__name__}"
            )

            if attempt == 3:
                raise

            time.sleep(3 * attempt)

# --------------------------------------------------
# 5. FINAL STATUS
# --------------------------------------------------

print("\n" + "=" * 50)
print("UPLOAD COMPLETE")
print("=" * 50)

print(f"Total vectors uploaded: {total_uploaded}")
print(f"Pinecone namespace: {NAMESPACE}")