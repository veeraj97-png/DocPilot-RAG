from pathlib import Path

# Find the folder containing main.py
project_folder = Path(__file__).resolve().parent

# Locate our knowledge document
file_path = project_folder / "data" / "sample.txt"

# Open and read the document
with open(file_path, "r", encoding="utf-8") as file:
    document = file.read()

# Display the document contents
print("Document successfully loaded!\n")
print(document)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

# Split our document into smaller chunks
chunks = text_splitter.split_text(document)

# Display the results
print("\n--- DOCUMENT CHUNKS ---")

for index, chunk in enumerate(chunks):
    print(f"\nChunk {index + 1}:")
    print(chunk)

print(f"\nTotal chunks created: {len(chunks)}")