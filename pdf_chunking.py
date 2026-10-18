from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Locate the PDF
project_folder = Path(__file__).resolve().parent
pdf_path = project_folder / "data" / "triumphspeed400.pdf"

# Open the PDF
reader = PdfReader(pdf_path)

# Configure text splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

all_chunks = []

# Process each page separately
for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text()

    if not text or not text.strip():
        continue

    # Split page text into smaller chunks
    page_chunks = splitter.split_text(text)

    for chunk_number, chunk in enumerate(page_chunks, start=1):

        all_chunks.append({
            "text": chunk,
            "source": "triumphspeed400.pdf",
            "page": page_number,
            "chunk_number": chunk_number
        })

# Display statistics
print("PDF processing complete!")
print(f"Total pages: {len(reader.pages)}")
print(f"Total chunks created: {len(all_chunks)}")

# Display a sample chunk
print("\n--- SAMPLE CHUNK ---")

if all_chunks:
    sample = all_chunks[0]

    print("Source:", sample["source"])
    print("Page:", sample["page"])
    print("Chunk:", sample["chunk_number"])
    print("\nText:")
    print(sample["text"])

    print("\n--- PAGE 134 CHUNKS ---")

for chunk in all_chunks:
    if chunk["page"] == 134:
        print(f"\nChunk {chunk['chunk_number']}:")
        print(chunk["text"])
        print("-" * 50)