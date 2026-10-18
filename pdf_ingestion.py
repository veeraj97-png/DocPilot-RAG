from pathlib import Path
from pypdf import PdfReader

# Step 1: Locate our project folder
project_folder = Path(__file__).resolve().parent

# Step 2: Locate the PDF document
pdf_path = project_folder / "data" / "triumphspeed400.pdf"

# Step 3: Open the PDF
reader = PdfReader(pdf_path)

# Step 4: Count the pages
total_pages = len(reader.pages)

print(f"PDF successfully loaded!")
print(f"Total pages: {total_pages}")

# Step 5: Extract text from selected pages
pages_to_check = [0, 115, 125, 133]

for page_index in pages_to_check:

    page = reader.pages[page_index]

    text = page.extract_text()

    print("\n" + "=" * 50)
    print(f"PDF Page: {page_index + 1}")
    print("=" * 50)

    print(text[:1000] if text else "No extractable text found.")