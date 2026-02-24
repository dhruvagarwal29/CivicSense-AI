import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import json


def chunk_nyc_code(pdf_path):
    print(f"Reading {pdf_path}...")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return []

    full_text = ""
    for page in doc:
        full_text += page.get_text()

    print("Text extracted. Splitting into overlapping chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". " " "],
    )

    raw_chunks = text_splitter.create_documents([full_text])
    processed_chunks = []

    for i, chunk_text in enumerate(raw_chunks):
        clean_text = chunk_text.page_content

        if len(clean_text) > 50:
            processed_chunks.append(
                {
                    "document_id": f"NYC_HMC_2025_chunk_{i}",
                    "code_section": f"Chunk_{i}",
                    "content": clean_text,
                    "city": "New York",
                    "state": "NY",
                    "effective_year": 2025,
                    "chunk_index": i,
                }
            )

    print(f"Success! Extracted {len(processed_chunks)} overlapping chunks.")
    return processed_chunks


if __name__ == "__main__":
    pdf_file = "data/nyc_hmc.pdf"
    chunks = chunk_nyc_code(pdf_file)

    if chunks:
        print("\n--- TEST: FIRST EXTRACTED CHUNK ---")

        print(json.dumps(chunks[0], indent=2))
        print(len(chunks))
