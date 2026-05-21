import os
from App import index_pdf

# Mock setup
pdf_path = "D:/qa bot/data/pdfs/llm q a paper.pdf" # Replace with a real file path if needed
collection_id = "test_collection"

# Ensure the pdf exists
if not os.path.exists(pdf_path):
    print(f"Error: PDF not found at {pdf_path}")
    exit(1)

try:
    print("Indexing PDF...")
    index_pdf(pdf_path, collection_id)
    print("Successfully indexed!")
except Exception as e:
    print(f"Failed to index: {e}")
