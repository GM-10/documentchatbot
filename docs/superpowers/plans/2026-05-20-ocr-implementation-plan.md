# OCR Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a fallback OCR pipeline in `App.py` to handle scanned PDF files.

**Architecture:** Modify `index_pdf` in `App.py` to attempt standard text extraction first. If empty, convert pages to images and perform OCR.

**Tech Stack:** Python, `pytesseract`, `pdf2image`, `langchain`.

---

### Task 1: Install and Configure OCR Dependencies

**Files:**
- System: Requires `tesseract` and `poppler` installed on the host machine.
- Modify: `requirements.txt`

- [ ] **Step 1: Add dependencies**
Run:
```bash
pip install pytesseract pdf2image
```
*Note: Verify Tesseract is installed and in your system PATH (`tesseract --version`).*

### Task 2: Implement OCR logic in App.py

**Files:**
- Modify: `App.py`

- [ ] **Step 1: Add imports and helper for OCR**

```python
import pytesseract
from pdf2image import convert_from_path

# Helper to check if text needs OCR
def needs_ocr(pdf_path):
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        if page.extract_text().strip():
            return False
    return True

# Helper to perform OCR
def perform_ocr(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image) + "
"
    return text
```

- [ ] **Step 2: Update `index_pdf` to include OCR logic**

```python
def index_pdf(pdf_path, collection_name):
    # ... existing vectordb check ...
    
    if needs_ocr(pdf_path):
        print("Scanned PDF detected, performing OCR...")
        text = perform_ocr(pdf_path)
        # Create a Document object from text
        from langchain.docstore.document import Document
        docs = [Document(page_content=text)]
    else:
        docs = PyPDFLoader(pdf_path).load()
        
    chunks = RecursiveCharacterTextSplitter(...).split_documents(docs)
    # ... existing Chroma.from_documents ...
```

### Task 3: Verification

- [ ] **Step 1: Verify with scanned PDF**

Run `python repro_bug.py` (which should now use the updated `index_pdf`).
Expected: OCR runs, text is extracted, and indexing completes without `ValueError`.
