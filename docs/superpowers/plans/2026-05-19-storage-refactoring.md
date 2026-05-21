# Storage Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `StorageManager` to ensure atomic metadata writes and dynamic dates, and update tests to use `pytest` fixtures for better isolation.

**Architecture:**
- `StorageManager` will utilize `tempfile.mkstemp` and `os.replace` for atomic JSON writes to prevent corruption.
- `datetime.now()` will replace hardcoded dates for document metadata.
- `tests/test_storage.py` will be refactored to use the `tmp_path` fixture, removing manual directory management.

**Tech Stack:** Python, `pytest`.

---

### Task 1: Refactor StorageManager

**Files:**
- Modify: `D:\qa bot\storage_manager.py`

- [ ] **Step 1: Update StorageManager with atomic writes and dynamic dates**

```python
import os
import json
import tempfile
from datetime import datetime

class StorageManager:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self.pdfs_dir = os.path.join(base_dir, "pdfs")
        self.db_dir = os.path.join(base_dir, "vector_db")
        self.metadata_path = os.path.join(base_dir, "metadata.json")
        
        os.makedirs(self.pdfs_dir, exist_ok=True)
        os.makedirs(self.db_dir, exist_ok=True)
        
        if not os.path.exists(self.metadata_path):
            self._save_metadata([])

    def _save_metadata(self, docs):
        fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(self.metadata_path))
        with os.fdopen(fd, 'w') as f:
            json.dump(docs, f)
        os.replace(temp_path, self.metadata_path)

    def get_all_docs(self):
        with open(self.metadata_path, 'r') as f:
            return json.load(f)

    def add_doc(self, name, collection_id):
        docs = self.get_all_docs()
        docs.append({
            "name": name, 
            "id": collection_id, 
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        self._save_metadata(docs)

    def delete_doc(self, collection_id):
        docs = self.get_all_docs()
        new_docs = [d for d in docs if d['id'] != collection_id]
        self._save_metadata(new_docs)
```

---

### Task 2: Refactor tests/test_storage.py

**Files:**
- Modify: `D:\qa bot\tests\test_storage.py`

- [ ] **Step 1: Refactor tests to use tmp_path fixture**

```python
import os
import pytest
from storage_manager import StorageManager

def test_metadata_init(tmp_path):
    test_dir = tmp_path / "data"
    sm = StorageManager(base_dir=str(test_dir))
    assert os.path.exists(os.path.join(str(test_dir), "metadata.json"))
    assert sm.get_all_docs() == []

def test_add_delete_doc(tmp_path):
    test_dir = tmp_path / "data"
    sm = StorageManager(base_dir=str(test_dir))
    
    # Test add
    sm.add_doc("Test Doc", "id123")
    docs = sm.get_all_docs()
    assert len(docs) == 1
    assert docs[0]["name"] == "Test Doc"
    assert docs[0]["id"] == "id123"
    assert "date" in docs[0]
    
    # Test delete
    sm.delete_doc("id123")
    assert sm.get_all_docs() == []
```

---

### Task 3: Verify Implementation

- [ ] **Step 1: Run tests**

Run: `pytest tests/test_storage.py -v`
Expected: ALL PASS
