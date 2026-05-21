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
