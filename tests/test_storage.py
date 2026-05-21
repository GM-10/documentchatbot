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
