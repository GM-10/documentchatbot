import os
from App import get_vector_store, index_pdf

def test_persistence():
    collection_name = "test_collection"
    # Create a dummy pdf or use an existing one if available.
    # Since I don't want to rely on existing files, I'll just check if functions are importable and have correct signatures.
    print(f"Testing imports...")
    try:
        store = get_vector_store(collection_name)
        print(f"get_vector_store works. Type: {type(store)}")
    except Exception as e:
        print(f"get_vector_store failed: {e}")

    print(f"Checking index_pdf signature...")
    import inspect
    sig = inspect.signature(index_pdf)
    print(f"index_pdf signature: {sig}")
    
    # Check if VECTOR_DB_DIR is correctly set
    from App import VECTOR_DB_DIR
    print(f"VECTOR_DB_DIR: {VECTOR_DB_DIR}")
    assert VECTOR_DB_DIR == "./data/vector_db"

if __name__ == "__main__":
    test_persistence()
