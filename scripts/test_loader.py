# scripts/test_loader.py

import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.services.rag_pipeline import load_documents_from_url, create_retriever

DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
PINECONE_INDEX_NAME = "hackrx-rag-index"

if __name__ == "__main__":
    start_time = time.time()
    
    print("--- Step 1: Loading Documents ---")
    documents = load_documents_from_url(DOCUMENT_URL)
    
    if not documents:
        print("Failed to load documents. Exiting.")
        sys.exit(1)
        
    print("\n--- Step 2: Creating Advanced Retriever with Re-ranking ---")
    try:
        retriever = create_retriever(documents, PINECONE_INDEX_NAME)
    except Exception as e:
        print(f"\nAn error occurred during retriever creation: {e}")
        print(f"Please ensure you have created a Pinecone index named '{PINECONE_INDEX_NAME}' with dimension 768.")
        sys.exit(1)

    # --- Step 3: Test the ADVANCED retriever ---
    print("\n--- Step 3: Testing Retriever with a specific query ---")
    # A more specific query where re-ranking can shine
    sample_query = "Are there any sub-limits on room rent and ICU charges for Plan A?"
    print(f"Sample Query: '{sample_query}'")
    
    # The first time you run this, flashrank will download its model. This is a one-time event.
    print("Invoking retriever... (First run may download the re-ranking model)")
    retrieved_docs = retriever.invoke(sample_query)
    
    print(f"\nRetrieved {len(retrieved_docs)} re-ranked documents.")
    
    if retrieved_docs:
        print("\n--- Content of TOP-RANKED Retrieved Document ---")
        # This document should be highly relevant thanks to the re-ranker
        print(retrieved_docs[0].page_content)
        print("\nMetadata:", retrieved_docs[0].metadata)
    
    end_time = time.time()
    print(f"\nTotal script execution time: {end_time - start_time:.2f} seconds.")