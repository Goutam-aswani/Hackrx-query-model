# scripts/test_full_pipeline.py

import sys
import os
import time
import json # Import the json library for pretty printing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.services.rag_pipeline import load_documents_from_url, create_retriever, create_rag_chain

DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
PINECONE_INDEX_NAME = "hackrx-rag-index"

if __name__ == "__main__":
    start_time = time.time()
    
    print("--- Step 1 & 2: Loading, Processing, and Creating Retriever ---")
    documents = load_documents_from_url(DOCUMENT_URL)
    retriever = create_retriever(documents, PINECONE_INDEX_NAME)
    print("--- Retriever setup complete. ---")

    rag_chain = create_rag_chain(retriever)
    
    print("\n--- Step 4: Invoking RAG chain for a structured, accountable answer ---")
    question = "Does this policy cover maternity expenses, and what are the conditions?"
    print(f"Question: {question}")
    
    final_answer_dict = rag_chain.invoke(question)
    
    # Use json.dumps for pretty printing the dictionary
    print("\n--- FINAL STRUCTURED ANSWER FROM GEMINI ---")
    print(json.dumps(final_answer_dict, indent=2))
    
    end_time = time.time()
    print(f"\nTotal script execution time: {end_time - start_time:.2f} seconds.")