# scripts/evaluate_pipeline.py

import sys
import os
import time  # Import the time library
import textwrap

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.rag_pipeline import load_documents_from_url, create_rag_chain , setup_hybrid_retriever

# --- Test Configuration ---
DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
PINECONE_INDEX_NAME = "hackrx-rag-index"
NUM_TEST_RUNS = 1  # Reduce to 1 for faster final testing

# --- The Golden Set (with refined question for 'cataract') ---
GOLDEN_SET = [
    {
        "question": "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "ideal_answer": "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."
    },
    {
        # REFINED QUESTION: Removed the word "surgery" to better match document text
        "question": "What is the waiting period for treatment of cataracts?",
        "ideal_answer": "The policy has a specific waiting period of twenty-four (24) months for the treatment of cataracts."
    },
    {
        "question": "Are the medical expenses for an organ donor covered under this policy?",
        "ideal_answer": "Yes, the policy covers the medical expenses for an organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994."
    },
    {
        "question": "How does the policy define a 'Hospital'?",
        "ideal_answer": "A hospital is defined as an institution with at least 10 inpatient beds in towns with a population below ten lakhs, or 15 beds in all other places. It must have qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and maintain daily records of patients."
    },
    {
        "question": "Are there any sub-limits on room rent and ICU charges for Plan A?",
        "ideal_answer": "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. However, these limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN) as a package."
    }
]

def format_text(text, indent="    "):
    """Helper function to format long text for pretty printing."""
    return textwrap.fill(text, width=100, initial_indent=indent, subsequent_indent=indent)

if __name__ == "__main__":
    print("="*80)
    print("INITIALIZING RAG PIPELINE (ONE-TIME SETUP)")
    print("="*80)
    
    start_time = time.time()
    documents = load_documents_from_url(DOCUMENT_URL)
    if not documents:
        print("\nFATAL: No documents were loaded. Exiting script.")
        sys.exit(1)
    
    vector_store, tfidf_vectorizer = setup_hybrid_retriever(documents, PINECONE_INDEX_NAME)
    rag_chain = create_rag_chain(vector_store, tfidf_vectorizer)
    
    end_time = time.time()
    print(f"\nPipeline initialization complete. Took {end_time - start_time:.2f} seconds.")
    print("="*80)
    
    for i in range(NUM_TEST_RUNS):
        print(f"\n\n===== STARTING FINAL EVALUATION RUN {i + 1}/{NUM_TEST_RUNS} =====")
        total_questions = len(GOLDEN_SET)
        
        for j, item in enumerate(GOLDEN_SET):
            question = item["question"]
            ideal_answer = item["ideal_answer"]
            
            print(f"\n--- Question {j + 1}/{total_questions} ---")
            print(f"QUESTION:\n{format_text(question)}")
            
            generated_answer = rag_chain.invoke(question)
            
            print(f"IDEAL ANSWER:\n{format_text(ideal_answer)}")
            print(f"GENERATED ANSWER:\n{format_text(generated_answer)}")
            
            # Add a short pause to respect API rate limits
            time.sleep(3) 

    print("\n\n===== EVALUATION COMPLETE =====")

