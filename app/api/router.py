# app/api/router.py

from fastapi import APIRouter, Header, HTTPException, Depends
from typing import List

# Import our data models (schemas) and the core service functions
from app.models.schemas import HackRxRequest, HackRxResponse
from app.services.rag_pipeline import (
    load_documents_from_url, 
    setup_hybrid_retriever, 
    create_rag_chain
)

# --- Authentication Dependency ---
API_TOKEN = "1855b3ba60f1f87b9ae8200bdcf543df929ac7640f30f49b4a3ee690145ae21c"

async def verify_token(authorization: str = Header(..., description="Bearer token for authentication")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization scheme.")
    token = authorization.split(" ")[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token.")
    return token

# --- API Router Definition ---
router = APIRouter()
PINECONE_INDEX_NAME = "hackrx-rag-index"

# A simple in-memory cache for the RAG chain. For a production app,
# you would replace this with a more robust cache like Redis.
# The key will be the document URL, the value will be the callable rag_chain.
rag_chain_cache = {}

@router.post("/hackrx/run", 
             response_model=HackRxResponse, 
             dependencies=[Depends(verify_token)],
             summary="Run Submission for HackRx",
             description="Processes a document from a URL and answers a list of questions using a RAG pipeline.")
async def run_submission(request: HackRxRequest):
    """
    This endpoint performs the following steps:
    1. Authenticates the request.
    2. Downloads and processes the document specified in the URL.
    3. Creates a Hybrid Search RAG pipeline for the document (or uses a cached one).
    4. Answers each question from the list using the pipeline.
    5. Returns a list of string-based answers.
    """
    document_url = request.documents
    questions = request.questions
    
    # Check if a RAG chain for this document is already created and cached
    if document_url not in rag_chain_cache:
        print(f"--- Document not in cache. Processing and indexing: {document_url} ---")
        try:
            # Step 1: Load the document content
            documents = load_documents_from_url(document_url)
            if not documents:
                raise HTTPException(status_code=400, detail="Failed to load or process document from URL. The URL might be invalid or the document empty.")

            # Step 2: Setup the hybrid retriever (chunks, embeds, and stores vectors)
            vector_store, tfidf_vectorizer = setup_hybrid_retriever(documents, PINECONE_INDEX_NAME)

            # Step 3: Create the final, callable RAG chain and cache it
            rag_chain = create_rag_chain(vector_store, tfidf_vectorizer)
            rag_chain_cache[document_url] = rag_chain
        except Exception as e:
            # Log the full error for debugging on the server side
            print(f"An error occurred during pipeline creation: {e}")
            raise HTTPException(status_code=500, detail=f"An internal server error occurred while creating the RAG pipeline.")
    else:
        print(f"--- Found cached RAG chain for document: {document_url} ---")
        rag_chain = rag_chain_cache[document_url]

    # Step 4: Process all questions using the RAG chain
    final_answers = []
    print(f"--- Processing {len(questions)} questions... ---")
    for question in questions:
        print(f"Answering question: '{question}'")
        try:
            answer = rag_chain.invoke(question)
            final_answers.append(answer)
        except Exception as e:
            print(f"An error occurred while answering question '{question}': {e}")
            # Add a placeholder answer to maintain the response structure
            final_answers.append(f"An error occurred while processing this question.")
    
    return HackRxResponse(answers=final_answers)