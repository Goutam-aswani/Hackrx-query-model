from fastapi import APIRouter, Header, HTTPException, Depends
from app.models.schemas import HackRxRequest, HackRxResponse
from services.rag_pipeline import (
    load_documents_from_url, 
    setup_hybrid_retriever, 
    create_rag_chain
)

# --- Authentication Dependency (unchanged) ---
API_TOKEN = "1855b3ba60f1f87b9ae8200bdcf543df929ac7640f30f49b4a3ee690145ae21c"

async def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization scheme.")
    token = authorization.split(" ")[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token.")
    return token

# --- API Router Definition ---
router = APIRouter()
PINECONE_INDEX_NAME = "hackrx-rag-index"

# A more robust cache that stores the components needed for each document.
# In production, this would be a persistent cache like Redis.
pipeline_component_cache = {}

@router.post("/hackrx/run", 
             response_model=HackRxResponse, 
             dependencies=[Depends(verify_token)])
async def run_submission(request: HackRxRequest):
    document_url = request.documents
    questions = request.questions
    
    # Check if this document's components are already processed and cached
    if document_url not in pipeline_component_cache:
        print(f"--- Document not in cache. Processing and indexing: {document_url} ---")
        try:
            documents = load_documents_from_url(document_url)
            if not documents:
                raise HTTPException(status_code=400, detail="Failed to load document.")

            # Setup the retriever and get the specific components for this document
            vector_store, tfidf_vectorizer = setup_hybrid_retriever(documents, PINECONE_INDEX_NAME)
            
            # Store the essential components in our cache
            pipeline_component_cache[document_url] = {
                "vector_store": vector_store,
                "tfidf_vectorizer": tfidf_vectorizer
            }
        except Exception as e:
            print(f"Error during pipeline creation for {document_url}: {e}")
            raise HTTPException(status_code=500, detail="Failed to create RAG pipeline.")
    else:
        print(f"--- Found cached components for document: {document_url} ---")

    # Retrieve the correct components from the cache for this request
    cached_components = pipeline_component_cache[document_url]
    vector_store = cached_components["vector_store"]
    tfidf_vectorizer = cached_components["tfidf_vectorizer"]
    
    # Create a fresh RAG chain for this query using the correct components
    rag_chain = create_rag_chain(vector_store, tfidf_vectorizer)

    final_answers = []
    print(f"--- Processing {len(questions)} questions... ---")
    for question in questions:
        print(f"Answering question: '{question}'")
        try:
            answer = rag_chain.invoke(question)
            final_answers.append(answer)
        except Exception as e:
            print(f"Error answering question '{question}': {e}")
            final_answers.append("An error occurred while processing this question.")
    
    return HackRxResponse(answers=final_answers)