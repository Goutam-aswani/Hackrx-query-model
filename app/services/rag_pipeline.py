
import requests
import tempfile
import os
from typing import List
import hashlib

# --- Existing Imports ---
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader

# --- NEW IMPORTS ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone as PineconeClient,ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Literal

from pinecone import Pinecone as PineconeClient, PodSpec

from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

from app.core import config

pinecone_client = PineconeClient(api_key=config.PINECONE_API_KEY)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=config.GOOGLE_API_KEY)

def _sanitize_metadata(docs: List[Document]) -> List[Document]:
    """
    Sanitizes the metadata of a list of documents to ensure compatibility with Pinecone.
    Converts all non-supported data types in metadata to strings.
    """
    print("Sanitizing document metadata for Pinecone compatibility...")
    for doc in docs:
        sanitized_meta = {}
        if doc.metadata:
            for k, v in doc.metadata.items():
                if isinstance(v, (str, int, float, bool, list)):
                    # Pinecone supports lists of strings, let's ensure all items are strings
                    if isinstance(v, list):
                        sanitized_meta[k] = [str(item) for item in v]
                    else:
                        sanitized_meta[k] = v
                else:
                    # Convert all other complex types (like dicts) to a string
                    sanitized_meta[k] = str(v)
        doc.metadata = sanitized_meta
    print("Metadata sanitization complete.")
    return docs

def load_documents_from_url(url: str) -> List[Document]:
    """
    Downloads a document from a URL, processes it using Unstructured,
    and then combines all content into a single LangChain Document object.
    This provides the full context to the ParentDocumentRetriever.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            print(f"Downloading document from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            
            temp_filepath = temp_file.name

        print(f"Document downloaded. Processing with Unstructured...")
        loader = UnstructuredLoader(temp_filepath, mode="elements")
        elements = loader.load() # This gives us the 719 small elements
        
        # --- THIS IS THE CRITICAL FIX ---
        # We will now combine the content of all elements back into one string.
        # This gives our ParentDocumentRetriever the full context to work with.
        full_text = "\n\n".join([el.page_content for el in elements])
        
        # We create a new, single Document with the combined text.
        # We can copy the metadata from the first element for source tracking.
        if elements:
            source_metadata = elements[0].metadata
            source_metadata.pop('page_number', None) # Page number is not relevant for the full doc
        else:
            source_metadata = {}

        full_doc = Document(page_content=full_text, metadata=source_metadata)
        
        print(f"Successfully combined document into a single text block of {len(full_text)} characters.")
        
        # We return a list containing just our one, complete document.
        return [full_doc]

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        return []
    except Exception as e:
        print(f"An error occurred during document processing: {e}")
        return []
    finally:
        if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            print(f"Cleaned up temporary file.")
def setup_hybrid_retriever(docs: List[Document], pinecone_index_name: str):
    """
    Sets up a Pinecone index for hybrid search, ensuring the chunk text is
    stored in the metadata for retrieval.
    """
    print("\n--- Starting Hybrid Search Setup ---")

    if pinecone_index_name not in pinecone_client.list_indexes().names():
        print(f"Creating new Pinecone index '{pinecone_index_name}' for hybrid search...")
        pinecone_client.create_index(
            name=pinecone_index_name,
            dimension=768,
            metric='dotproduct',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print("Index created successfully in aws/us-east-1 environment.")

    index = pinecone_client.Index(pinecone_index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"Document split into {len(chunks)} chunks.")
    
    chunks = _sanitize_metadata(chunks)

    print("Creating sparse vectors with TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer().fit([c.page_content for c in chunks])
    sparse_embeddings = tfidf_vectorizer.transform([c.page_content for c in chunks])

    print("Creating dense vectors with Gemini...")
    dense_embeddings = embedding_model.embed_documents([c.page_content for c in chunks])

    print("Upserting dense and sparse vectors to Pinecone...")
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_dense = dense_embeddings[i:i+batch_size]
        batch_sparse = sparse_embeddings[i:i+batch_size]
        
        vectors_to_upsert = []
        for j, chunk in enumerate(batch_chunks):
            chunk_id = hashlib.md5(chunk.page_content.encode('utf-8')).hexdigest()
            sparse_dict = {
                "indices": batch_sparse[j].indices.tolist(),
                "values": batch_sparse[j].data.tolist()
            }
            
            # --- THIS IS THE FINAL, CRITICAL FIX ---
            # We must explicitly add the chunk's text to the metadata payload.
            metadata_payload = chunk.metadata
            metadata_payload['page_content'] = chunk.page_content
            # --- END OF FIX ---
            
            vectors_to_upsert.append({
                "id": chunk_id,
                "values": batch_dense[j],
                "sparse_values": sparse_dict,
                "metadata": metadata_payload # Pass the corrected metadata
            })
        
        index.upsert(vectors=vectors_to_upsert)
    
    print("Upsert complete.")
    print("\n--- Hybrid Search Setup Complete ---")
    return vector_store, tfidf_vectorizer



# --- FINAL, TUNED create_rag_chain FUNCTION ---
def create_rag_chain(vector_store: PineconeVectorStore, tfidf_vectorizer: TfidfVectorizer):
    """
    Creates the final RAG chain with a prompt tuned for concise, accurate answers.
    """
    print("Creating the RAG Chain with Hybrid Search and Tuned Prompt...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.0
    )
    
    # --- PROMPT TEMPLATE TUNED FOR CONCISENESS ---
    prompt_template = """
    You are an expert assistant for analyzing insurance policy documents.
    Your task is to answer the user's question based ONLY on the provided context.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    INSTRUCTIONS:
    1. Carefully read the provided context to find the most direct answer to the question.
    2. Your answer MUST be concise and to the point, ideally 1-3 sentences.
    3. If the answer includes a specific number, duration, or key condition (like an exception), you MUST include it.
    4. Do not start your answer with phrases like "Based on the context..." or "The provided text states...".
    5. If the information is not present in the context, state that the answer cannot be found in the document.
    6. Be factually accurate and directly sourced from the document.
    7. Use formal language and retain any legal or technical terminology as mentioned in the policy.
    8. Follow the same order as the questions.
    9. If the question is not answerable, respond with "The answer to this question is not available in the document."
    10.Include all relevant numeric data found in the document, such as:

Age limits (e.g., minimum and maximum entry age)

Waiting periods (e.g., “36 months")

Coverage limits (e.g., “₹25,000 per claim, maximum of 3 claims per year”)

Percentages (e.g., “10% co-payment for claimants over 60 years”)

Timeframes (e.g., “policy term: 20 years”, “grace period: 30 days”)

Specific figures, quantities, and durations stated in the document

Use policy wording, specific section references, and precise details from the document.

EXAMPLE ANSWERS:
Example 1:
Q: What is the entry age and renewal policy for the Star Comprehensive Insurance?
A:Entry age: Minimum 3 months; maximum 65 years at entry.Lifelong renewals are guaranteed.There is no exit age.Dependent children can be covered up to 25 years of age.

Example 2:
Q: Are maternity and newborn expenses covered, and what are the waiting periods?
A:Maternity (delivery/hospitalization) is covered, with limits based on the sum insured.Normal delivery: ₹10,000–₹25,000 per delivery (see table in policy for exact SI).Caesarean section: ₹15,000–₹40,000 per delivery.Coverage for the newborn: Up to ₹50,000 or ₹1,00,000, depending on the plan.Vaccination expenses for the newborn: Up to ₹1,000 (only if delivery claim is approved).A 36-month waiting period applies before these benefits become available.

ALWAYS extract and include any and all applicable numeric values (age, rupees, years, months, %, days, etc.) provided in the document and present them in your answer.

    FINAL CONCISE ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    def hybrid_search_retriever(query: str):
        # This inner function remains unchanged
        print(f"Performing hybrid search for: '{query}'")
        dense_vec = embedding_model.embed_query(query)
        sparse_matrix = tfidf_vectorizer.transform([query])
        query_sparse_dict = {
            "indices": sparse_matrix.indices.tolist(),
            "values": sparse_matrix.data.tolist()
        }
        results = vector_store._index.query(
            vector=dense_vec,
            sparse_vector=query_sparse_dict,
            top_k=7,
            alpha=0.5, # Explicitly setting the balance
            include_metadata=True
        )
        context_chunks = [match['metadata'].get('page_content', '') for match in results['matches']]
        context = "\n\n---\n\n".join(context_chunks)
        return context

    rag_chain = (
        {"context": RunnableLambda(hybrid_search_retriever), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("Hybrid search RAG chain with tuned prompt created successfully.")
    return rag_chain

