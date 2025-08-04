from pinecone import Pinecone
# from app.core import config
GOOGLE_API_KEY="AIzaSyCfignZnH4_NBKNF8Hzeg5MOrAyGSfwXqk"
PINECONE_API_KEY="pcsk_6eULSZ_5y4wQURgb6CzwNHZ46wUk8JM6QXgMmtLskSruMTBDWityB1DRVAeM2HQ44PaguW"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("hackrx-rag-index")