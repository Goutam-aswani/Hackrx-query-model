from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()
import time 

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                               temperature=0.2,
                               top_p=0.95,
                               google_api_key=os.getenv("GEMINI_API_KEY"))


loader = PyPDFLoader("hackerx_pdf1.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} documents from the text file.")



prompt = PromptTemplate(
    template="What is the coverage limit for cataract treatment per eye in the following document?\n{text}",
    input_variables=["text"]
)
# prompt = "What is the waiting period for Pre-Existing Diseases (PED)"
start_time = time.time()
response = model.invoke(prompt.invoke({"text": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"}))
print("A:")
print(response.content)    
# response = model.invoke("write a 5 line poem about the beauty of nature")
end_time = time.time()
# print(response.content)
print(f"\n⏱️ Response time: {end_time - start_time:.2f} seconds")