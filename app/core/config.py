import os
from dotenv import load_dotenv

# Load environment variables from the .env file in the project root
# This line looks for the .env file in the parent directory of the current file's directory
# which is the project root where we placed it.
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# Retrieve the environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Basic validation to ensure the keys are set
if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise ValueError("API keys for Google and Pinecone must be set in the .env file")