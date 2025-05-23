import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Document processing configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "4"))

# File paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INGESTED_FILES_PATH = os.path.join(DATA_DIR, "ingested_files.json")

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)
