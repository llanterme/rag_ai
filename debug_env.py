#!/usr/bin/env python3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Print environment variables
print("Environment variables:")
print(f"PINECONE_API_KEY = '{os.getenv('PINECONE_API_KEY')[:5]}...' (truncated for security)")
print(f"PINECONE_ENV = '{os.getenv('PINECONE_ENV')}'")
print(f"PINECONE_INDEX_NAME = '{os.getenv('PINECONE_INDEX_NAME')}'")
print(f"PINECONE_NAMESPACE = '{os.getenv('PINECONE_NAMESPACE')}'")
