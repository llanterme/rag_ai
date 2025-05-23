import os
from dotenv import load_dotenv
from pinecone import Pinecone

def check_pinecone_index():
    """Check if Pinecone index exists and is properly configured."""
    # Load environment variables
    load_dotenv()
    
    # Get Pinecone configuration
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "pdf-chat")
    
    if not api_key:
        print("Error: PINECONE_API_KEY not found in environment variables.")
        return False
    
    # Initialize Pinecone client
    try:
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists
        existing_indexes = pc.list_indexes().names()
        
        if index_name in existing_indexes:
            print(f"✅ Pinecone index '{index_name}' exists.")
            
            # Get index details
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            
            print(f"Index statistics:")
            print(f"  - Total vector count: {stats.get('total_vector_count', 'N/A')}")
            print(f"  - Namespaces: {', '.join(stats.get('namespaces', {}).keys()) or 'None'}")
            
            for ns, ns_stats in stats.get('namespaces', {}).items():
                print(f"  - Namespace '{ns}' vector count: {ns_stats.get('vector_count', 'N/A')}")
            
            return True
        else:
            print(f"❌ Pinecone index '{index_name}' does not exist.")
            print(f"Available indexes: {', '.join(existing_indexes) or 'None'}")
            print("Run the ingest.py script to create the index and upload documents.")
            return False
    
    except Exception as e:
        print(f"Error connecting to Pinecone: {str(e)}")
        return False

if __name__ == "__main__":
    check_pinecone_index()
