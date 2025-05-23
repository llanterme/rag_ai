import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

def check_pinecone_index():
    """Check if Pinecone index exists and is properly configured.
    If the index doesn't exist, create it using the environment variables.
    """
    # Print current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if .env file exists
    env_path = os.path.join(os.getcwd(), '.env')
    print(f"Checking for .env file at: {env_path}")
    print(f"File exists: {os.path.exists(env_path)}")
    
    # Try to read the .env file directly
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r') as f:
                env_content = f.read()
            print("\n.env file content (partial):")
            for line in env_content.split('\n'):
                if line.startswith('PINECONE_'):
                    # Show the line but hide API keys
                    if 'API_KEY' in line:
                        key_parts = line.split('=')
                        if len(key_parts) > 1:
                            print(f"{key_parts[0]}=[HIDDEN]")
                    else:
                        print(line)
        except Exception as e:
            print(f"Error reading .env file: {str(e)}")
    
    # Load environment variables
    print("\nLoading environment variables with dotenv...")
    load_dotenv(dotenv_path=env_path, override=True)
    
    # Get Pinecone configuration
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    namespace = os.getenv("PINECONE_NAMESPACE")
    region = os.getenv("PINECONE_ENV", "us-east-1")
    
    print("\nEnvironment variables after loading:")
    print(f"PINECONE_INDEX_NAME = '{index_name}'")
    print(f"PINECONE_NAMESPACE = '{namespace}'")
    print(f"PINECONE_ENV = '{region}'")
    
    # If still not set, try to parse from file content
    if not index_name and 'env_content' in locals():
        for line in env_content.split('\n'):
            if line.startswith('PINECONE_INDEX_NAME='):
                index_name = line.split('=', 1)[1].strip()
                print(f"Parsed index_name from file: '{index_name}'")
    
    if not namespace and 'env_content' in locals():
        for line in env_content.split('\n'):
            if line.startswith('PINECONE_NAMESPACE='):
                namespace = line.split('=', 1)[1].strip()
                print(f"Parsed namespace from file: '{namespace}'")
    
    # Final values to use
    print("\nFinal Pinecone configuration:")
    print(f"Index Name: '{index_name}'")
    print(f"Namespace: '{namespace}'")
    print(f"Region: '{region}'")
    
    
    
    if not api_key:
        print("❌ Error: PINECONE_API_KEY not found in environment variables.")
        return False
    
    if not index_name:
        print("❌ Error: PINECONE_INDEX_NAME not found in environment variables.")
        return False
    
    if not namespace:
        print("❌ Error: PINECONE_NAMESPACE not found in environment variables.")
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
            print(f"⚠️ Pinecone index '{index_name}' does not exist. Creating it now...")
            
            try:
                # Create the index
                pc.create_index(
                    name=index_name,
                    dimension=1536,  # OpenAI embeddings dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=region
                    )
                )
                print(f"✅ Successfully created Pinecone index '{index_name}' in region '{region}'")
                print(f"✅ Using namespace '{namespace}' as specified in .env file")
                print("Now you can run the ingest.py script to upload documents.")
                return True
            except Exception as e:
                print(f"❌ Error creating Pinecone index: {str(e)}")
                return False
    
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {str(e)}")
        return False

if __name__ == "__main__":
    check_pinecone_index()
