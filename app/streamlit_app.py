import os
import streamlit as st
import asyncio
from pathlib import Path
from httpx import AsyncClient
import tempfile
import json
import logging
import traceback
from datetime import datetime

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f'rag_ai_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Create logger for this module
logger = logging.getLogger("streamlit_app")
logger.info(f"Starting RAG AI application, logging to {log_file}")

from pydantic_ai.messages import ModelRequest, ModelResponse, PartDeltaEvent, PartStartEvent, TextPartDelta

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent import RAGAgentWrapper
from app.advanced_agent import AdvancedRAGAgentWrapper
from ingest import ingest_documents
from config import DATA_DIR
from utils.user_manager import UserManager
from utils.vector_store import PineconeStore

logger.info("All modules imported successfully")

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize session state
def init_session_state():
    """Initialize the Streamlit session state with default values."""
    defaults = {
        'user_code': None,        # User's access code
        'index_name': None,       # Name of the Pinecone index
        'show_chat': False,       # Whether to show the chat interface
        'last_query': None,       # Last query made by the user
        'last_sources': None,     # Sources for the last query
        'messages': [],           # Chat message history
    }
    
    # Set default values for any missing session state variables
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize the session state
init_session_state()

def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    """
    try:
        # User messages
        if part.part_kind == 'user-prompt' and part.content:
            with st.chat_message("user"):
                st.markdown(part.content)
        # AI messages
        elif part.part_kind == 'text' and part.content:
            with st.chat_message("assistant"):
                st.markdown(part.content)
        # Tool calls and returns for displaying source attribution
        elif part.part_kind == 'tool-call' and part.tool_name == 'query_knowledge_base':
            # Safely get query from args, handling cases where args might be None or not a dict
            query = ''
            if hasattr(part, 'args') and part.args:
                if isinstance(part.args, dict):
                    query = part.args.get('query', '')
                elif hasattr(part.args, 'get'):
                    query = part.args.get('query', '')
            st.session_state.last_query = query
        elif part.part_kind == 'tool-return' and part.tool_name == 'query_knowledge_base':
            if part.content:
                st.session_state.last_sources = part.content
    except Exception as e:
        st.error(f"Error displaying message part: {str(e)}")
        st.error(f"Part details: {part}")
        if hasattr(part, 'args'):
            st.error(f"Part args type: {type(part.args)}")
            st.error(f"Part args: {part.args}")
    
async def run_agent_with_streaming(user_input):
    """Run the agent with streaming response."""
    import logging
    import traceback
    logger = logging.getLogger("streamlit_app")
    
    logger.info(f"Starting run_agent_with_streaming with input: {user_input[:50]}...")
    logger.info(f"Session state contains user_code: {'user_code' in st.session_state}")
    
    if 'user_code' not in st.session_state:
        logger.error("No user_code in session state")
        yield "Error: No user code found. Please restart the application."
        return
    
    logger.info(f"Using user_code: {st.session_state.user_code}")
    
    try:
        # Initialize the agent with the user's code
        logger.info("Initializing RAGAgentWrapper")
        agent_wrapper = RAGAgentWrapper(user_code=st.session_state.user_code)
        logger.info("RAGAgentWrapper initialized successfully")
        
        # Process the user input and get the response
        logger.info("Calling agent_wrapper.query")
        response = await agent_wrapper.query(user_input)
        logger.info(f"Agent query completed, response type: {type(response)}")
        
        if not response:
            logger.warning("Empty response from agent")
            yield "I couldn't generate a response. Please try again."
            return
        
        # Store the query for sources display
        st.session_state.last_query = user_input
            
        # Handle different response types
        # First, check if it's a pydantic_ai.agent.AgentRunResult
        if hasattr(response, 'output') and str(type(response)) == "<class 'pydantic_ai.agent.AgentRunResult'>":
            logger.info(f"AgentRunResult response, extracting output")
            # Extract the output content from the AgentRunResult
            content = response.output
            logger.info(f"Extracted content from AgentRunResult, length: {len(content) if content else 0}")
            yield content
        # If the response is a string, yield it directly
        elif isinstance(response, str):
            logger.info(f"String response, length: {len(response)}")
            # For consistency, yield the entire string at once
            yield response
        # If the response is a list or other iterable, yield each item
        elif hasattr(response, '__iter__') and not isinstance(response, dict):
            logger.info(f"Iterable response, type: {type(response)}")
            # Convert to list first to avoid async generator issues
            response_list = []
            try:
                # Handle both regular iterables and async generators
                if hasattr(response, '__aiter__'):  # It's an async iterable
                    logger.info("Processing async iterable response")
                    async for chunk in response:
                        if chunk:  # Skip empty chunks
                            # Check if chunk is AgentRunResult
                            if hasattr(chunk, 'output') and str(type(chunk)) == "<class 'pydantic_ai.agent.AgentRunResult'>":
                                response_list.append(chunk.output)
                            else:
                                response_list.append(str(chunk))
                else:  # It's a regular iterable
                    logger.info("Processing regular iterable response")
                    for chunk in response:
                        if chunk:  # Skip empty chunks
                            # Check if chunk is AgentRunResult
                            if hasattr(chunk, 'output') and str(type(chunk)) == "<class 'pydantic_ai.agent.AgentRunResult'>":
                                response_list.append(chunk.output)
                            else:
                                response_list.append(str(chunk))
            except Exception as e:
                logger.error(f"Error processing iterable response: {str(e)}")
                logger.error(traceback.format_exc())
                yield f"Error processing response: {str(e)}"
                return
                
            # Now yield each chunk
            for chunk in response_list:
                logger.info(f"Yielding chunk: {chunk[:50] if chunk else ''}...")
                yield chunk
        else:
            logger.info(f"Other response type: {type(response)}")
            # Try to convert to string in a way that gets the content, not the object representation
            if hasattr(response, '__str__'):
                yield str(response)
            else:
                yield f"Response of type {type(response)} could not be processed"
            
    except Exception as e:
        error_msg = f"Error processing your request: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        st.error(error_msg)
        yield error_msg

def display_sources():
    """Display source attribution for the last query."""
    if hasattr(st.session_state, 'last_sources') and st.session_state.last_sources:
        sources = st.session_state.last_sources
        if not isinstance(sources, list):
            return
            
        st.markdown("### Sources")
        
        # Create tabs for each source
        tabs = st.tabs([f"Source {i+1}" for i in range(len(sources))])
        
        for i, (tab, source) in enumerate(zip(tabs, sources)):
            with tab:
                # Display source metadata
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"**File**: {source.get('metadata', {}).get('source', 'Unknown')}")
                with col2:
                    st.markdown(f"**Relevance**: {source.get('metadata', {}).get('score', 'N/A'):.4f}")
                
                # Display content in a scrollable container
                st.markdown("**Content:**")
                content = source.get('content', 'No content available')
                st.markdown(
                    f'<div style="border: 1px solid #e0e0e0; border-radius: 0.5rem; '
                    f'padding: 1rem; max-height: 300px; overflow-y: auto;">'
                    f'{content}</div>',
                    unsafe_allow_html=True
                )
                
                # Add some spacing between sources
                if i < len(sources) - 1:
                    st.markdown("---")

def handle_file_upload():
    """Handle file upload and ingestion."""
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Process Files"):
        with st.spinner("Processing files..."):
            # Save uploaded files to temporary directory
            temp_files = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                temp_files.append(file_path)
            
            # Check if user is logged in
            if 'user_code' not in st.session_state or not st.session_state.user_code:
                st.error("Please log in with a valid user code first.")
                return
                
            try:
                # Ingest documents with user code
                num_ingested = ingest_documents(file_paths=temp_files, user_code=st.session_state.user_code)
                
                if num_ingested > 0:
                    st.success(f"Successfully ingested {num_ingested} new documents!")
                    # Clear chat history to ensure new documents are considered
                    st.session_state.messages = []
                else:
                    st.info("No new documents were ingested. They may have been processed already.")
            except Exception as e:
                st.error(f"Error ingesting documents: {str(e)}")

def show_user_management():
    """Display the user management interface."""
    st.title("📚 RAG AI Assistant")
    
    # Show either the new index creation form or the access code form
    tab1, tab2 = st.tabs(["Access Existing Index", "Create New Index"])
    
    with tab1:
        st.subheader("Access Your Index")
        with st.form("returning_user_form"):
            user_code = st.text_input("Enter Your Access Code", 
                                   placeholder="e.g., A1B2C3D4",
                                   help="Enter the code you received when you created your index")
            submit_button = st.form_submit_button("Access Index")
            
            if submit_button and user_code:
                try:
                    # Check if the user exists
                    user_manager = UserManager()
                    user_config = user_manager.get_user_index(user_code)
                    
                    if not user_config:
                        st.error("Invalid access code. Please check and try again.")
                    else:
                        # Initialize vector store and check if index exists
                        vector_store = PineconeStore(user_code=user_code)
                        if not vector_store.ensure_index_exists():
                            st.error("Index not found. Please create a new index.")
                        else:
                            st.session_state.user_code = user_code
                            st.session_state.index_name = user_config["index_name"]
                            st.session_state.show_chat = True
                            st.rerun()
                except Exception as e:
                    st.error(f"Error accessing index: {str(e)}")
    
    with tab2:
        st.subheader("Create New Index")
        with st.form("new_user_form"):
            index_name = st.text_input("Index Name", 
                                     placeholder="e.g., my-documents",
                                     help="Choose a unique name for your index")
            access_code = st.text_input("Choose an Access Code", 
                                      placeholder="e.g., MYCODE123", 
                                      help="Create a code to access this index later")
            
            submit_button = st.form_submit_button("Create Index")
            
            if submit_button:
                if not index_name or not access_code:
                    st.error("Please provide both an index name and an access code")
                else:
                    try:
                        user_manager = UserManager()
                        
                        # Check if access code already exists
                        try:
                            existing_config = user_manager.get_user_index(access_code)
                            if existing_config:
                                st.error("This access code is already in use. Please choose a different one.")
                                return
                        except:
                            pass  # Access code doesn't exist yet, which is what we want
                        
                        # Create the user index
                        result = user_manager.create_user_index(index_name, access_code)
                        
                        # Create the Pinecone index
                        vector_store = PineconeStore(user_code=access_code)
                        vector_store.create_index()
                        
                        st.session_state.user_code = access_code
                        st.session_state.index_name = result["index_name"]
                        st.session_state.show_chat = True
                        
                        st.success("Index created successfully!")
                        st.info(f"Your access code is: **{access_code}**")
                        st.info("You can now upload documents and start querying your index.")
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating index: {str(e)}")

def show_chat_interface():
    """Display the main chat interface."""
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("streamlit_app")
    
    logger.info(f"Showing chat interface for user_code={st.session_state.user_code}, index_name={st.session_state.index_name}")
    logger.info(f"Session state keys: {list(st.session_state.keys())}")
    
    st.title(f"📚 {st.session_state.index_name}")
    st.caption(f"User Code: {st.session_state.user_code}")
    
    # Display chat messages from history on app rerun
    logger.info(f"Chat history has {len(st.session_state.messages)} messages")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # File upload and chat interface
    with st.sidebar:
        st.header("Document Management")
        handle_file_upload()
        
        st.divider()
        if st.button("Clear Chat History"):
            logger.info("Clearing chat history")
            st.session_state.messages = []
            st.rerun()
        
        if st.button("Switch User"):
            logger.info("Switching user - resetting session state")
            st.session_state.show_chat = False
            st.session_state.user_code = None
            st.session_state.index_name = None
            st.session_state.messages = []
            st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        logger.info(f"Received chat input: {prompt[:50]}...")
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        logger.info(f"Added user message to history. New length: {len(st.session_state.messages)}")
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            logger.info("Starting streaming response")
            # Stream the response
            try:
                # We need to adapt our async generator to work in Streamlit's environment
                # First, create a synchronous wrapper function to get chunks from the async generator
                async def get_response_chunks():
                    # Get the generator
                    generator = run_agent_with_streaming(prompt)
                    chunks = []
                    # Collect all chunks
                    async for chunk in generator:
                        if chunk:
                            chunks.append(chunk)
                    return chunks
                
                # Use asyncio.run_coroutine_threadsafe or just get all chunks at once
                # Since we're in Streamlit, we'll use a simpler approach
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    chunks = loop.run_until_complete(get_response_chunks())
                finally:
                    loop.close()
                
                # Now process the chunks synchronously
                for chunk in chunks:
                    logger.info(f"Processing chunk: {chunk[:50] if chunk else 'Empty'}...")
                    if chunk:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                
                # Update with final response
                logger.info(f"Final response length: {len(full_response)}")
                response_placeholder.markdown(full_response)
                
                # Add the assistant's response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                logger.info(f"Added assistant response to history. New length: {len(st.session_state.messages)}")
                
            except Exception as e:
                error_msg = f"Error streaming response: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
        
        # Display sources after response
        logger.info("Displaying sources")
        display_sources()
        
        # Rerun to update the UI
        logger.info("Rerunning Streamlit app to update UI")
        st.rerun()

def main():
    """Main function for the Streamlit app."""
    # Set page config
    st.set_page_config(
        page_title="RAG AI Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add some custom CSS for better styling
    st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        margin: 5px 0;
    }
    .stTextInput>div>div>input {
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Show either the user management or chat interface
    if not st.session_state.show_chat:
        show_user_management()
    else:
        # Verify the user still has access to the index
        try:
            vector_store = PineconeStore(user_code=st.session_state.user_code)
            if not vector_store.ensure_index_exists():
                st.error("Your index is no longer available. Please create a new one.")
                st.session_state.show_chat = False
                st.rerun()
            show_chat_interface()
        except Exception as e:
            st.error(f"Error accessing your index: {str(e)}")
            st.session_state.show_chat = False
            st.rerun()

if __name__ == "__main__":
    main()
