import os
import streamlit as st
import asyncio
from pathlib import Path
from httpx import AsyncClient
import tempfile
import time

from pydantic_ai.messages import ModelRequest, ModelResponse, PartDeltaEvent, PartStartEvent, TextPartDelta

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.advanced_agent import AdvancedRAGAgentWrapper
from ingest import ingest_documents
from config import DATA_DIR

# Create a temporary directory for uploaded files
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    """
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
        if isinstance(part.args, dict):
            st.session_state.last_query = part.args.get('query', '')
        elif isinstance(part.args, str):
            st.session_state.last_query = part.args
    elif part.part_kind == 'tool-return' and part.tool_name == 'query_knowledge_base':
        if part.content:
            st.session_state.last_sources = part.content
    elif part.part_kind == 'tool-call' and part.tool_name == 'refine_search':
        if isinstance(part.args, dict):
            st.session_state.last_refined_query = part.args.get('refined_query', '')
        elif isinstance(part.args, str):
            st.session_state.last_refined_query = part.args
    elif part.part_kind == 'tool-return' and part.tool_name == 'refine_search':
        if part.content:
            # Add to existing sources or create new list
            if hasattr(st.session_state, 'last_sources') and st.session_state.last_sources:
                st.session_state.last_sources.extend(part.content)
            else:
                st.session_state.last_sources = part.content
    
async def run_agent_with_streaming(user_input):
    """Run the agent with streaming response."""
    async with AsyncClient() as http_client:
        agent_wrapper = AdvancedRAGAgentWrapper()
        
        async with agent_wrapper.agent.iter(user_input, message_history=st.session_state.messages) as run:
            async for node in run:
                if agent_wrapper.agent.is_model_request_node(node):
                    # Stream tokens from the model's request
                    async with node.stream(run.ctx) as request_stream:
                        async for event in request_stream:
                            if isinstance(event, PartStartEvent) and event.part.part_kind == 'text':
                                yield event.part.content
                            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                delta = event.delta.content_delta
                                yield delta
        
        # Add the new messages to the chat history
        st.session_state.messages.extend(run.result.new_messages())

def display_sources():
    """Display source attribution for the last query."""
    if hasattr(st.session_state, 'last_sources') and st.session_state.last_sources:
        sources = st.session_state.last_sources
        if isinstance(sources, list):
            # Group sources by filename
            sources_by_file = {}
            for source in sources:
                filename = source.get('metadata', {}).get('source', 'Unknown')
                if filename not in sources_by_file:
                    sources_by_file[filename] = []
                sources_by_file[filename].append(source)
            
            # Create a main expander for all sources
            with st.expander("📚 View Sources", expanded=False):
                # Display sources grouped by file
                for filename, file_sources in sources_by_file.items():
                    st.markdown(f"### 📄 {filename}")
                    for i, source in enumerate(file_sources):
                        score = source.get('metadata', {}).get('score', 'N/A')
                        section = source.get('metadata', {}).get('section_title', '')
                        section_info = f" - Section: **{section}**" if section else ""
                        
                        # Display excerpt info and content directly without nested expander
                        st.markdown(f"**Excerpt {i+1}**{section_info} (Relevance: {score:.4f})")
                        st.markdown("```")
                        st.markdown(source.get('content', 'No content available'))
                        st.markdown("```")
                    st.divider()
                
                # If a refined query was used, show it
                if hasattr(st.session_state, 'last_refined_query') and st.session_state.last_refined_query:
                    st.markdown(f"**Refined Query:** {st.session_state.last_refined_query}")

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
            
            # Ingest documents
            num_ingested = ingest_documents(file_paths=temp_files)
            
            if num_ingested > 0:
                st.success(f"Successfully ingested {num_ingested} new documents!")
            else:
                st.info("No new documents were ingested. They may have been processed already.")

def display_chat_history():
    """Display the chat history."""
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

def display_app_info():
    """Display information about the app."""
    st.markdown("""
    ### About this RAG AI Assistant
    
    This application uses Retrieval-Augmented Generation (RAG) to provide accurate answers based on your documents.
    
    **Features:**
    - Upload PDF and TXT files
    - Ask questions about your documents
    - Get AI-generated answers with source attribution
    - View the source documents used to generate answers
    
    **How it works:**
    1. Upload your documents
    2. Ask a question
    3. The system retrieves relevant information from your documents
    4. The AI generates an answer based on the retrieved information
    
    **Tips for best results:**
    - Ask specific questions
    - Upload high-quality documents
    - Check the sources to verify information
    """)

async def main():
    """Main function for the Streamlit app."""
    st.set_page_config(
        page_title="RAG AI Assistant",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 RAG AI Assistant")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Create two columns: sidebar and main content
    col1, col2 = st.columns([1, 3])
    
    # Sidebar for document upload and app info
    with col1:
        st.header("Document Management")
        handle_file_upload()
        
        st.divider()
        display_app_info()
    
    # Main content for chat
    with col2:
        # Chat input at the top
        st.subheader("Ask a question")
        user_input = st.chat_input("Ask a question about your documents...")
        
        # Process user input
        if user_input:
            # Add a spinner while waiting for the response
            with st.spinner("Thinking..."):
                # Process the user input and get the response
                generator = run_agent_with_streaming(user_input)
                full_response = ""
                async for message in generator:
                    full_response += message
                # The messages are automatically added to session state in run_agent_with_streaming
        
        # Display conversation history (will include the latest messages)
        st.subheader("Conversation History")
        display_chat_history()
        
        # Display sources for the last query
        display_sources()

if __name__ == "__main__":
    asyncio.run(main())
