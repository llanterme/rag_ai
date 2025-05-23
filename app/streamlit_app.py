import os
import streamlit as st
import asyncio
from pathlib import Path
from httpx import AsyncClient
import tempfile

from pydantic_ai.messages import ModelRequest, ModelResponse, PartDeltaEvent, PartStartEvent, TextPartDelta

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent import RAGAgentWrapper
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
        st.session_state.last_query = part.args.get('query', '')
    elif part.part_kind == 'tool-return' and part.tool_name == 'query_knowledge_base':
        if part.content:
            st.session_state.last_sources = part.content
    
async def run_agent_with_streaming(user_input):
    """Run the agent with streaming response."""
    async with AsyncClient() as http_client:
        agent_wrapper = RAGAgentWrapper()
        
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
        with st.expander("View Sources", expanded=False):
            sources = st.session_state.last_sources
            if isinstance(sources, list):
                for i, source in enumerate(sources):
                    st.markdown(f"**Source {i+1}**: {source.get('metadata', {}).get('source', 'Unknown')}")
                    st.markdown(f"**Relevance Score**: {source.get('metadata', {}).get('score', 'N/A'):.4f}")
                    with st.expander("View Content", expanded=False):
                        st.markdown(source.get('content', 'No content available'))
                    st.divider()

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

async def main():
    """Main function for the Streamlit app."""
    st.title("RAG AI Assistant")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("Document Management")
        handle_file_upload()
    
    # Display chat history
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)
    
    # Display sources for the last query
    display_sources()
    
    # Chat input
    user_input = st.chat_input("Ask a question about your documents...")
    
    if user_input:
        # Display user prompt
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Display assistant response with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            generator = run_agent_with_streaming(user_input)
            async for message in generator:
                full_response += message
                message_placeholder.markdown(full_response + "▌")
            
            # Final response without cursor
            message_placeholder.markdown(full_response)
        
        # Display sources after response
        display_sources()

if __name__ == "__main__":
    asyncio.run(main())
