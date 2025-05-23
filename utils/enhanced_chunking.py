import re
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

class EnhancedChunker:
    """Enhanced text chunking with improved strategies for different document types."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata using a recursive character splitter
        that respects natural boundaries like paragraphs and sentences.
        
        Args:
            text: The text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of dictionaries with text and metadata
        """
        # Clean the text by removing excessive newlines and whitespace
        text = self._clean_text(text)
        
        # Create a recursive splitter that respects natural boundaries
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            length_function=len
        )
        
        # Split the text
        chunks = splitter.split_text(text)
        
        # Create result with metadata
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "chunk_count": len(chunks)
            })
            
            result.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        
        return result
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing excessive whitespace and normalizing newlines."""
        # Replace multiple newlines with double newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def chunk_with_titles(self, sections: List[Dict[str, str]], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk text while preserving section titles.
        
        Args:
            sections: List of dictionaries with 'title' and 'content' keys
            metadata: Base metadata to attach to each chunk
            
        Returns:
            List of dictionaries with text and metadata
        """
        result = []
        
        for section in sections:
            title = section.get('title', '')
            content = section.get('content', '')
            
            if not content:
                continue
            
            # Add title to metadata
            section_metadata = metadata.copy()
            section_metadata['section_title'] = title
            
            # Chunk the section content
            section_chunks = self.chunk_text(content, section_metadata)
            result.extend(section_chunks)
        
        return result
