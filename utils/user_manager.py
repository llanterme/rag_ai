"""
User Manager for handling user-specific indexes.
"""
import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

from config import DATA_DIR

# File to store user index mappings
USER_INDEX_MAPPING_FILE = os.path.join(DATA_DIR, "user_index_mapping.json")

class UserManager:
    """Manages user indexes and their configurations."""
    
    def __init__(self):
        self.user_mapping = self._load_user_mapping()
        os.makedirs(DATA_DIR, exist_ok=True)
    
    def _load_user_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Load the user to index mapping from disk."""
        if os.path.exists(USER_INDEX_MAPPING_FILE):
            try:
                with open(USER_INDEX_MAPPING_FILE, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}
    
    def _save_user_mapping(self):
        """Save the user to index mapping to disk."""
        with open(USER_INDEX_MAPPING_FILE, 'w') as f:
            json.dump(self.user_mapping, f, indent=2)
    
    def is_valid_access_code(self, code: str) -> bool:
        """Check if an access code is valid (alphanumeric and reasonable length)."""
        import re
        return bool(re.match(r'^[A-Z0-9]{4,20}$', code.upper()))
    
    def create_user_index(self, index_name: str, access_code: str) -> Dict[str, str]:
        """
        Create a new user index with the provided access code.
        
        Args:
            index_name: Name for the new index
            access_code: User-provided access code (must be unique and valid)
            
        Returns:
            Dict containing user_code and index_name
        """
        # Validate access code format
        if not self.is_valid_access_code(access_code):
            raise ValueError("Access code must be 4-20 characters long and contain only letters and numbers")
            
        # Normalize the access code
        user_code = access_code.upper()
            
        # Check if access code already exists
        if user_code in self.user_mapping:
            raise ValueError("Access code already in use. Please choose a different one.")
        
        # Clean and validate index name
        clean_index_name = index_name.lower().strip().replace(" ", "-")
        if not clean_index_name:
            raise ValueError("Index name cannot be empty")
            
        # Store the mapping with the user's chosen access code
        self.user_mapping[user_code] = {
            "index_name": clean_index_name,
            "document_chunks_file": f"document_chunks_{user_code}.json",
            "ingested_files_file": f"ingested_files_{user_code}.json"
        }
        
        # Save the mapping
        self._save_user_mapping()
        
        return {
            "user_code": user_code,
            "index_name": index_name.lower().replace(" ", "-")
        }
    
    def get_user_index(self, user_code: str) -> Optional[Dict[str, Any]]:
        """
        Get the index configuration for a user code.
        
        Args:
            user_code: The user's access code (case-insensitive)
            
        Returns:
            Dictionary with user configuration or None if not found
        """
        return self.user_mapping.get(user_code.upper())
    
    def get_document_chunks_file(self, user_code: str) -> str:
        """
        Get the path to the user's document chunks file.
        
        Args:
            user_code: The user's access code (case-insensitive)
            
        Returns:
            Full path to the document chunks file
        """
        user_config = self.user_mapping.get(user_code.upper(), {})
        return os.path.join(DATA_DIR, user_config.get("document_chunks_file", f"document_chunks_{user_code.upper()}.json"))
    
    def get_ingested_files_file(self, user_code: str) -> str:
        """
        Get the path to the user's ingested files file.
        
        Args:
            user_code: The user's access code (case-insensitive)
            
        Returns:
            Full path to the ingested files index
        """
        user_config = self.user_mapping.get(user_code.upper(), {})
        return os.path.join(DATA_DIR, user_config.get("ingested_files_file", f"ingested_files_{user_code.upper()}.json"))
