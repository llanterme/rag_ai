#!/usr/bin/env python3
import os
import re
from pathlib import Path

def update_env_file(env_path, updates):
    """
    Update the .env file with new values.
    
    Args:
        env_path: Path to the .env file
        updates: Dictionary of key-value pairs to update
    """
    # Read the current content
    with open(env_path, 'r') as f:
        content = f.read()
    
    # Update each key-value pair
    for key, value in updates.items():
        # Check if the key exists
        pattern = re.compile(f'^{key}=.*$', re.MULTILINE)
        if pattern.search(content):
            # Update existing key
            content = pattern.sub(f'{key}={value}', content)
        else:
            # Add new key
            content += f'\n{key}={value}'
    
    # Write the updated content
    with open(env_path, 'w') as f:
        f.write(content)
    
    print(f"Updated .env file at {env_path}")
    for key, value in updates.items():
        print(f"  - {key} = {value}")

if __name__ == "__main__":
    # Path to .env file
    env_path = Path(os.path.dirname(os.path.abspath(__file__))) / '.env'
    
    # Updates to make
    updates = {
        'PINECONE_INDEX_NAME': 'redhill-school',
        'PINECONE_NAMESPACE': 'redhill'
    }
    
    update_env_file(env_path, updates)
    
    print("\nPlease restart your application for changes to take effect.")
