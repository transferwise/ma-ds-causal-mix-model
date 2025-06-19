#!/usr/bin/env python3
"""
Script to add 'parameters' tag to the first cell of a Jupyter notebook.
This is required for papermill to properly inject parameters.
"""

import json
import sys
from pathlib import Path

def add_parameters_tag(notebook_path):
    """Add 'parameters' tag to the first cell of a notebook."""
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Check if there are cells
    if not notebook.get('cells'):
        print("No cells found in notebook")
        return False
    
    # Get the first cell
    first_cell = notebook['cells'][0]
    
    # Initialize metadata if it doesn't exist
    if 'metadata' not in first_cell:
        first_cell['metadata'] = {}
    
    # Initialize tags if they don't exist
    if 'tags' not in first_cell['metadata']:
        first_cell['metadata']['tags'] = []
    
    # Add 'parameters' tag if not already present
    if 'parameters' not in first_cell['metadata']['tags']:
        first_cell['metadata']['tags'].append('parameters')
        print("✅ Added 'parameters' tag to first cell")
    else:
        print("ℹ️  'parameters' tag already exists in first cell")
    
    # Write back to file
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"✅ Notebook updated: {notebook_path}")
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python fix_parameters_tag.py <notebook_path>")
        sys.exit(1)
    
    notebook_path = Path(sys.argv[1])
    
    if not notebook_path.exists():
        print(f"Error: Notebook not found: {notebook_path}")
        sys.exit(1)
    
    if not notebook_path.suffix == '.ipynb':
        print(f"Error: File is not a Jupyter notebook: {notebook_path}")
        sys.exit(1)
    
    success = add_parameters_tag(notebook_path)
    
    if success:
        print("\n🎉 Your notebook is now ready for papermill parameter injection!")
        print("\nThe first cell is now tagged as 'parameters' and papermill will:")
        print("1. Find this cell by its 'parameters' tag")
        print("2. Replace the demo values with actual parameters")
        print("3. Execute the notebook with your real values")
    else:
        print("❌ Failed to update notebook")
        sys.exit(1)

if __name__ == "__main__":
    main()
