"""
build_index.py

This script handles the data preparation and indexing process. It performs the following tasks:
1. Loads raw documents (each representing an advertisement) from our data source.
2. Computes embeddings for each document using the selected embedding model.
3. Constructs a FAISS index from the computed embeddings.
4. Saves the FAISS index and associated document metadata to disk

Usage:
    python data_processing/build_index.py --input_path /full/path/to/file --output_dir faiss_index
"""

import argparse
import json
import os
import uuid
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(file_path: str) -> List[Document]:
    """
    Load documents from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of LangChain Document objects
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    for item in data:
        # Combining title and description for richer context
        page_content = f"Title: {item.get('title', '')}\n\nDescription: {item.get('text', '')}"
        if item.get('brand'):
            page_content += f"\n\nBrand: {item.get('brand')}"

        # Create a LangChain Document object for each website
        doc = Document(
            page_content=page_content,
            metadata={
                'url': item.get('url', ''),
                'seller': item.get('seller', ''),
                'user_query': item.get('user_query', ''),
                'source': item.get('source', ''), 
                'doc_id': str(uuid.uuid4())  # Unique ID to track document chunks
            }
        )
        documents.append(doc)
    
    return documents

def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        documents: List of original documents
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of chunked documents with parent document metadata
    """
    # Configure the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,       
        chunk_overlap=chunk_overlap,     
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Try to split on paragraphs first
    )
    
    chunked_docs = []
    
    for doc in documents:
        # Save original document metadata
        parent_metadata = doc.metadata.copy()
        doc_id = parent_metadata['doc_id']
        
        # Split the document into chunks
        chunks = text_splitter.split_text(doc.page_content)
        
        # Create a new document for each chunk
        for i, chunk_text in enumerate(chunks):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    **parent_metadata,
                    'chunk_id': i,
                    'chunk_count': len(chunks),
                    'is_chunk': True,
                    'parent_doc_id': doc_id
                }
            )
            chunked_docs.append(chunk_doc)
            
    print(f"Split {len(documents)} documents into {len(chunked_docs)} chunks")
    return chunked_docs

def build_index(documents: List[Document], output_dir: str):
    """
    Build a FAISS index from the documents.
    
    Args:
        documents: List of LangChain Document objects
        output_dir: Directory to save the index
    """
    # First chunk the documents
    chunked_docs = chunk_documents(documents)
    
    # Initialize the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create the FAISS index from chunks
    print(f"Building FAISS index from {len(chunked_docs)} document chunks...")
    db = FAISS.from_documents(chunked_docs, embeddings)
    
    # Save the index to disk
    os.makedirs(output_dir, exist_ok=True)
    db.save_local(output_dir)
    
    print(f"Index successfully built and saved to {output_dir}")
    return db

def main(input_path: str, output_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Main function to run the indexing process.
    
    Args:
        input_path: Full path to the input JSON file
        output_dir: Directory name for the output index 
                   (will be created in the same directory as the input file)
        chunk_size: Character length of each document chunk
        chunk_overlap: Character overlap between chunks
    """
    # Ensure input path exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Get the directory of the input file
    input_dir = os.path.dirname(input_path)
    
    # Create full path for output directory in the same location as input file
    index_path = os.path.join(input_dir, output_dir)
    
    print(f"Loading documents from {input_path}")
    documents = load_documents(input_path)
    print(f"Loaded {len(documents)} documents")
    
    # Build the index
    build_index(documents, index_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from advertisements dataset."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Full path to the input JSON file containing the advertisements."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="faiss_index",
        help="Directory name for saving the FAISS index. Will be created in the same directory as the input file. Default is 'faiss_index'."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Target chunk size in characters. Default is 1000."
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters. Default is 200."
    )
    
    args = parser.parse_args()
    main(args.input_path, args.output_dir, args.chunk_size, args.chunk_overlap)