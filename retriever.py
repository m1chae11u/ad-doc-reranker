"""
retriever.py

This script is responsible for initializing the retrieval component for your RAG system. It performs the following tasks:
1. Loads the previously built FAISS index and document metadata from disk.
2. Instantiates a custom retriever that leverages the FAISS index.
3. Provides an interface to query the retriever and retrieve the most relevant documents for a given input.

Usage:
    python retriever.py "What are some comfortable shoes?" --index_dir /full/path/to/faiss_index_directory --full_docs

    ie. python retriever.py "What are some comfortable shoes?" --index_dir /Users/mikel/Documents/LLM-SEO-OPTIMIZER-V1/data/CommercialAdsDataset/5000_subset_train/commercial_ads_faiss_index --full_docs
"""

import argparse
import os
import json
import glob
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

class AdSiteRetriever:
    def __init__(self, index_dir: str, top_k: int = 3):
        """
        Initialize the retriever with a FAISS index.
        
        Args:
            index_dir: Full path to directory containing the FAISS index
            top_k: Number of documents to retrieve
        """
        self.top_k = top_k
        
        # Load the embedding model (same as used for building the index)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Use the provided full path directly
        self.index_path = index_dir
        
        print(f"Loading FAISS index from {self.index_path}")
        # Added allow_dangerous_deserialization flag to fix the error
        self.db = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
        
        # Load the original documents for full document retrieval
        self.original_docs = {}
        
        # Look for JSON files in the parent directory of the index
        parent_dir = os.path.dirname(self.index_path)
        json_files = glob.glob(os.path.join(parent_dir, "*.json"))
        
        if json_files:
            # Use the first JSON file found in the directory
            self.original_docs_path = json_files[0]
            print(f"Found source data file: {self.original_docs_path}")
            
            try:
                with open(self.original_docs_path, 'r', encoding='utf-8') as f:
                    documents = json.load(f)
                    # Create a lookup by ad_id for quick access
                    for doc in documents:
                        if 'ad_id' in doc and doc['ad_id']:
                            self.original_docs[doc['ad_id']] = doc
                print(f"Loaded {len(self.original_docs)} original documents")
            except Exception as e:
                print(f"Warning: Could not load original documents: {e}")
        else:
            print("Warning: No JSON files found in the index directory for full document retrieval")
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a given query.
        
        Args:
            query: The search query
            
        Returns:
            List of relevant documents (chunks)
        """
        # Search the index for similar document chunks
        chunks = self.db.similarity_search(query, k=self.top_k * 2)  # Get more chunks initially
        
        # Group by parent document and deduplicate
        seen_docs = set()
        unique_docs = []
        
        for chunk in chunks:
            parent_id = chunk.metadata.get('parent_doc_id')

            if parent_id not in seen_docs:
                seen_docs.add(parent_id)
                
                # Create a document with the parent metadata
                # but keep the specific chunk content that matched
                doc = Document(
                    page_content=chunk.page_content,
                    metadata={
                        'url': chunk.metadata.get('url'),
                        'seller': chunk.metadata.get('seller', ''),
                        'source': chunk.metadata.get('source'),
                        'matched_chunk': True,
                        'chunk_id': chunk.metadata.get('chunk_id'),
                        'parent_doc_id': parent_id
                    }
                )
                unique_docs.append(doc)
                
                # Stop once we have enough unique parent documents
                if len(unique_docs) >= self.top_k:
                    break
        
        return unique_docs
    
    def retrieve_full_documents(self, query: str) -> List[Document]:
        """
        Retrieve the full original documents corresponding to the chunks that match the query.
        
        Args:
            query: The search query
            
        Returns:
            List of full documents
        """
        # First retrieve the chunks
        chunk_docs = self.retrieve(query)
        
        # Then get the full documents
        full_docs = []
        for chunk_doc in chunk_docs:
            doc_id = chunk_doc.metadata.get('parent_doc_id')
            if doc_id and doc_id in self.original_docs:
                # Create a document with the full content
                original_data = self.original_docs[doc_id]
                
                # Format the content nicely
                full_content = f"Title: {original_data.get('title', '')}\n\n"
                full_content += f"Description: {original_data.get('text', '')}\n\n"
                
                if 'brand' in original_data and original_data['brand']:
                    full_content += f"Brand: {original_data['brand']}\n\n"
                    
                if 'seller' in original_data and original_data['seller']:
                    full_content += f"Seller: {original_data['seller']}\n\n"
                
                full_content += f"doc_id: {doc_id}"
                
                # Create full document
                full_doc = Document(
                    page_content=full_content,
                    metadata={
                        'url': original_data.get('url', ''),
                        'title': original_data.get('title', ''),
                        'source': original_data.get('source', ''),
                        'is_full_doc': True,
                        'matched_chunk': chunk_doc.page_content,  # Include the matching chunk for context
                        'doc_id': doc_id
                    }
                )
                full_docs.append(full_doc)
                # print("appended full")
            else:
                # If we can't find the full document, just use the chunk
                full_docs.append(chunk_doc)
                # print('append chunk')
        
        return full_docs
    
    def get_relevant_context(self, query: str, use_full_docs: bool = True) -> str:
        """
        Get the relevant context for a given query.
        
        Args:
            query: The search query
            use_full_docs: Whether to retrieve full documents or just chunks
            
        Returns:
            Formatted context string
        """
        if use_full_docs:
            docs = self.retrieve_full_documents(query)
        else:
            docs = self.retrieve(query)
        
        context = ""
        for i, doc in enumerate(docs):
            context += f"\n--- Document {i+1} ---\n"
            
            # For full documents, include metadata
            if doc.metadata.get('is_full_doc'):
                context += doc.page_content
                context += f"\n\nMatching passage: {doc.metadata.get('matched_chunk')}\n"
            else:
                # For chunks, format differently
                context += f"Content: {doc.page_content}\n"
                if doc.metadata.get('doc_id'):
                    context += f"doc_id: {doc.metadata.get('doc_id')}\n"
            
            context += "\n" + "-" * 50 + "\n"
        
        return context

def main(query: str, index_dir: str, top_k: int, use_full_docs: bool = False):
    # Initialize the retriever
    retriever = AdSiteRetriever(index_dir=index_dir, top_k=top_k)
    
    # Get relevant context
    context = retriever.get_relevant_context(query, use_full_docs=use_full_docs)
    
    # Print the results
    print(f"\nQuery: {query}\n")
    print(f"Retrieval Mode: {'Full Documents' if use_full_docs else 'Document Chunks'}")
    print("\nRelevant Documents:")
    print(context)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query the ad content retriever."
    )
    parser.add_argument(
        "query",
        type=str,
        help="The search query."
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        required=True,
        help="Full path to the directory containing the FAISS index."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of documents to retrieve. Default is 3."
    )
    parser.add_argument(
        "--full_docs",
        action="store_true",
        help="Retrieve full documents instead of just chunks."
    )
    
    args = parser.parse_args()
    main(args.query, args.index_dir, args.top_k, use_full_docs=args.full_docs)

