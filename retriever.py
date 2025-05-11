import argparse
import os
import json
import glob
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

"""
retriever.py

This script is responsible for initializing the retrieval component for your RAG system. It performs the following tasks:
1. Loads the previously built FAISS index and document metadata from disk.
2. Instantiates a custom retriever that leverages the FAISS index.
3. Provides an interface to query the retriever and retrieve the most relevant documents for a given input.

Usage:
    python retriever.py "What are some comfortable shoes?" --index_dir ds/faiss_index --original_file 200_sampled_ads.json --full_docs
"""

class AdSiteRetriever:
    def __init__(self, index_dir: str, top_k: int = 3, original_file: str = None):
        """
        Initialize the retriever with a FAISS index.
        
        Args:
            index_dir: Full path to directory containing the FAISS index
            top_k: Number of documents to retrieve
            original_file: Optional path to JSON file with original documents
        """
        self.top_k = top_k
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.index_path = index_dir

        print(f"Loading FAISS index from {self.index_path}")
        self.db = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
        self.original_docs = {}

        # Determine source file
        if original_file and os.path.isfile(original_file):
            self.original_docs_path = original_file
            print(f"Found source data file: {self.original_docs_path}")
        else:
            parent_dir = os.path.dirname(self.index_path)
            json_files = glob.glob(os.path.join(parent_dir, "*.json"))
            self.original_docs_path = json_files[0] if json_files else None
        
        with open(self.original_docs_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
            # Create a lookup by ad_id for quick access
            for doc in documents:
                if 'ad_id' in doc and doc['ad_id']:
                    self.original_docs[doc['ad_id']] = doc
            print(f"Loaded {len(self.original_docs)} original documents")

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a given query.
        """
        chunks = self.db.similarity_search(query, k=self.top_k * 3)
        seen_docs = set()
        unique_docs = []

        for chunk in chunks:
            parent_id = chunk.metadata.get('parent_doc_id')
            if parent_id not in seen_docs:
                seen_docs.add(parent_id)
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
                if len(unique_docs) >= self.top_k:
                    break
        return unique_docs

    def retrieve_full_documents(self, query: str) -> List[Document]:
        """
        Retrieve the full original documents corresponding to the chunks that match the query.
        """
        chunk_docs = self.retrieve(query)
        full_docs = []

        for chunk_doc in chunk_docs:
            doc_id = chunk_doc.metadata.get('parent_doc_id')
            if doc_id and doc_id in self.original_docs:
                original_data = self.original_docs[doc_id]
                full_content = f"Title: {original_data.get('title', '')}\n\n"
                full_content += f"Description: {original_data.get('text', '')}\n\n"
                if 'brand' in original_data and original_data['brand']:
                    full_content += f"Brand: {original_data['brand']}\n\n"
                if 'seller' in original_data and original_data['seller']:
                    full_content += f"Seller: {original_data['seller']}\n\n"
                full_content += f"url: {original_data['url']}"
                full_doc = Document(
                    page_content=full_content,
                    metadata={
                        'url': original_data.get('url', ''),
                        'title': original_data.get('title', ''),
                        'source': original_data.get('source', ''),
                        'is_full_doc': True,
                        'matched_chunk': chunk_doc.page_content,
                        'doc_id': doc_id
                    }
                )
                full_docs.append(full_doc)
            else:
                full_docs.append(chunk_doc)
        return full_docs

    def get_relevant_context(self, query: str, use_full_docs: bool = True) -> str:
        """
        Get the relevant context for a given query.
        """
        docs = self.retrieve_full_documents(query) if use_full_docs else self.retrieve(query)
        context = ""

        for i, doc in enumerate(docs):
            context += f"\n--- Document {i+1} ---\n"
            if doc.metadata.get('is_full_doc'):
                context += doc.page_content
                context += f"\n\nid: {doc.metadata.get('doc_id')}\n"
            else:
                context += f"Content: {doc.page_content}\n"
                if doc.metadata.get('doc_id'):
                    context += f"doc_id: {doc.metadata.get('doc_id')}\n"
            context += "\n" + "-" * 50 + "\n"
        return context

def main(query: str, index_dir: str, top_k: int, use_full_docs: bool = False, original_file: str = None):
    retriever = AdSiteRetriever(index_dir=index_dir, top_k=top_k, original_file=original_file)
    context = retriever.get_relevant_context(query, use_full_docs=use_full_docs)
    print(f"\nQuery: {query}\n")
    print(f"Retrieval Mode: {'Full Documents' if use_full_docs else 'Document Chunks'}")
    print("\nRelevant Documents:")
    print(context)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the ad content retriever.")
    parser.add_argument("query", type=str, help="The search query.")
    parser.add_argument("--index_dir", type=str, required=True, help="Full path to the directory containing the FAISS index.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of documents to retrieve. Default is 3.")
    parser.add_argument("--full_docs", action="store_true", help="Retrieve full documents instead of just chunks.")
    parser.add_argument("--original_file", type=str, default=None, help="Optional path to the original document JSON file.")

    args = parser.parse_args()
    main(args.query, args.index_dir, args.top_k, use_full_docs=args.full_docs, original_file=args.original_file)
