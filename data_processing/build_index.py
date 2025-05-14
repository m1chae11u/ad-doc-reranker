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

import json
import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class IndexBuilder:
    def __init__(self, input_path: str, output_dir: str = "faiss_index", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.input_path = input_path
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self) -> List[Document]:
        """Load documents from a JSON file."""
        with open(self.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []
        for item in data:
            page_content = f"Title: {item.get('title', '')}\n\nDescription: {item.get('text', '')}"
            if item.get('brand'):
                page_content += f"\n\nBrand: {item.get('brand')}"

            doc = Document(
                page_content=page_content,
                metadata={
                    'url': item.get('url', ''),
                    'seller': item.get('seller', ''),
                    'user_query': item.get('user_query', ''),
                    'source': item.get('source', ''),
                    'doc_id': item.get('ad_id', '')
                }
            )
            documents.append(doc)
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks for better retrieval."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunked_docs = []

        for doc in documents:
            parent_metadata = doc.metadata.copy()
            doc_id = parent_metadata['doc_id']
            chunks = text_splitter.split_text(doc.page_content)

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

    def build_index(self, documents: List[Document]):
        """Build and save FAISS index from documents."""
        chunked_docs = self.chunk_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        print(f"Building FAISS index from {len(chunked_docs)} document chunks...")
        db = FAISS.from_documents(chunked_docs, embeddings)

        full_output_dir = os.path.join(os.path.dirname(self.input_path), self.output_dir)
        os.makedirs(full_output_dir, exist_ok=True)
        db.save_local(full_output_dir)

        print(f"Index successfully built and saved to {full_output_dir}")
        return db

    def run(self):
        """Main method to run the indexing pipeline."""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        print(f"Loading documents from {self.input_path}")
        documents = self.load_documents()
        print(f"Loaded {len(documents)} documents")
        self.build_index(documents)


        for doc in documents:
            parent_metadata = doc.metadata.copy()
            doc_id = parent_metadata['doc_id']
            chunks = text_splitter.split_text(doc.page_content)

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

    def build_index(self, documents: List[Document]):
        """Build and save FAISS index from documents."""
        chunked_docs = self.chunk_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        print(f"Building FAISS index from {len(chunked_docs)} document chunks...")
        db = FAISS.from_documents(chunked_docs, embeddings)

        full_output_dir = os.path.join(os.path.dirname(self.input_path), self.output_dir)
        os.makedirs(full_output_dir, exist_ok=True)
        db.save_local(full_output_dir)

        print(f"Index successfully built and saved to {full_output_dir}")
        return db

    def run(self):
        """Main method to run the indexing pipeline."""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        print(f"Loading documents from {self.input_path}")
        documents = self.load_documents()
        print(f"Loaded {len(documents)} documents")
        self.build_index(documents)

if __name__ == "__main__":
    indexer = IndexBuilder(input_path="ds/test_data.json", output_dir="faiss_index_test")
    indexer.run()

    indexer = IndexBuilder(input_path="ds/train_data.json", output_dir="faiss_index_train")
    indexer.run()