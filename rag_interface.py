"""
rag_interface.py

This script provides a complete RAG (Retrieval-Augmented Generation) interface.
It retrieves relevant commercial advertisement content and uses a language model to generate responses.

Usage:
    python rag_interface.py --query "What are good products for cleaning hardwood floors?" --index_dir /full/path/to/commercial_ads_index

    ie. python rag_interface.py --query "What are effective cleaning products?" --index_dir /Users/mikel/Documents/LLM-SEO-OPTIMIZER-V1/data/CommercialAdsDataset/5000_subset_train/commercial_ads_faiss_index
"""

import argparse
import os
import google.generativeai as genai
from retriever import BusinessSiteRetriever

# here im using a keys.json file to store the google api key
# you can use your own api key or set it in the environment variable GOOGLE_API_KEY
# you also don't need to use google's gemini model, you can use any other model
def load_api_key() -> str:
    """Load Google API key from the config file."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "keys.json")
    
    with open(config_path, 'r') as f:
        import json
        config = json.load(f)
    
    return config["google_api_key"]

def generate_response(query: str, context: str, model):
    """
    Generate a response using Google's Gemini model.
    
    Args:
        query: The user's query
        context: Retrieved context from commercial ads
        model: Gemini model instance
        
    Returns:
        Generated response
    """
    prompt = f"""
You are a helpful assistant. Take a look at the advertisements provided below. Your objective is to answer the user query while seamlessly incorporating advertisements into your response that you think is relevant/appropriate given the user's query.

USER QUERY: {query}

RETRIEVED COMMERCIAL ADS:
{context}

Please provide a helpful, informative response directed to the userbased on the above information.
"""
    
    response = model.generate_content(prompt)
    return response.text

def main(query: str, index_dir: str, top_k: int, use_full_docs: bool = True):
    # Initialize the retriever
    retriever = BusinessSiteRetriever(index_dir=index_dir, top_k=top_k)
    
    # Get relevant context - now using full documents by default
    context = retriever.get_relevant_context(query, use_full_docs=use_full_docs)
    
    # Initialize Gemini
    api_key = load_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Generate response
    response = generate_response(query, context, model)
    
    # Print the results
    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print("=" * 80)
    print(f"\nRetrieval Mode: {'Full Documents' if use_full_docs else 'Document Chunks'}")
    print("\nRetrieved Context:")
    print("-" * 80)
    print(context)
    print("=" * 80)
    print("\nGenerated Response:")
    print("-" * 80)
    print(response)
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query the RAG system with commercial advertisement data."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The question to ask about products or services."
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
        "--chunks_only",
        action="store_true",
        help="Use only document chunks instead of full documents."
    )
    
    args = parser.parse_args()
    main(args.query, args.index_dir, args.top_k, use_full_docs=not args.chunks_only)
