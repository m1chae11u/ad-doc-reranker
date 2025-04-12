import argparse
import os
import json
import google.generativeai as genai
from retriever import AdSiteRetriever
from tqdm import tqdm

'''

generates a response for every query and saves the query and responses into a new dataset 

usage: python rag.py --query_file queries.json --index_dir /full/path/to/commercial_ads_index --output_file query_responses.json --top_k 3

'''

def load_api_key():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "keys.json")
    with open(config_path, 'r') as f:
        return json.load(f)["google_api_key"]

def generate_response(query: str, context: str, model):
    prompt = f"""
You are a helpful assistant. Take a look at the products provided below. Your objective is to answer the user query without them knowing you have a set of documents while seamlessly incorporating a couple of products into your response that you think is relevant/appropriate given the user's query.

USER QUERY: {query}

RETRIEVED COMMERCIAL ADS:
{context}

Please provide a helpful, informative response directed to the user based on the above information.
"""
    response = model.generate_content(prompt)
    return response.text.strip()

def batch_generate(query_file: str, index_dir: str, output_file: str, top_k: int = 3, use_full_docs: bool = True):
    # Load queries
    with open(query_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    # Init retriever + model
    retriever = AdSiteRetriever(index_dir=index_dir, top_k=top_k)
    api_key = load_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    responses = []

    for item in tqdm(queries, desc="Processing queries"):
        query = item["query"]
        domain = item.get("domain", "Unknown")
        subdomain = item.get("subdomain", "Unknown")

        # Retrieve context and generate response
        context = retriever.get_relevant_context(query, use_full_docs=use_full_docs)
        response = generate_response(query, context, model)

        responses.append({
            "query": query,
            "domain": domain,
            "subdomain": subdomain,
            "retrieved_context": context,
            "response": response
        })

    # Save to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(responses)} query-response pairs to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch generate responses for ad-related queries using RAG.")
    parser.add_argument("--query_file", type=str, required=True, help="Path to the queries JSON file.")
    parser.add_argument("--index_dir", type=str, required=True, help="Path to the FAISS index directory.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the generated responses.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top documents to retrieve.")
    parser.add_argument("--chunks_only", action="store_true", help="Use document chunks instead of full documents.")

    args = parser.parse_args()
    batch_generate(
        query_file=args.query_file,
        index_dir=args.index_dir,
        output_file=args.output_file,
        top_k=args.top_k,
        use_full_docs=not args.chunks_only
    )
