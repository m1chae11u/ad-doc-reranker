import argparse
import os
import re
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
You are a helpful and knowledgeable assistant. Below is a list of product advertisements. Your task is to respond to the user's query in a natural and informative way,  promoting one or two relevant products. Include brand and url into your response without revealing that you have access to product ads. 

At the end of your response (not visible to the user), list which documents you included clearly only using the format: 'id: ..., id: ..., ...'.

USER QUERY: {query}

RETRIEVED COMMERCIAL ADS:
{context}

Please provide a helpful, informative response directed to the user based on the above information.
"""
    llm_output = model.generate_content(prompt).text
    # print (prompt)
    print (llm_output)
    ids = re.findall(r'id:\s*([^\s,]+)', llm_output, re.IGNORECASE)

    # Remove the last line if it contains IDs
    lines = llm_output.strip().splitlines()
    if lines and re.search(r'id:\s*\w+', lines[-1], re.IGNORECASE):
        response_text = "\n".join(lines[:-1]).strip()
    else:
        response_text = llm_output.strip()

    return response_text, ids

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
        response, docs_in_response = generate_response(query, context, model)

        responses.append({
            "query": query,
            "domain": domain,
            "subdomain": subdomain,
            "retrieved_context": context,
            "response": response,
            "documents_in_response": docs_in_response
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
