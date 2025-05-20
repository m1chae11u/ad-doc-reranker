import os
import re
import json
import google.generativeai as genai
from retriever import AdSiteRetriever
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

'''
retrieves top k ads and gives them to LLM which generates responses to query
'''
class RAGGenerator:
    def __init__(self, retriever=None):
        self.api_key = self.load_api_key()
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        self.retriever = retriever  # optional shared retriever

    @staticmethod
    def load_api_key():
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "keys.json")
        with open(config_path, 'r') as f:
            return json.load(f)["google_api_key"]

    def generate_response(self, query: str, context: str):
        prompt = f"""
You are a helpful and knowledgeable assistant. Below is a list of product advertisements. Your task is to respond to the user's query in a natural and informative way, promoting one or two relevant products. Include brand and url into your response without revealing that you have access to product ads. 

At the end of your response (not visible to the user), list which documents you included clearly only using the format: 'id: ..., id: ..., ...'.

USER QUERY: {query}

RETRIEVED COMMERCIAL ADS:
{context}

Please provide a helpful, informative response directed to the user based on the above information.
"""
        llm_output = self.model.generate_content(prompt).text
        # print(llm_output)
        ids = re.findall(r'id:\s*([^\s,]+)', llm_output, re.IGNORECASE)

        # Remove the last line if it contains IDs
        lines = llm_output.strip().splitlines()
        if lines and re.search(r'id:\s*\w+', lines[-1], re.IGNORECASE):
            response_text = "\n".join(lines[:-1]).strip()
        else:
            response_text = llm_output.strip()

        return response_text, ids

    def generate_single(self, item: dict, top_k: int = 3, use_full_docs: bool = True):
        """
        Generate a single response for a given query item using a shared retriever.
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized. Pass it when constructing RAGGenerator.")

        query = item["query"]
        domain = item.get("domain", "Unknown")
        subdomain = item.get("subdomain", "Unknown")

        context = self.retriever.get_relevant_context(query, use_full_docs=use_full_docs)
        response, docs_in_response = self.generate_response(query, context)

        return {
            "query": query,
            "domain": domain,
            "subdomain": subdomain,
            "retrieved_context": context,
            "response": response,
            "documents_in_response": docs_in_response
        }

    def batch_generate(self, query_file: str, index_dir: str, output_file: str, top_k: int = 3, use_full_docs: bool = True, original_file: str = None, max_workers: int = 25):
        # Load queries
        with open(query_file, 'r', encoding='utf-8') as f:
            queries = json.load(f)

        # Init retriever only once
        self.retriever = AdSiteRetriever(index_dir=index_dir, top_k=top_k, original_file=original_file)

        responses = []

        print(f"Processing {len(queries)} queries in parallel using {max_workers} workers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query = {executor.submit(self.generate_single, item, top_k, use_full_docs): item for item in queries}

            for future in tqdm(as_completed(future_to_query), total=len(queries), desc="Processing queries"):
                try:
                    result = future.result()
                    responses.append(result)
                except Exception as e:
                    print(f"Error processing query {future_to_query[future]['query']}: {e}")

        # Save to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(responses, f, ensure_ascii=False, indent=2)

        print(f"\nSaved {len(responses)} query-response pairs to {output_file}")

if __name__ == "__main__":
    # generator = RAGGenerator()
    
    # generator.batch_generate(
    #     query_file="10_queries.json",
    #     index_dir="ds/10_faiss_index",
    #     output_file="10_query_responses_original.json",
    #     top_k=10,
    #     use_full_docs=True,
    #     original_file="ds/10_sampled_ads.json"
    # )    

    # generator = RAGGenerator()
    
    # generator.batch_generate(
    #     query_file="train_queries.json",
    #     index_dir="ds/faiss_index_train",
    #     output_file="query_responses_original.json",
    #     top_k=10,
    #     use_full_docs=True,
    #     original_file="ds/train_data.json"
    # )

    generator = RAGGenerator()
    
    generator.batch_generate(
        query_file="test_queries.json",
        index_dir="ds/faiss_index_test",
        output_file="test20_query_responses_original.json",
        top_k=20,
        use_full_docs=True,
        original_file="ds/test_data.json"
    )
    generator.batch_generate(
        query_file="test_queries.json",
        index_dir="ds/faiss_index_test",
        output_file="test30_query_responses_original.json",
        top_k=30,
        use_full_docs=True,
        original_file="ds/test_data.json"
    )