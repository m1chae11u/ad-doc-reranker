import os
import re
import json
import google.generativeai as genai
from retriever import AdSiteRetriever
from tqdm import tqdm

class RAGGenerator:
    def __init__(self):
        self.api_key = self.load_api_key()
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

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
        print(llm_output)
        ids = re.findall(r'id:\s*([^\s,]+)', llm_output, re.IGNORECASE)

        # Remove the last line if it contains IDs
        lines = llm_output.strip().splitlines()
        if lines and re.search(r'id:\s*\w+', lines[-1], re.IGNORECASE):
            response_text = "\n".join(lines[:-1]).strip()
        else:
            response_text = llm_output.strip()

        return response_text, ids

    def batch_generate(self, query_file: str, index_dir: str, output_file: str, top_k: int = 3, use_full_docs: bool = True, original_file: str = None):
        # Load queries
        with open(query_file, 'r', encoding='utf-8') as f:
            queries = json.load(f)

        # Init retriever with optional original file
        retriever = AdSiteRetriever(index_dir=index_dir, top_k=top_k, original_file=original_file)

        responses = []

        for item in tqdm(queries, desc="Processing queries"):
            query = item["query"]
            domain = item.get("domain", "Unknown")
            subdomain = item.get("subdomain", "Unknown")

            # Retrieve context and generate response
            context = retriever.get_relevant_context(query, use_full_docs=use_full_docs)
            response, docs_in_response = self.generate_response(query, context)

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

# if __name__== "__main__":
#     generator = RAGGenerator()

#     generator.batch_generate(
#         query_file="queries_200.json",
#         index_dir="ds/faiss_index",
#         output_file="query_responses.json",
#         top_k=10,
#         use_full_docs=True,
#         original_file="200_sampled_ads.json"
#     )
