import os
import re
import json
import google.generativeai as genai
from retriever import AdSiteRetriever
from tqdm import tqdm

class RAGGenerator:
    def __init__(self, index_dir, top_k=3, use_full_docs=True):
        """
        Args:
            index_dir (str): Path to FAISS index directory.
            top_k (int): Number of top documents to retrieve per query.
            use_full_docs (bool): Whether to use full documents or chunks.
        """
        self.index_dir = index_dir
        self.top_k = top_k
        self.use_full_docs = use_full_docs
        self.retriever = AdSiteRetriever(index_dir=index_dir, top_k=top_k)
        self.model = None  # Will be initialized when needed
        self._load_model()

    def _load_model(self):
        api_key = self._load_api_key()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def _load_api_key(self):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "keys.json")
        with open(config_path, 'r') as f:
            return json.load(f)["google_api_key"]

    def _generate_response(self, query, context):
        prompt = f"""
You are a helpful and knowledgeable assistant. Below is a list of product advertisements. Your task is to respond to the user's query in a natural and informative way, promoting one or two relevant products. Include brand and url into your response without revealing that you have access to product ads.

At the end of your response (not visible to the user), list which documents you included clearly only using the format: 'id: ..., id: ..., ...'.

USER QUERY: {query}

RETRIEVED COMMERCIAL ADS:
{context}

Please provide a helpful, informative response directed to the user based on the above information.
"""
        llm_output = self.model.generate_content(prompt).text
        ids = re.findall(r'id:\s*([^\s,]+)', llm_output, re.IGNORECASE)

        # Clean up the last line if it contains IDs
        lines = llm_output.strip().splitlines()
        if lines and re.search(r'id:\s*\w+', lines[-1], re.IGNORECASE):
            response_text = "\n".join(lines[:-1]).strip()
        else:
            response_text = llm_output.strip()

        return response_text, ids

    def batch_generate(self, queries, output_file):
        """
        Generate responses for a batch of queries and save to file.

        Args:
            queries (list): List of query dictionaries.
            output_file (str): Path to save the generated responses.
        """
        responses = []

        for item in tqdm(queries, desc="Processing queries"):
            query_text = item["query"]
            domain = item.get("domain", "Unknown")
            subdomain = item.get("subdomain", "Unknown")

            context = self.retriever.get_relevant_context(query_text, use_full_docs=self.use_full_docs)
            response, docs_in_response = self._generate_response(query_text, context)

            responses.append({
                "query": query_text,
                "domain": domain,
                "subdomain": subdomain,
                "retrieved_context": context,
                "response": response,
                "documents_in_response": docs_in_response
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(responses, f, ensure_ascii=False, indent=2)

        print(f"\nSaved {len(responses)} query-response pairs to {output_file}")

