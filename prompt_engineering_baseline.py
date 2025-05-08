import json
import google.generativeai as genai
import os
import argparse
from typing import List, Dict

"""
prompt engineering baseline

To run:
python prompt_engineering_baseline.py --ads_file ds/faiss_index/200_sampled_ads.json --queries_file query_responses_original_200.json --classified_file classified_ads_200.json --output_file prompt_output.json
"""

def load_api_key() -> str:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "keys.json")
    with open(config_path, 'r') as f:
        return json.load(f)["google_api_key"]

def initialize_gemini():
    api_key = load_api_key()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-pro")

def create_prompt(query: str, ad: str) -> str:
    return f"""You are given an ad and a user query. Rewrite the ad to better match the user query while keeping factual content and tone.

Query: {query}
Original Ad: {ad}
Rewritten Ad:"""


def rewrite_ads(ads: List[Dict], queries: List[Dict], classified_ads: List[Dict], model) -> List[Dict]:
    
    rewritten = []

    for a, c in zip(ads, classified_ads):
        ad = a.get("text")
        ad_domain = c.get("domain")
        ad_subdomain = c.get("subdomain")

        # print(f"Processing query: {query}, ad: {ad}")  # Check if it's entering the loop

        for query in queries:
            if query.get('domain')==ad_domain and query['subdomain']==ad_subdomain:
                prompt = create_prompt(query['query'], ad)
                print(f"Generated prompt: {prompt}")  # Check the prompt that is being generated

                # Generate the rewritten ad
                response = model.generate_content(prompt)
                print(f"Response: {response.text}")  # Check the response from the model

                rewritten.append(response.text.strip())
                
    return rewritten

def main(ads_file: str, queries_file: str, classified_file: str, output_file: str):
    # Load all input files
    with open(ads_file, 'r', encoding='utf-8') as f:
        ads = json.load(f)
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    with open(classified_file, 'r', encoding='utf-8') as f:
        classified_ads = json.load(f)

    model = initialize_gemini()
    rewritten = rewrite_ads(ads, queries, classified_ads, model)

    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rewritten, f, ensure_ascii=False, indent=2)

    print(f"Rewritten ads saved to {output_file}")
    
    # add metrics!!!

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewrite ads to better match queries based on domain and subdomain.")
    parser.add_argument("--ads_file", type=str, required=True, help="Path to the original ads JSON file.")
    parser.add_argument("--queries_file", type=str, required=True, help="Path to the user queries JSON file.")
    parser.add_argument("--classified_file", type=str, required=True, help="Path to the domain/subdomain mapping JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the rewritten ads JSON output.")

    args = parser.parse_args()
    main(args.ads_file, args.queries_file, args.classified_file, args.output_file)
