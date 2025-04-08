import json
import os
import argparse
import google.generativeai as genai
from collections import defaultdict
from typing import List, Dict

# to run: python queries.py --input_file sampled_ads.json --output_file queries.json --num_queries 3

def load_api_key():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "keys.json")
    with open(config_path, 'r') as f:
        return json.load(f)["google_api_key"]


def initialize_gemini():
    api_key = load_api_key()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-pro")


def classify_ad_domain_and_subdomain(ad: Dict, known_domains: List[str], model) -> (str, str):
    prompt = f"""
You are classifying product advertisements into logical domains and subdomains.

- The **domain** is a broad category (e.g., "Technology", "Health", "Home & Garden").
- The **subdomain** is a more specific category within that domain (e.g., for Technology: "Wearables", "Laptops", "Smartphones").

Domains so far:
{', '.join(known_domains) if known_domains else 'None'}

Do NOT make the subdomain the same as the domain.

Ad details:
Headline: {ad.get("headline", "")}
Description: {ad.get("description", "")}
Product: {ad.get("product_name", "")}
Brand: {ad.get("brand", "")}

Respond in the following JSON format:
{{
  "domain": "...",
  "subdomain": "..."
}}
"""
    response = model.generate_content(prompt)

    try:
        result = json.loads(response.text.strip())
        domain = result.get("domain", "").strip()
        subdomain = result.get("subdomain", "").strip()

        # Prevent subdomain = domain
        if subdomain.lower() == domain.lower():
            subdomain = "General"

        return domain, subdomain
    except Exception as e:
        print("Error parsing domain/subdomain response:", response.text.strip())
        return "Unknown", "General"



def generate_queries_for_domain(domain: str, num_queries: int, model) -> List[str]:
    prompt = f"""
You are helping a marketing team generate realistic user search queries that would retrieve ads related to the domain: "{domain}".

Generate {num_queries} unique, natural-sounding search queries that a user might type when looking for products in this domain.

Respond with a numbered list.
"""
    response = model.generate_content(prompt)
    raw_output = response.text.strip()
    queries = []

    for line in raw_output.splitlines():
        if line.strip():
            # Try to extract "1. query" format
            parts = line.split(". ", 1)
            if len(parts) == 2:
                queries.append(parts[1].strip())
            else:
                queries.append(line.strip())

    return queries[:num_queries]


def process_ads(input_file: str, output_file: str, num_queries_per_domain: int = 3):
    with open(input_file, 'r', encoding='utf-8') as f:
        ads = json.load(f)

    model = initialize_gemini()
    domain_to_ads = defaultdict(list)
    known_domains = []

    # Step 1: Classify ads by domain
    for i, ad in enumerate(ads):
        domain, subdomain = classify_ad_domain_and_subdomain(ad, known_domains, model)
        if domain not in known_domains:
            known_domains.append(domain)
        domain_to_ads[domain].append(ad)
        print(f"[{i+1}/{len(ads)}] Ad classified into domain: {domain}")

    # Step 2: Generate queries for each domain
    query_dataset = []

    for domain in known_domains:
        print(f"\nGenerating {num_queries_per_domain} queries for domain: {domain}")
        queries = generate_queries_for_domain(domain, num_queries_per_domain, model)

        for q in queries:
            query_dataset.append({
                "domain": domain,
                "subdomain": subdomain,
                "query": q
            })

    # Step 3: Save query dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(query_dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(query_dataset)} queries across {len(known_domains)} domains to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group ads by domain and generate realistic queries per domain.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input ad dataset (JSON list).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the query dataset (JSON).")
    parser.add_argument("--num_queries", type=int, default=3, help="Number of queries to generate per domain.")

    args = parser.parse_args()
    process_ads(args.input_file, args.output_file, args.num_queries)
