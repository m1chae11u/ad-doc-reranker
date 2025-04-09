import json
import os
import argparse
import google.generativeai as genai
from collections import defaultdict
from typing import List, Dict, Tuple

'''
classifies each ad into a domain and subdomain

for each domain subdomain pair, x number (default 3) of queries are created 

to run: python queries.py --input_file sampled_ads.json --output_file queries.json --num_queries 3
'''

def load_api_key():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "keys.json")
    with open(config_path, 'r') as f:
        return json.load(f)["google_api_key"]

def initialize_gemini():
    api_key = load_api_key()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-pro")

def classify_ad_to_domain_and_subdomain(ad: Dict, known_domains: List[str], known_subdomains: Dict[str, set], model) -> Tuple[str, str]:
    domain_list = ', '.join(known_domains) if known_domains else 'None'
    subdomain_list = ''
    for domain in known_domains:
        subs = known_subdomains.get(domain, set())
        if subs:
            subdomain_list += f"- {domain}: {', '.join(sorted(subs))}\n"
    if not subdomain_list:
        subdomain_list = 'None'

    prompt = f"""
You are classifying product advertisements into domains and subdomains.
For example:
- Domain (e.g., 'Fashion', 'Electronics', 'Healthcare')
- Subdomain (a very specific category within the domain, e.g., 'Men’s Shoes', 'Smartphones', 'Skincare')

Domains so far:
{domain_list}

Known subdomains under each domain:
{subdomain_list}

Ad details:
Headline: {ad.get("headline", "")}
Description: {ad.get("description", "")}
Product: {ad.get("product_name", "")}
Brand: {ad.get("brand", "")}

Your task:
1. Classify this ad into one of the existing domains and subdomains listed above, if appropriate.
2. If it does not fit any existing domain or subdomain, propose a new one.
3. Respond **only** in the format: `Domain: <domain name> Subdomain: <subdomain name>`
"""
    response = model.generate_content(prompt)
    raw_text = response.text.strip().strip('"')

    domain, subdomain = "Uncategorized", "General"
    try:
        if "Domain:" in raw_text and "Subdomain:" in raw_text:
            parts = raw_text.split("Subdomain:")
            domain_part = parts[0].split("Domain:")[1].strip()
            subdomain_part = parts[1].strip()
            domain, subdomain = domain_part, subdomain_part
    except Exception:
        print("Could not parse domain/subdomain from response:", raw_text)

    return domain, subdomain

def generate_queries_for_domain_and_subdomain(domain: str, subdomain: str, num_queries: int, model) -> List[str]:
    prompt = f"""
You are helping a marketing team generate realistic user search queries that would retrieve ads related to the subdomain: "{subdomain}" under the domain: "{domain}".

Generate {num_queries} unique, natural-sounding queries a user might say to a LLM that will allow a LLM to promote an ad in this category.

Respond with a numbered list.
"""
    response = model.generate_content(prompt)
    raw_output = response.text.strip()
    queries = []

    for line in raw_output.splitlines():
        if line.strip():
            parts = line.split(". ", 1)
            if len(parts) == 2:
                queries.append(parts[1].strip())
            else:
                queries.append(line.strip())

    return queries[:num_queries]

def process_ads(input_file: str, output_file: str, num_queries_per_subdomain: int = 3):
    with open(input_file, 'r', encoding='utf-8') as f:
        ads = json.load(f)

    model = initialize_gemini()
    domain_to_subdomains = defaultdict(lambda: defaultdict(list))
    known_domains = []
    known_subdomains = defaultdict(set)

    # Step 1: Classify ads by domain and subdomain
    for i, ad in enumerate(ads):
        domain, subdomain = classify_ad_to_domain_and_subdomain(ad, known_domains, known_subdomains, model)

        if domain not in known_domains:
            known_domains.append(domain)
        if subdomain not in known_subdomains[domain]:
            known_subdomains[domain].add(subdomain)

        domain_to_subdomains[domain][subdomain].append(ad)
        print(f"[{i+1}/{len(ads)}] Ad classified → Domain: {domain}, Subdomain: {subdomain}")

    # Step 2: Generate queries for each (domain, subdomain) pair
    query_dataset = []

    for domain, sub_map in domain_to_subdomains.items():
        for subdomain in sub_map:
            print(f"\nGenerating {num_queries_per_subdomain} queries for Domain: {domain} | Subdomain: {subdomain}")
            queries = generate_queries_for_domain_and_subdomain(domain, subdomain, num_queries_per_subdomain, model)

            for q in queries:
                query_dataset.append({
                    "domain": domain,
                    "subdomain": subdomain,
                    "query": q
                })

    # Step 3: Save query dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(query_dataset, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(query_dataset)} queries across {len(known_domains)} domains to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group ads by domain & subdomain and generate realistic queries.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input ad dataset (JSON list).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the query dataset (JSON).")
    parser.add_argument("--num_queries", type=int, default=3, help="Number of queries to generate per subdomain.")

    args = parser.parse_args()
    process_ads(args.input_file, args.output_file, args.num_queries)
