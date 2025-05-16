import json
import os
import argparse
import google.generativeai as genai
from collections import defaultdict
from typing import List, Dict, Tuple

'''
classifies each ad into a domain and subdomain

for each domain subdomain pair, x number (default 3) of queries are created 

saves 2 datasets, queries and domain subdomain for each ad

to run: python queries.py --input_file sampled_ads.json --output_file queries.json --classified_output_file classified_ads.json --num_queries 3
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
You are classifying product advertisements into specific domains and very specific subdomains.
For example:
- Domain (e.g., 'Fashion', 'Electronics', 'Healthcare')
- Subdomain (a **very** specific category within the domain that is **different** from the domain, e.g., 'Men’s Shoes', 'Smartphones', 'Skincare')

Domains so far:
{domain_list}

Known subdomains under each domain:
{subdomain_list}

Ad details:
Headline: {ad.get("user_query", "")}
Description: {ad.get("text", "")}
Product: {ad.get("title", "")}
Brand: {ad.get("brand", "")}

Your task:
1. Classify this ad into one of the existing domains and subdomains listed above, if appropriate.
2. If it does not fit into any existing domain or subdomain, propose a new one.
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
Generate realistic user LLM chat related to the subdomain: "{subdomain}" under the domain: "{domain}".

Generate {num_queries} unique, natural-sounding chats a user might say to a LLM.

Respond **only** with a numbered list.
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

def process_ads(input_file: str, output_file: str, classified_output_file: str, num_queries_per_subdomain: int = 3):
    with open(input_file, 'r', encoding='utf-8') as f:
        ads = json.load(f)

    model = initialize_gemini()
    domain_to_subdomains = defaultdict(lambda: defaultdict(list))
    known_domains = []
    known_subdomains = defaultdict(set)
    classified_ads = []
    domain_subdomain_counts = defaultdict(lambda: defaultdict(int))

    # Step 1: Classify ads by domain and subdomain
    for i, ad in enumerate(ads):
        domain, subdomain = classify_ad_to_domain_and_subdomain(ad, known_domains, known_subdomains, model)

        if domain not in known_domains:
            known_domains.append(domain)
        if subdomain not in known_subdomains[domain]:
            known_subdomains[domain].add(subdomain)

        domain_to_subdomains[domain][subdomain].append(ad)
        domain_subdomain_counts[domain][subdomain] += 1
        
        classified_ads.append({
            "id": ad["ad_id"],
            "domain": domain,
            "subdomain": subdomain
        })
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

    # Step 3: Save query and domain subdomain datasets
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(query_dataset, f, ensure_ascii=False, indent=2)
    with open(classified_output_file, 'w', encoding='utf-8') as f:
        json.dump(classified_ads, f, ensure_ascii=False, indent=2)

    print("\nNumber of ads per domain-subdomain pair:")
    for domain, sub_map in domain_to_subdomains.items():
        for subdomain, ads_list in sub_map.items():
            print(f"- {domain} / {subdomain}: {len(ads_list)} ads")
    print(f"\nSaved {len(query_dataset)} queries across {len(known_domains)} domains to {output_file}")
    print(f"Saved domain/subdomain classification for {len(classified_ads)} ads to {classified_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group ads by domain & subdomain and generate realistic queries.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input ad dataset (JSON list).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the query dataset (JSON).")
    parser.add_argument("--classified_output_file", type=str, required=True, help="Path to save domain/subdomain mapping (JSON).")
    parser.add_argument("--num_queries", type=int, default=3, help="Number of queries to generate per subdomain.")

    args = parser.parse_args()
    process_ads(args.input_file, args.output_file, args.classified_output_file, args.num_queries)
