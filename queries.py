'''
classifies each ad into a domain and subdomain

for each domain subdomain pair, x number (default 3) of queries are created 

saves 2 datasets, queries and domain subdomain for each ad

to run: python queries.py --input_file sampled_ads.json --output_file queries.json --classified_output_file classified_ads.json --num_queries 3
'''

import json
import os
import argparse
import asyncio
import google.generativeai as genai
from collections import defaultdict
from typing import List, Dict, Tuple
from functools import partial
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()

def load_api_key():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "keys.json")
    with open(config_path, 'r') as f:
        return json.load(f)["google_api_key"]

def initialize_gemini():
    api_key = load_api_key()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-pro")

async def classify_ad_async(ad, known_domains, known_subdomains, model):
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
    response = await asyncio.get_event_loop().run_in_executor(executor, partial(model.generate_content, prompt))
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

    return ad["ad_id"], domain, subdomain

async def generate_queries_async(domain, subdomain, num_queries, model):
    prompt = f"""
Generate realistic user LLM chat related to the subdomain: "{subdomain}" under the domain: "{domain}".
Generate {num_queries} unique, natural-sounding chats a user might say to a LLM.
Respond **only** with a numbered list.
"""
    response = await asyncio.get_event_loop().run_in_executor(executor, partial(model.generate_content, prompt))
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

async def process_ads_async(input_file: str, output_file: str, classified_output_file: str, num_queries_per_subdomain: int = 3):
    with open(input_file, 'r', encoding='utf-8') as f:
        ads = json.load(f)

    model = initialize_gemini()
    known_domains = []
    known_subdomains = defaultdict(set)

    classified_ads = []
    domain_to_subdomains = defaultdict(lambda: defaultdict(list))
    domain_subdomain_counts = defaultdict(lambda: defaultdict(int))

    async def classify_with_index(i, ad):
        # Pass current known domains and subdomains to avoid race conditions
        result = await classify_ad_async(ad, known_domains.copy(), known_subdomains.copy(), model)
        ad_id, domain, subdomain = result
        print(f"[{i + 1}/{len(ads)}] Ad classified → Domain: {domain}, Subdomain: {subdomain}")
        return result

    classify_tasks = [classify_with_index(i, ad) for i, ad in enumerate(ads)]
    classified_results = await asyncio.gather(*classify_tasks)

    for ad_id, domain, subdomain in classified_results:
        if domain not in known_domains:
            known_domains.append(domain)
        if subdomain not in known_subdomains[domain]:
            known_subdomains[domain].add(subdomain)

        ad = next(ad for ad in ads if ad["ad_id"] == ad_id)
        domain_to_subdomains[domain][subdomain].append(ad)
        domain_subdomain_counts[domain][subdomain] += 1

        classified_ads.append({
            "id": ad_id,
            "domain": domain,
            "subdomain": subdomain
        })

    # Step 2: Generate queries concurrently
    query_tasks = [
        generate_queries_async(domain, subdomain, num_queries_per_subdomain, model)
        for domain in domain_to_subdomains
        for subdomain in domain_to_subdomains[domain]
    ]
    query_results = await asyncio.gather(*query_tasks)

    query_dataset = []
    index = 0
    for domain in domain_to_subdomains:
        for subdomain in domain_to_subdomains[domain]:
            queries = query_results[index]
            for q in queries:
                query_dataset.append({
                    "domain": domain,
                    "subdomain": subdomain,
                    "query": q
                })
            index += 1

    # Step 3: Save results
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
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--classified_output_file", type=str, required=True)
    parser.add_argument("--num_queries", type=int, default=3)

    args = parser.parse_args()
    asyncio.run(process_ads_async(args.input_file, args.output_file, args.classified_output_file, args.num_queries))
