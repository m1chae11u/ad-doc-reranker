import json
import argparse
from typing import List, Dict, Any, Optional
import os

def load_json_file(file_path: str) -> Any:
    """Load data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data: Any, file_path: str) -> None:
    """Save data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def create_sft_dataset(
    ads_file: str,
    queries_file: str,
    rewritten_ads_file: str,
    rankings_file: str,
    classified_ads_file: str,
    output_file: str
) -> None:
    """
    Create a dataset for supervised fine-tuning using original ads, queries, and rewritten ads.
    
    Args:
        ads_file: JSON file with original ads
        queries_file: JSON file with queries
        rewritten_ads_file: JSON file with rewritten ads
        rankings_file: JSON file with rankings of ads for queries
        classified_ads_file: JSON file with domain/subdomain classifications for ads
        output_file: Path to save the output SFT dataset
    """
    # Load data
    original_ads = load_json_file(ads_file)
    queries = load_json_file(queries_file)
    rewritten_ads = load_json_file(rewritten_ads_file)
    rankings = load_json_file(rankings_file)
    classified_ads = load_json_file(classified_ads_file)
    
    # Create dictionaries for easy lookup
    ad_id_to_ad = {ad.get("ad_id"): ad for ad in original_ads if "ad_id" in ad}
    ad_id_to_classification = {item["id"]: {"domain": item["domain"], "subdomain": item["subdomain"]} 
                               for item in classified_ads}
    
    # Create query lookup by domain/subdomain
    query_by_domain_subdomain = {}
    for query_item in queries:
        domain = query_item.get("domain")
        subdomain = query_item.get("subdomain")
        if domain and subdomain:
            key = f"{domain}|{subdomain}"
            query_by_domain_subdomain[key] = query_item.get("query", "").strip('"')
    
    # Create the SFT dataset
    sft_data = []
    
    # Process each ad with its rewritten version (using matching indices)
    # Assuming rewritten_ads is in the same order as original_ads
    for i, ad in enumerate(original_ads):
        if i >= len(rewritten_ads):
            break
            
        ad_id = ad.get("ad_id")
        if not ad_id:
            continue
            
        classification = ad_id_to_classification.get(ad_id)
        if not classification:
            continue
        
        domain = classification["domain"]
        subdomain = classification["subdomain"]
        key = f"{domain}|{subdomain}"
        
        # Get the query that matches this ad's domain/subdomain
        query = query_by_domain_subdomain.get(key)
        if not query:
            continue
        
        # Get the original and rewritten texts
        original_text = ad.get("text", "")
        if not original_text:
            continue
            
        # Handle different formats of rewritten_ads
        rewritten_text = ""
        if isinstance(rewritten_ads[i], str):
            # Format: rewritten_ads is a list of strings (rewritten ads)
            rewritten_text = rewritten_ads[i]
        elif isinstance(rewritten_ads[i], dict):
            # Format: rewritten_ads is a list of dicts with original_ad and rewritten_ad keys
            rewritten_text = rewritten_ads[i].get("rewritten_ad", "")
            
        if not rewritten_text:
            continue
            
        sft_data.append({
            "ad_id": ad_id,
            "domain": domain,
            "subdomain": subdomain,
            "query": query,
            "original_ad": original_text,
            "rewritten_ad": rewritten_text
        })
    
    # Save the SFT dataset
    save_json_file(sft_data, output_file)
    print(f"Created SFT dataset with {len(sft_data)} examples. Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dataset for supervised fine-tuning.")
    parser.add_argument("--ads_file", type=str, required=True, help="JSON file with original ads")
    parser.add_argument("--queries_file", type=str, required=True, help="JSON file with queries")
    parser.add_argument("--rewritten_ads_file", type=str, required=True, help="JSON file with rewritten ads")
    parser.add_argument("--rankings_file", type=str, required=True, help="JSON file with rankings of ads for queries")
    parser.add_argument("--classified_ads_file", type=str, required=True, help="JSON file with domain/subdomain classifications for ads")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output SFT dataset")
    
    args = parser.parse_args()
    
    create_sft_dataset(
        args.ads_file,
        args.queries_file,
        args.rewritten_ads_file,
        args.rankings_file,
        args.classified_ads_file,
        args.output_file
    ) 