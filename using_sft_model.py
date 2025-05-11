import json
import google.generativeai as genai
import os
import argparse
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from metric_calculations import MetricEvaluator

"""
prompt engineering baseline

To run:
python using_sft_model.py --ads_file ds/faiss_index/200_sampled_ads.json --output_file sft_rewritten_ads.json
"""

def create_prompt(ad: str) -> str:
    return f"""You are given an advertisement. Please rewrite it so that it is more likely to rank higher when retrieved by a search system for queries relevant to its content.

Make sure not to add any new information or make assumptions that are not already present in the original ad. 

Original Ad:
{ad}

Rewritten Ad:"""

def rewrite_ads(ads: List[Dict], model) -> List[Dict]:
    rewritten = []

    for a in ads:
        ad = f"Title: {a.get('title', '')}\n\nDescription: {a.get('text', '')}"
        prompt = create_prompt(ad)
        print(f"Generated prompt: {prompt}")

        response = model.generate_content(prompt)
        # print(f"{response.text}")
        
        title_match = re.search(r'Title:\s*(.*)', response.text)
        description_match = re.search(r'Description:\s*(.*)', response.text, re.DOTALL)

        title = title_match.group(1).strip() if title_match else ""
        description = description_match.group(1).strip() if description_match else ""

        rewritten.append({
            "user_query": a['user_query'],
            "title": title,
            "text": description,
            "url": a['url'],
            "seller": a['seller'],
            "brand": a['brand'],
            "source": a['source'],
            "ad_id": a['ad_id']
        })
        print(f"title: {title} \n\ndescription: {description}")

    return rewritten

def main(ads_file: str, output_file: str):
    # Load ads
    with open(ads_file, 'r', encoding='utf-8') as f:
        ads = json.load(f)

    model_dir = "sft_output" 
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    rewritten = rewrite_ads(ads, model)

    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rewritten, f, ensure_ascii=False, indent=2)

    print(f"Rewritten ads saved to {output_file}")
    
    evaluator = MetricEvaluator(
        original_ads_path="ds/faiss_index/200_sampled_ads.json",
        queries_path="queries_200.json",
        index_input_path="sft_responses.json",
        index_output_dir="faiss_index_rewritten",
        original_rankings_path="rankings_original.json",
        rewritten_rankings_path="rankings_rewritten.json",
        original_responses_path="query_responses_original_200.json",
        rewritten_responses_path="query_responses_rewritten.json",
        classified_ads_path="classified_ads_200.json"
    )
    evaluator.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewrite ads to improve general quality using prompt engineering.")
    parser.add_argument("--ads_file", type=str, required=True, help="Path to the original ads JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the rewritten ads JSON output.")

    args = parser.parse_args()
    main(args.ads_file, args.output_file)
