import json
import google.generativeai as genai
import os
import argparse
import re
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
# from metric_calculations import MetricEvaluator

"""
prompt engineering baseline

To run:
python using_sft_model.py --ads_file test_data.json --output_file sft_rewritten_ads_cot.json
"""

def create_prompt(ad: str) -> str:
    return f"""You are given an advertisement. Your task is to rewrite it so that its ranking in retrieval and inclusion in LLM responses improves. Focus on semantic relevance and matching the user’s likely search intent.

Original Ad: {ad}

Think step by step first, then provide the improved version.

Respond with the improved version at the end of your response in the following format:
Title: ...
Description: …
"""

def rewrite_ads(ads: List[Dict], model, tokenizer) -> List[Dict]:
    rewritten = []

    for a in ads:
        ad = f"Title: {a.get('title', '')}\n\nDescription: {a.get('text', '')}"
        prompt = create_prompt(ad)
        print(f"Generated prompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                top_p=0.95,
                temperature=0.7
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Decoded output: {decoded}")

        # Extract improved ad
        title_match = re.search(r'Title:\s*(.*)', decoded)
        description_match = re.search(r'Description:\s*(.*)', decoded, re.DOTALL)

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

        print(f"title: {title}\n\ndescription: {description}\n")

    return rewritten


def main(ads_file: str, output_file: str):
    # Load ads
    with open(ads_file, 'r', encoding='utf-8') as f:
        ads = json.load(f)

    model_dir = "sft_output" 
    base = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base, model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    rewritten = rewrite_ads(ads, model, tokenizer)

    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rewritten, f, ensure_ascii=False, indent=2)

    print(f"Rewritten ads saved to {output_file}")
    
    # evaluator = MetricEvaluator(
    #     original_ads_path="ds/faiss_index/200_sampled_ads.json",
    #     queries_path="queries_200.json",
    #     index_input_path="sft_responses.json",
    #     index_output_dir="faiss_index_rewritten",
    #     original_rankings_path="rankings_original.json",
    #     rewritten_rankings_path="rankings_rewritten.json",
    #     original_responses_path="query_responses_original_200.json",
    #     rewritten_responses_path="query_responses_rewritten.json",
    #     classified_ads_path="classified_ads_200.json"
    # )
    # evaluator.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewrite ads to improve general quality using prompt engineering.")
    parser.add_argument("--ads_file", type=str, required=True, help="Path to the original ads JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the rewritten ads JSON output.")

    args = parser.parse_args()
    main(args.ads_file, args.output_file)
