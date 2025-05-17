import json
import google.generativeai as genai
import os
import argparse
import re
from typing import List, Dict
import asyncio
from tqdm.asyncio import tqdm_asyncio
from concurrent.futures import ThreadPoolExecutor

"""
Prompt engineering with multiple strategies, executed in parallel.

To run:
python prompt_engineering.py --ads_file ds/10_sampled_ads.json --output_file 10_prompt_output.json
python prompt_engineering.py --ads_file ds/test_data.json --output_file test_prompt_output.json
python prompt_engineering.py --ads_file ds/train_data.json --output_file train_rewritten_ads.json
"""

def load_api_key() -> str:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "keys.json")
    with open(config_path, 'r') as f:
        return json.load(f)["google_api_key"]

def initialize_gemini():
    api_key = load_api_key()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-pro")

def create_prompt(ad: str) -> Dict[str, str]:
    return f"""You are given an advertisement. Your task is to rewrite it so that its ranking in retrieval and inclusion in LLM response improves. Focus on semantic relevance and matching the user’s likely search intent. 

Original Ad: {ad}

Respond with the improved version:
Title: ... 
Description: ...
""" 

def extract_title_description(response_text: str) -> Dict[str, str]:
    title_match = re.findall(r'Title:\s*(.*)', response_text)
    description_match = re.findall(r'Description:\s*(.*)', response_text)

    return {
        "title": title_match[-1] if title_match else "",
        "text": description_match[-1] if description_match else ""
    }

# Async wrapper to call Gemini
def sync_model_call(model, prompt: str) -> str:
    return model.generate_content(prompt).text

async def run_prompt(ad: Dict, model, executor) -> Dict[str, Dict]:
    ad_text = f"Title: {ad.get('title', '')}\n\nDescription: {ad.get('text', '')}"
    prompt = create_prompt(ad_text)

    loop = asyncio.get_event_loop()
    response_text = await loop.run_in_executor(executor, sync_model_call, model, prompt)
    return extract_title_description(response_text)

async def rewrite_ads_async(ads: List[Dict], model) -> List[Dict]:
    rewritten = []
    executor = ThreadPoolExecutor()

    async def process_single_ad(ad):
        rewritten_ad = await run_prompt(ad, model, executor)
        return {
            "user_query": ad['user_query'],
            "url": ad['url'],
            "seller": ad['seller'],
            "brand": ad['brand'],
            "source": ad['source'],
            "ad_id": ad['ad_id'],
            "title": rewritten_ad["title"],
            "text": rewritten_ad["text"]
        }

    tasks = [process_single_ad(ad) for ad in ads]
    rewritten = await tqdm_asyncio.gather(*tasks, desc="Rewriting ads", total=len(tasks))

    return rewritten

def main(ads_file: str, output_file: str):
    with open(ads_file, 'r', encoding='utf-8') as f:
        ads = json.load(f)
    print (f"loaded {ads_file}")

    model = initialize_gemini()
    rewritten = asyncio.run(rewrite_ads_async(ads, model))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rewritten, f, ensure_ascii=False, indent=2)
    print(f"Rewritten ads saved to {output_file}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewrite ads with multiple prompting strategies using Gemini.")
    parser.add_argument("--ads_file", type=str, required=True, help="Path to the original ads JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the rewritten ads JSON output.")

    args = parser.parse_args()
    main(args.ads_file, args.output_file)
