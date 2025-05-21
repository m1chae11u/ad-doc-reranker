import json
import google.generativeai as genai
import os
import argparse
import re
import torch
import asyncio
from peft import PeftModel
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

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

def process_one_ad(ad: Dict, model, tokenizer) -> Dict:
    ad_text = f"Title: {ad.get('title', '')}\n\nDescription: {ad.get('text', '')}"
    prompt = create_prompt(ad_text)

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

    title_match = re.search(r'Title:\s*(.*)', decoded)
    description_match = re.search(r'Description:\s*(.*)', decoded, re.DOTALL)

    title = title_match.group(1).strip() if title_match else ""
    description = description_match.group(1).strip() if description_match else ""

    return {
        "user_query": ad['user_query'],
        "title": title,
        "text": description,
        "url": ad['url'],
        "seller": ad['seller'],
        "brand": ad['brand'],
        "source": ad['source'],
        "ad_id": ad['ad_id']
    }

async def rewrite_ads_parallel(ads: List[Dict], model, tokenizer, max_concurrent_tasks: int = 4) -> List[Dict]:
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)

    async def run_in_executor(ad):
        return await loop.run_in_executor(executor, process_one_ad, ad, model, tokenizer)

    sem = asyncio.Semaphore(max_concurrent_tasks)

    async def sem_task(ad):
        async with sem:
            return await run_in_executor(ad)

    tasks = [sem_task(ad) for ad in ads]

    # Use tqdm to show progress
    results = []
    for future in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Rewriting Ads"):
        result = await future
        results.append(result)

    return results

def main(ads_file: str, output_file: str):
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

    rewritten = asyncio.run(rewrite_ads_parallel(ads, model, tokenizer, max_concurrent_tasks=20))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rewritten, f, ensure_ascii=False, indent=2)

    print(f"Rewritten ads saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewrite ads to improve general quality using prompt engineering.")
    parser.add_argument("--ads_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args.ads_file, args.output_file)
