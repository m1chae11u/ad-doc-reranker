import json
import google.generativeai as genai
import os
import argparse
import re
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm_asyncio

"""
prompt engineering baseline

To run:
python using_sft_model.py --ads_file test_data.json --output_file sft_rewritten_ads_cot.json
"""

def create_prompt(ad: str) -> str:
    return f"""You are given an advertisement. Your task is to rewrite the ad so that its ranking in retrieval and inclusion in LLM responses improves. Focus on semantic relevance and matching the user's likely search intent.

Original Ad: {ad}

Think step by step first, then provide the improved version.

Respond with the improved version at the end of your response **only** in the following format:
Thinking: ...
Title: ...
Description: ...
"""

def process_single_ad(a: Dict, model, tokenizer) -> Dict:
    """Process a single ad with the model"""
    ad = f"Title: {a.get('title', '')}\n\nDescription: {a.get('text', '')}"
    prompt = create_prompt(ad)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=700,
            do_sample=True,
            top_p=0.95,
            temperature=0.7
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract improved ad
    title_match = re.findall(r'Title:\s*(.*)', decoded)
    description_match = re.findall(r'Description:\s*(.*)', decoded)

    title = title_match[-1] if title_match else ""
    description = description_match[-1] if description_match else ""

    return {
        "user_query": a['user_query'],
        "title": title,
        "text": description,
        "url": a['url'],
        "seller": a['seller'],
        "brand": a['brand'],
        "source": a['source'],
        "ad_id": a['ad_id']
    }

async def rewrite_ads_parallel(ads: List[Dict], model, tokenizer, max_workers: int = 4) -> List[Dict]:
    """Process ads in parallel using a thread pool"""
    rewritten = []
    loop = asyncio.get_event_loop()
    
    # Create a thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a semaphore to limit concurrent processing
        sem = asyncio.Semaphore(max_workers)
        
        async def process_with_semaphore(ad):
            async with sem:
                # Run model inference in a separate thread to not block the event loop
                return await loop.run_in_executor(executor, process_single_ad, ad, model, tokenizer)
        
        # Create tasks for all ads
        tasks = [process_with_semaphore(ad) for ad in ads]
        
        # Process tasks with progress bar
        for result in await tqdm_asyncio.gather(*tasks, desc="Rewriting Ads"):
            rewritten.append(result)
            # Print some details about the processed ad
            print(f"Processed ad: {result['ad_id']}")
            print(f"Title: {result['title']}")
            print(f"Description: {result['text'][:100]}...")
            print("-" * 40)
    
    return rewritten

def batch_process_ads(ads: List[Dict], model, tokenizer, batch_size: int = 8) -> List[Dict]:
    """Process ads in batches to speed up inference"""
    rewritten = []
    
    # Process ads in batches
    for i in range(0, len(ads), batch_size):
        batch = ads[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(ads) + batch_size - 1) // batch_size}")
        
        # Prepare all prompts in the batch
        prompts = [create_prompt(f"Title: {a.get('title', '')}\n\nDescription: {a.get('text', '')}") for a in batch]
        
        # Tokenize all prompts in the batch
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
        
        # Generate outputs for the entire batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=700,
                do_sample=True,
                top_p=0.95,
                temperature=0.7
            )
        
        # Decode and process each output
        decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        for a, decoded in zip(batch, decoded_outputs):
            title_match = re.findall(r'Title:\s*(.*)', decoded)
            description_match = re.findall(r'Description:\s*(.*)', decoded)
            
            title = title_match[-1] if title_match else ""
            description = description_match[-1] if description_match else ""
            
            result = {
                "user_query": a['user_query'],
                "title": title,
                "text": description,
                "url": a['url'],
                "seller": a['seller'],
                "brand": a['brand'],
                "source": a['source'],
                "ad_id": a['ad_id']
            }
            
            rewritten.append(result)
            print(f"Processed ad: {result['ad_id']}")
            print(f"Title: {title}")
            print(f"Description: {description[:100]}...")
            print("-" * 40)
    
    return rewritten

def rewrite_ads(ads: List[Dict], model, tokenizer) -> List[Dict]:
    """Choose the best method based on available hardware"""
    # Detect if we have a GPU with sufficient memory for batch processing
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 10e9:
        print("Using batch processing for faster inference")
        return batch_process_ads(ads, model, tokenizer)
    else:
        print("Using parallel processing for faster inference")
        return asyncio.run(rewrite_ads_parallel(ads, model, tokenizer, max_workers=4))

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewrite ads to improve general quality using prompt engineering.")
    parser.add_argument("--ads_file", type=str, required=True, help="Path to the original ads JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the rewritten ads JSON output.")
    args = parser.parse_args()
    main(args.ads_file, args.output_file)