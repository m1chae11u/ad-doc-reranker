import argparse
import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import ReinforceTrainer, ReinforceConfig
from datasets import load_dataset, Dataset
from tqdm import tqdm

from loss import SimilarityLoss

'''
Fine-tunes LLaMA 3.1 8B using REINFORCE and a custom loss function as reward.

Usage:
python reinforce_custom_loss.py --data_file ads.json --output_dir ./finetuned_model --batch_size 1
'''

def load_original_ads_by_id(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_ads = json.load(f)

    ads_by_id = {}
    for ad in raw_ads:
        ad_id = ad.get("ad_id")
        if ad_id:
            ads_by_id[ad_id] = ad

    return ads_by_id

def load_rankings(rankings_path):
    with open(rankings_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    formatted_data = {}
    for entry in raw_data:
        query_text = entry["query"]["query"]
        formatted_data[query_text] = {
            "domain": entry["query"].get("domain", "Unknown"),
            "subdomain": entry["query"].get("subdomain", "Unknown"),
            "ranked_ad_ids": entry.get("ranked_ad_ids", [])
        }

    return formatted_data

def load_query_responses_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    formatted_data = {}
    for entry in raw_data:
        query_text = entry["query"]
        formatted_data[query_text] = {
            "domain": entry.get("domain", "Unknown"),
            "subdomain": entry.get("subdomain", "Unknown"),
            "retrieved_context": entry.get("retrieved_context", ""),
            "response": entry.get("response", ""),
            "documents_in_response": entry.get("documents_in_response", [])
        }

    return formatted_data

def build_top_k_docs(rankings, k=5):
    return {
            query: info.get("ranked_ad_ids", [])[:k]
            for query, info in rankings.items()
        }

def load_classified_ads_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    formatted_data = {}
    for entry in raw_data:
        ad_id = entry["id"]
        formatted_data[ad_id] = {
            "domain": entry.get("domain", "Unknown"),
            "subdomain": entry.get("subdomain", "Unknown")
        }

    return formatted_data

def main(original_ads_file, rankings, query_responses, classified_ads, output_dir, batch_size, k):
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    
    with open(original_ads_file, "r", encoding="utf-8") as f:
        raw_ads = json.load(f)

    original_ads = load_original_ads_by_id(original_ads_file) # key is id
    rankings = load_rankings(rankings) #key is query
    responses = load_query_responses_from_json(query_responses) # key is query
    classified_ads = load_classified_ads_from_json(classified_ads) # key is id
    top_k_docs = build_top_k_docs(rankings, k) # key is query 

    def collate_fn(batch):
        original_ads_batch = [item["original_ad"] for item in batch] # this needs to be from raw ads

        inputs = tokenizer(
            [f"Original Ad: {ad}\nRewrite the ad:" for ad in original_ads_batch],
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "original_ads": original_ads_batch,
        }

    config = ReinforceConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=3,
        logging_steps=1,
        save_steps=100,
        learning_rate=5e-6
    )

    loss_fn = SimilarityLoss(alpha=1.0, beta=1.0, gamma=1.0)

    def reward_fn(samples, **kwargs):
        decoded_responses = tokenizer.batch_decode(samples["generated_responses"], skip_special_tokens=True)
        original_ads = samples["original_ads"]

        rewards = []
        for original, rewritten in zip(original_ads, decoded_responses):
            losses = []
            relevant_queries = []
            
            for query in relevant_queries:
                loss = loss_fn(query, original, rewritten, top_k_docs)
                losses.append(loss)
            rewards.append(sum(losses)/len(losses))

        return torch.tensor(rewards)

    trainer = ReinforceTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=original_ads,
        collate_fn=collate_fn,
        reward_fn=reward_fn,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model fine-tuned and saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA 3.1 8B using REINFORCE with custom loss.")
    parser.add_argument("--data_file", type=str, required=True, help="JSON file with queries, original ads, and top-k docs.")
    parser.add_argument("--rankings_file", type=str, required=True, help="Path to query rankings JSON.")
    parser.add_argument("--responses_file", type=str, required=True, help="Path to query responses JSON.")
    parser.add_argument("--ads_file", type=str, required=True, help="Path to classified ads JSON.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the fine-tuned model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    args = parser.parse_args()

    main(
        args.data_file,
        args.rankings_file,
        args.responses_file,
        args.ads_file,
        args.output_dir,
        args.batch_size
    )
