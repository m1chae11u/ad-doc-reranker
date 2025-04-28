import argparse
import os
import json
import torch
import copy
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
from metric_retrieval import RetrievalMetric
from metric_inclusion import InclusionAccuracyMetric
from tqdm import tqdm
from loss import SimilarityLoss

'''
Fine-tunes LLaMA 3.1 8B using REINFORCE and a custom loss function as reward.

Usage:
python pipeline.py --data_file sampled_ads_200.json --rankings_file rankings.json --responses_file query_responses_original_200.json --ads_file classified_ads_200.json --output_dir finetuned_model --batch_size 1 --k 10
'''

class CustomRewardModel(torch.nn.Module):
    def __init__(self, tokenizer, base_model, similarity_loss_fn):
        super().__init__()
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.similarity_loss_fn = similarity_loss_fn

    def forward(self, raw_ads, decoded_responses, top_k_docs, classified_ads, responses):
        all_gen_ads = []
        
        for ad in tqdm(raw_ads):
            input_ids = self.tokenizer(ad['text'], return_tensors="pt", padding=True, truncation=True).input_ids.cuda()

            # Generate response from the model
            gen_ids = self.base_model.generate(
                input_ids,
                max_new_tokens=50,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Decode generated text
            gen_text = self.tokenizer.decode(gen_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
            all_gen_ads.append({"ad_id": ad["ad_id"], "rewrite": gen_text})

        total_losses = []
        for original, rewritten in zip(raw_ads, decoded_responses):
            relevant_queries = []  # Find relevant queries for the ad
            for query, q_info in responses.items():
                ad_domain = classified_ads.get(original['ad_id'], {}).get("domain")
                ad_subdomain = classified_ads.get(original['ad_id'], {}).get("subdomain")
                if q_info.get("domain") == ad_domain and q_info.get("subdomain") == ad_subdomain:
                    relevant_queries.append(query)

            # Calculate loss for relevant queries
            losses = []
            sample_size = min(len(relevant_queries), 8)
            for query in random.sample(relevant_queries, sample_size):
                docs_for_query = top_k_docs.get(query, [])
                loss = self.similarity_loss_fn(query, original, rewritten, docs_for_query)
                losses.append(loss)
            total_losses.append(sum(losses) / len(losses))
        
        # delta_mrr = RetrievalMetric(ad["ad_id"], queries, rankings, rewritten_rankings).evaluate_doc(ad)
        # delta_dir = InclusionAccuracyMetric(
        #     k=10,
        #     rankings_before_dict=rankings,
        #     rankings_after_dict=rewritten_rankings,
        #     inclusions_before_dict=responses,  # original
        #     inclusions_after_dict=responses_after    # new
        # ).compute_inclusion_accuracy(ad["ad_id"])
        # print(f"Epoch {epoch+1}: ΔMRR@10 {delta_mrr:.4f}, ΔDIR@10 {delta_dir:.2f}%")

        return -(sum(total_losses) / len(total_losses)) 

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

def format_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    formatted_queries = []
    for item in raw_data:
        formatted_queries.append({
            "query": item["query"],  
            "domain": item["domain"],
            "subdomain": item["subdomain"]
        })
    return formatted_queries

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

def build_top_k_docs(rankings, original_ads_by_id):
    return {
        query: [original_ads_by_id[ad_id] for ad_id in info.get("ranked_ad_ids", []) if ad_id in original_ads_by_id]
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

def main(original_ads_file, rankings_file, query_responses_file, classified_ads_file, output_dir, batch_size, k):
    model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

    # Load the base model for PPO
    base = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16).cuda()
    # base = AutoModelForCausalLMWithValueHead.from_pretrained(model, torch_dtype=torch.float16).cuda() 

    # Lora config setup
    lora_cfg = LoraConfig(r=32, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    peft_model = get_peft_model(base, lora_cfg)  # PEFT params will train

    # base.pretrained_model.generation_config.eos_token_id = tokenizer.eos_token_id

    # Create a reference model without LoRA modifications (used for PPO)
    ref_model = copy.deepcopy(peft_model).eval()
    
    with open(original_ads_file, "r", encoding="utf-8") as f:
        raw_ads = json.load(f)

    original_ads = load_original_ads_by_id(original_ads_file) # key is id
    rankings = load_rankings(rankings_file) # key is query
    queries = format_queries(query_responses_file) # list of queries w/ domain & subdomain
    responses = load_query_responses_from_json(query_responses_file) # key is query
    classified_ads = load_classified_ads_from_json(classified_ads_file) # key is id
    top_k_docs = build_top_k_docs(rankings, original_ads) # key is query, mapped to list of ids

    config = PPOConfig(
        model_adapter_name=base,
        learning_rate=5e-6,
        batch_size=4,  
        mini_batch_size=1,
        logging_steps=1,
        save_steps=100,
        output_dir="./ppo_output",
    )
    

    similarity_loss_fn = SimilarityLoss(alpha=1.0, beta=1.0, gamma=1.0)
    reward_model = CustomRewardModel(tokenizer, base, similarity_loss_fn)

    train_dataset = [
        tokenizer(ad["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        for ad in raw_ads
    ]

    trainer = PPOTrainer(
        args=config,
        model=base,
        ref_model=ref_model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        reward_model=CustomRewardModel, 
        value_model=base,
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
    parser.add_argument("--k", type=int, default=5, help="Top-k documents")  
    args = parser.parse_args()

    main(
        args.data_file,
        args.rankings_file,
        args.responses_file,
        args.ads_file,
        args.output_dir,
        args.batch_size,
        args.k
    )
