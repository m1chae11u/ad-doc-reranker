import argparse
import os
import json
import torch
import torch.nn as nn
import copy
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, TextEnvironment
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
from rank_documents import DocumentRanker
from metric_retrieval import RetrievalMetric
from metric_inclusion import InclusionAccuracyMetric
from tqdm import tqdm
from loss import SimilarityLoss

'''
Fine-tunes LLaMA 3.1 8B using REINFORCE and a custom loss function as reward.

Usage:
python pipeline.py --data_file sampled_ads.json --rankings_file rankings.json --responses_file query_responses_original_200.json --ads_file classified_ads.json --output_dir finetuned_model --batch_size 1 --k 10
'''

class DummyRewardModel(nn.Module):
    def forward(self, *args, **kwargs):
        return torch.tensor([0.0])
    def score(self, hidden_states):
        batch_size = hidden_states.shape[0]
        return torch.zeros(batch_size).to(hidden_states.device)

class CustomPPOTrainer(PPOTrainer):
    def __init__(self, *args, reward_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_fn = reward_fn

    def train(self):
        for epoch in range(int(self.args.num_train_epochs)):
            for batch in self.dataloader:
                prompts = batch["input_ids"]
                # Generate responses (you can customize sampling)
                responses = self.model.value_model.generate(prompts)

                # Convert to strings if needed
                prompt_texts = self.tokenizer.batch_decode(prompts, skip_special_tokens=True)
                response_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

                # Rule-based rewards
                # rewards = self.reward_fn(prompt_texts, response_texts)
                rewards = [1]*200 #dummy reward for testing...
                
                # Run PPO step
                # self.step(prompts, responses, rewards)
                self.model.train()
                stats = self.step(queries, responses, rewards)
                self.tokenizer.padding_side = "left"  # restore padding side
                loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
                reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))
                
    # def __init__(self, *args, raw_ads, classified_ads, original_responses, similarity_loss_fn, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.raw_ads = raw_ads
    #     self.classified_ads = classified_ads
    #     self.original_responses = original_responses
    #     self.similarity_loss_fn = similarity_loss_fn

    # def get_rewards(self, decoded_responses, top_k_docs, **kwargs):
    #     total_losses = []
    #     for original, rewritten in zip(self.raw_ads, decoded_responses):
    #         relevant_queries = []  # Find relevant queries for the ad
    #         for query, q_info in self.original_responses.items():
    #             ad_domain = self.classified_ads.get(original['ad_id'], {}).get("domain")
    #             ad_subdomain = self.classified_ads.get(original['ad_id'], {}).get("subdomain")
    #             if q_info.get("domain") == ad_domain and q_info.get("subdomain") == ad_subdomain:
    #                 relevant_queries.append(query)

    #         # Calculate loss for relevant queries
    #         losses = []
    #         sample_size = min(len(relevant_queries), 8)
    #         for query in random.sample(relevant_queries, sample_size):
    #             docs_for_query = top_k_docs.get(query, [])
    #             loss = self.similarity_loss_fn(query, original, rewritten, docs_for_query)
    #             losses.append(loss)
    #         total_losses.append(sum(losses) / len(losses))
        
    #     # delta_mrr = RetrievalMetric(ad["ad_id"], queries, rankings, rewritten_rankings).evaluate_doc(ad)
    #     # delta_dir = InclusionAccuracyMetric(
    #     #     k=10,
    #     #     rankings_before_dict=rankings,
    #     #     rankings_after_dict=rewritten_rankings,
    #     #     inclusions_before_dict=responses,  # original
    #     #     inclusions_after_dict=responses_after    # new
    #     # ).compute_inclusion_accuracy(ad["ad_id"])
    #     # print(f"Epoch {epoch+1}: ΔMRR@10 {delta_mrr:.4f}, ΔDIR@10 {delta_dir:.2f}%")
        
    #     # return -(sum(total_losses) / len(total_losses)) 
    #     return total_losses

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
    model = "sft_output"
    tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

    # Load the base model for PPO
    #base = AutoModelForCausalLMWithValueHead.from_pretrained(model, torch_dtype=torch.float16).cuda() 
    base = AutoModelForCausalLMWithValueHead.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        return_dict=True
    ).cuda() 

    # Lora config setup
    lora_cfg = LoraConfig(r=32, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    peft_model = get_peft_model(base, lora_cfg)  # PEFT params will train
    peft_model.pretrained_model.config.return_dict = True
    peft_model.config.return_dict = True

    # Create a reference model without LoRA modifications (used for PPO)
    #ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model, torch_dtype=torch.float16).cuda()
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model, 
        torch_dtype=torch.float16,
        return_dict=True
    ).cuda()
    ref_model.pretrained_model.config.return_dict = True
    ref_model.config.return_dict = True

    # reward_model = RewardModel(peft_model)

    with open(original_ads_file, "r", encoding="utf-8") as f:
        raw_ads = json.load(f)

    original_ads = load_original_ads_by_id(original_ads_file) # key is id
    rankings = load_rankings(rankings_file) # key is query
    queries = format_queries(query_responses_file) # list of queries w/ domain & subdomain
    responses = load_query_responses_from_json(query_responses_file) # key is query
    classified_ads = load_classified_ads_from_json(classified_ads_file) # key is id
    top_k_docs = build_top_k_docs(rankings, original_ads) # key is query, mapped to list of ids
    
    config = PPOConfig(
        model_adapter_name=peft_model,
        learning_rate=5e-6,
        batch_size=4,  
        mini_batch_size=1,
        logging_steps=1,
        save_steps=100,
        output_dir="./ppo_output",
    )
    
    similarity_loss_fn = SimilarityLoss(alpha=1.0, beta=1.0, gamma=1.0)
    # reward_model = CustomRewardModel(tokenizer, raw_ads, classified_ads, queries, similarity_loss_fn)
    def compute_reward(raw_ads, decoded_responses):
        total_losses = []
        for original, rewritten in zip(raw_ads, decoded_responses):
            relevant_queries = [] # queries that have same domain subdomain as document
            for query, q_info in responses.items():
                ad_domain = classified_ads.get(original['ad_id'],{}).get("domain")
                ad_subdomain = classified_ads.get(original['ad_id'],{}).get("subdomain")
                if q_info.get("domain") == ad_domain and q_info.get("subdomain") == ad_subdomain:
                    relevant_queries.append(query)
 
            losses = []
            for query in random.sample(relevant_queries, 8):
                loss = loss_fn(query, original, rewritten, top_k_docs)
                losses.append(loss)
            total_losses.append(sum(losses)/len(losses))
        return -(sum(total_losses) / len(total_losses))

    trainer = PPOTrainer(args=config, model=base, ref_model=ref_model, processing_class=tokenizer, train_dataset=None, reward_model=None, value_model=None)
    loss_fn = SimilarityLoss(alpha=1.0, beta=1.0, gamma=1.0)

    for epoch in range(3):  # number of PPO passes
        print(f"Epoch {epoch + 1}")
        all_gen_ads = []
        for ad in tqdm(raw_ads):
            # Step 1: Tokenize input
            input_ids = tokenizer(ad, return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
            
            # Step 2: Generate response from the model
            gen_ids = trainer.model.generate(
                input_ids,
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Step 3: Extract text response
            gen_text = tokenizer.decode(gen_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
            all_gen_ads.append({"ad_id": ad["ad_id"], "rewrite": gen_text})

            

            responses_after = 
            rewritten_rankings = DocumentRanker(index_dir="./ds/faiss_index_rewritten/", top_k=10)
            
            delta_mrr = RetrievalMetric(ad["ad_id"], queries, rankings, rewritten_rankings).evaluate_doc(ad)
            delta_dir = InclusionAccuracyMetric(
                k=10,
                rankings_before_dict=rankings,
                rankings_after_dict=rewritten_rankings,
                inclusions_before_dict=responses,  # original
                inclusions_after_dict=responses_after    # new
            ).compute_inclusion_accuracy(ad["ad_id"])
            print(f"Epoch {epoch+1}: ΔMRR@10 {delta_mrr:.4f}, ΔDIR@10 {delta_dir:.2f}%")

        # Step 4: Compute reward
        rewards = torch.tensor( [compute_reward([orig], [gen]) for orig, gen in zip(raw_ads, all_gen_ads)]
        ).to(model.device)

        # Step 5: Pass into PPO step
        trainer.step(input_ids, gen_ids, rewards)

    trainer = CustomPPOTrainer(
        args=config,
        model=peft_model.pretrained_model,
        ref_model=ref_model.pretrained_model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        reward_model=DummyRewardModel(), 
        value_model=peft_model.pretrained_model,
        # raw_ads=raw_ads,
        # classified_ads=classified_ads,
        # original_responses=responses,
        # similarity_loss_fn=similarity_loss_fn,
        reward_fn=compute_reward
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