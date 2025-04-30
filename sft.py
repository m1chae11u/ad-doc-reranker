import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import argparse
from typing import Dict, List, Union, Optional

'''
pip install -U bitsandbytes accelerate

usage: 
python sft.py --json_file faiss_index/sampled_ads_200.json --output_dir sft_output --batch_size 1 --epochs 3

'''

class AdDataset(Dataset):
    def __init__(self, 
                 json_file: str, 
                 tokenizer,
                 max_length: int = 512):
        with open(json_file, 'r') as file:
            self.data = json.load(file)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ad = self.data[idx]
        original_document = ad['text'] if 'text' in ad else ""
        
        prompt = f"Original document:\n{original_document}\n\nRewrite the document:"
        completion = original_document
        
        full_text = f"{prompt} {completion}{self.tokenizer.eos_token}"
        
        # Tokenize the text
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"][0],
            "attention_mask": encodings["attention_mask"][0],
            "labels": encodings["input_ids"][0].clone() 
        }

def train_sft_model(
    model_name: str,
    train_dataloader: DataLoader,
    output_dir: str,
    epochs: int,
    logging_steps: int = 100,
    save_steps: int = 500,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    fp16: bool = True,
    gradient_accumulation_steps: int = 1
) -> AutoModelForCausalLM:
    
    # Load the pre-trained model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_dataloader.batch_size,
        #save_strategy="steps",
        #save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        bf16=True, 
        save_strategy="no",
        save_only_model=True,
        save_safetensors=False,
        optim="paged_adamw_8bit",
        gradient_accumulation_steps=gradient_accumulation_steps,
        overwrite_output_dir=True,
        #save_total_limit=3,  # Keep only the 3 most recent checkpoints
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=train_dataloader.dataset.tokenizer,
            mlm=False  # We're not using masked language modeling
        )
    )
    
    trainer.train()
    import shutil

    def safe_model_save(model, output_dir, min_required_gb=10):
        total, used, free = shutil.disk_usage(output_dir)
        free_gb = free // (2**30)
        print(f"[INFO] Available disk space in '{output_dir}': {free_gb} GB")

        if free_gb < min_required_gb:
            print(f"[WARNING] Low disk space (<{min_required_gb}GB). Saving model to fallback directory: /tmp/llama_sft_output")
            fallback_dir = "/tmp/llama_sft_output"
            os.makedirs(fallback_dir, exist_ok=True)
            output_dir = fallback_dir

        try:
            model.save_pretrained(output_dir, safe_serialization=False, max_shard_size="2GB")
            print(f"[SUCCESS] Model saved to: {output_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")

    safe_model_save(model, output_dir)

    # model.save_pretrained(output_dir, safe_serialization=False, max_shard_size="5GB")  
    
    return model

def get_sft_dataset(
    json_file: str, 
    tokenizer, 
    max_length: int = 512
) -> AdDataset:
    return AdDataset(json_file, tokenizer, max_length)

def main(json_file: str, output_dir: str, batch_size: int, epochs: int) -> None:

    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = get_sft_dataset(json_file, tokenizer)
    
    train_dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    fine_tuned_model = train_sft_model(
        model_name=model_name,
        train_dataloader=train_dataloader,
        output_dir=output_dir,
        epochs=epochs
    )
    
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fine-tune a language model on self-supervised task using ad data.")
    parser.add_argument("--json_file", type=str, required=True, help="Path to JSON file containing ad data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model and tokenizer.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for fine-tuning.")
    
    args = parser.parse_args()
    
    main(args.json_file, args.output_dir, args.batch_size, args.epochs)
