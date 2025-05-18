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
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
import bitsandbytes as bnb  
import argparse
from typing import Dict, List, Union, Optional

'''
pip install -U bitsandbytes accelerate

usage: 

python prompt_engineering.py \
  --ads_file 200_sampled_ads.json \
  --output_file prompt_output.json

python sft.py \
  --original_file train_data.json \
  --rewritten_file train_rewritten_ads.json \
  --output_dir sft_output \
  --batch_size 1 \
  --epochs 1

'''

class AdDataset(Dataset):
    def __init__(
        self, 
        original_file: str, 
        rewritten_file: str, 
        tokenizer,
        max_length: int = 512
    ):
        # Load original and rewritten data from separate files
        with open(original_file, 'r') as f:
            self.original_data = json.load(f)

        with open(rewritten_file, 'r') as f:
            self.rewritten_data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.original_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        original_document = self.original_data[idx]["text"]
        rewritten_document = self.rewritten_data[idx]["text"]

        prompt = f"Original document:\n{original_document}\n\nRewrite the document to improve retrieval for relevant queries:"
        completion = f" {rewritten_document}{self.tokenizer.eos_token}"
        
        # Combine into a full prompt
        full_text = prompt + completion
        
        # Tokenize the text
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create labels - set prompt tokens to -100 so they're ignored in loss calculation
        labels = encodings["input_ids"].clone()
        prompt_tokens = self.tokenizer(
            prompt, 
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"].shape[1]
        
        labels[0, :prompt_tokens] = -100
        
        return {
            "input_ids": encodings["input_ids"][0],
            "attention_mask": encodings["attention_mask"][0],
            "labels": labels[0]
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
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     device_map="auto",
    #     load_in_8bit=True,
    #     torch_dtype=torch.float16
    # )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Define LoRA config
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Inject LoRA adapters into the model
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_dataloader.batch_size,
        logging_dir=os.path.join(output_dir, "logs"),  
        logging_strategy="steps",
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        fp16=True,
        save_strategy="epoch",
        save_safetensors=True,
        optim="paged_adamw_8bit",
        gradient_accumulation_steps=8,
        overwrite_output_dir=True,
        label_names=["labels"],  # <-- Add this line
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
    model.save_pretrained(output_dir)
    
    return model

def get_sft_dataset(
    original_file: str, 
    rewritten_file: str, 
    tokenizer, 
    max_length: int = 512
) -> AdDataset:
    return AdDataset(original_file, rewritten_file, tokenizer, max_length)

def main(original_file: str, rewritten_file: str,  output_dir: str, batch_size: int, epochs: int) -> None:

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = get_sft_dataset(original_file, rewritten_file, tokenizer)
    
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
    parser.add_argument("--original_file", type=str, required=True, help="Path to JSON file containing original ads.")
    parser.add_argument("--rewritten_file", type=str, required=True, help="Path to JSON file containing rewritten ads.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model and tokenizer.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for fine-tuning.")
    
    args = parser.parse_args()
    
    main(args.original_file, args.rewritten_file, args.output_dir, args.batch_size, args.epochs)
