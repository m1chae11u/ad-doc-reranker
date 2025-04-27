import argparse
import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

class AdRewriteDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ad = self.data[idx]
        input_text = ad["original"]
        target_text = ad["rewrite"]

        inputs = self.tokenizer(input_text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        labels = self.tokenizer(target_text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100  # mask pad tokens

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }

def train_sft(data_path, output_dir, batch_size=1):
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    lora_cfg = LoraConfig(r=32, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(base_model, lora_cfg)

    with open(data_path, "r", encoding="utf-8") as f:
        raw_ads = json.load(f)

    dataset = AdRewriteDataset(raw_ads, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=200,
        logging_steps=50,
        save_steps=500,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        bf16=True,
        save_total_limit=2,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def parse_args_and_run():
    parser = argparse.ArgumentParser(description="Run supervised fine-tuning (SFT) on ad rewrites.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the ad rewrite JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output dir for model artifacts.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")

    args = parser.parse_args()
    train_sft(args.data_file, args.output_dir, args.batch_size)

if __name__ == "__main__":
    parse_args_and_run()
