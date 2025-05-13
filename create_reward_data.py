import json
import argparse
import os
import shutil
from typing import List, Dict, Any

"""
python create_reward_data.py \
  --original_ads_file 200_sampled_ads.json \
  --rewritten_ads_file aprompt_output.json \
  --output_dir data/ad_reward
"""

def create_reward_dataset(
    original_ads_file: str,
    rewritten_ads_file: str,
    output_dir: str
) -> None:
    """
    Convert advertisement data to the preference dataset format required for 
    LLaMA-Factory reward model training.
    
    Args:
        original_ads_file: Path to original ads JSON file
        rewritten_ads_file: Path to rewritten ads JSON file
        output_dir: Directory to output the preference dataset and dataset_info.json
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    with open(original_ads_file, 'r', encoding='utf-8') as f:
        original_ads = json.load(f)
    
    with open(rewritten_ads_file, 'r', encoding='utf-8') as f:
        rewritten_ads = json.load(f)
    
    # Create preference dataset format according to LLaMA-Factory specs
    preference_data = []
    
    # Process each ad pair
    for i, (orig_ad, rewrite_ad) in enumerate(zip(original_ads, rewritten_ads)):
        original_text = orig_ad.get("text", "")
        rewritten_text = rewrite_ad.get("text", "")
        
        # For each ad, create an entry where the rewritten ad is "chosen"
        # and the original ad is "rejected" since we want to train the model to prefer
        # the rewritten versions
        entry = {
            "instruction": "Rewrite the advertisement to improve retrieval for relevant queries:",
            "input": original_text,
            "chosen": rewritten_text,
            "rejected": original_text
        }
        
        preference_data.append(entry)
    
    # Save the preference dataset
    output_file = os.path.join(output_dir, "train.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(preference_data, f, indent=2, ensure_ascii=False)
    
    # Create dataset_info.json according to LLaMA-Factory specs
    dataset_info = {
        "ad_reward": {
            "file_name": "train.json",
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "chosen": "chosen",
                "rejected": "rejected"
            }
        }
    }
    
    # Save dataset_info.json
    info_file = os.path.join(output_dir, "dataset_info.json")
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"Created reward model dataset with {len(preference_data)} examples at {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert ad data to preference format for LLaMA-Factory reward model training")
    parser.add_argument("--original_ads_file", type=str, required=True, help="Path to original ads JSON file")
    parser.add_argument("--rewritten_ads_file", type=str, required=True, help="Path to rewritten ads JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to output the dataset files")
    
    args = parser.parse_args()
    
    create_reward_dataset(
        args.original_ads_file,
        args.rewritten_ads_file,
        args.output_dir
    )

if __name__ == "__main__":
    main()
