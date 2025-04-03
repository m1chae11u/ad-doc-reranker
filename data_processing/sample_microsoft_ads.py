"""
sample_microsoft_ads.py

This script samples a subset of the Microsoft CommercialAdsDataset to create a smaller dataset for
testing and development purposes.

Usage:
    python data_processing/sample_microsoft_ads.py --input_file /full/path/to/commercial_ads.json --output_file /full/path/to/sampled_ads.json --sample_size 1000
"""

import argparse
import json
import os
import random
from typing import List, Dict, Any

def sample_ads(input_file: str, output_file: str, sample_size: int, random_seed: int = 42):
    """
    Sample a subset of ads from the full dataset.
    
    Args:
        input_file: Path to the full dataset JSON file
        output_file: Path to save the sampled dataset
        sample_size: Number of ads to sample
        random_seed: Random seed for reproducibility
    """
    print(f"Loading ads from {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            all_ads = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading input file: {e}")
    
    total_ads = len(all_ads)
    print(f"Loaded {total_ads} ads")
    
    # Adjust sample size if needed
    if sample_size > total_ads:
        print(f"Warning: Requested sample size ({sample_size}) is larger than the dataset size ({total_ads})")
        sample_size = total_ads
        print(f"Using full dataset instead")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Sample ads
    print(f"Sampling {sample_size} ads ({(sample_size/total_ads)*100:.2f}% of original dataset)...")
    sampled_ads = random.sample(all_ads, sample_size)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save sampled ads
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_ads, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully saved {len(sampled_ads)} sampled ads to {output_file}")
    
    # Print some statistics
    categories = {}
    brands = set()
    
    for ad in sampled_ads:
        brand = ad.get('brand', '')
        if brand:
            brands.add(brand)
    
    print(f"Sample includes {len(brands)} unique brands")

def main():
    parser = argparse.ArgumentParser(
        description="Sample a subset of the Microsoft CommercialAdsDataset."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Full path to the input JSON file containing all ads."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Full path to save the sampled ads JSON file."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Number of ads to sample. Default is 1000."
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default is 42."
    )
    
    args = parser.parse_args()
    sample_ads(args.input_file, args.output_file, args.sample_size, args.random_seed)

if __name__ == "__main__":
    main()