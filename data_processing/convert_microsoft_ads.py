"""
convert_microsoft_ads.py

This script converts the Microsoft CommercialAdsDataset to the format expected by our RAG system.

Usage:
    python data_processing/convert_microsoft_ads.py --input_dir /path/to/commercial_ads_dataset --output_file commercial_ads.json
"""

import argparse
import json
import os
import csv
import glob
import uuid

def convert_microsoft_ads(input_dir: str, output_file: str):
    """
    Convert Microsoft CommercialAdsDataset to our format.
    
    Args:
        input_dir: Directory containing the Microsoft CommercialAdsDataset
        output_file: Output JSON file path
    """
    all_ads = []
    
    # Look for files in the dataset directory
    data_files = glob.glob(os.path.join(input_dir, "*.tsv"))
    if not data_files:
        data_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not data_files:
        raise ValueError(f"No data files found in {input_dir}")
    
    print(f"Found {len(data_files)} data files")
    
    for file_path in data_files:
        print(f"Processing {file_path}...")
        
        # Determine the delimiter based on file extension
        delimiter = '\t' if file_path.endswith('.tsv') else ','
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Use csv reader without headers since dataset has no column names
            reader = csv.reader(f, delimiter=delimiter)
            
            for row in reader:
                # Skip rows that don't have enough columns
                if len(row) < 8:
                    continue
                
                # Map columns to their meaning based on your information
                # Column indices are 0-based, so column 3 is index 2, etc.
                ad = {
                    'user_query': row[2].strip(),
                    'title': row[3].strip(),
                    'text': row[4].strip(),  # ad_description
                    'url': row[5].strip(),
                    'seller': row[6].strip(),
                    'brand': row[7].strip(),
                    'source': 'microsoft_commercial_ads',
                    'ad_id': str(uuid.uuid4())
                }
                
                # Only add ads with content
                if ad['text'] and ad['title']:
                    all_ads.append(ad)
    
    # Save the converted data to the output file
    print(f"Saving {len(all_ads)} ads to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_ads, f, ensure_ascii=False, indent=2)
    
    print(f"Conversion complete!")

def main():
    parser = argparse.ArgumentParser(
        description="Convert Microsoft CommercialAdsDataset to our format."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the Microsoft CommercialAdsDataset."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="commercial_ads.json",
        help="Output JSON filename. Default is 'commercial_ads.json'."
    )
    
    args = parser.parse_args()
    
    # If output_file doesn't have directory information, save in the input_dir
    if not os.path.dirname(args.output_file):
        output_file = os.path.join(args.input_dir, args.output_file)
    else:
        output_file = args.output_file
        # Ensure the output directory exists if specified
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    convert_microsoft_ads(args.input_dir, output_file)

if __name__ == "__main__":
    main()
