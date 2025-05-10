import json
import google.generativeai as genai
import os
import argparse
import re
from typing import List, Dict

"""
prompt engineering baseline

To run:
python prompt_engineering.py --ads_file ds/faiss_index/200_sampled_ads.json --output_file prompt_output.json
"""

def load_api_key() -> str:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "keys.json")
    with open(config_path, 'r') as f:
        return json.load(f)["google_api_key"]

def initialize_gemini():
    api_key = load_api_key()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-pro")

def create_prompt(ad: str) -> str:
    return f"""You are given an advertisement. Please rewrite it so that it is more likely to rank higher when retrieved by a search system for queries relevant to its content.

Make sure not to add any new information or make assumptions that are not already present in the original ad. 

Original Ad:
{ad}

Rewritten Ad:"""

def rewrite_ads(ads: List[Dict], model) -> List[Dict]:
    rewritten = []

    for a in ads:
        ad = f"Title: {a.get('title', '')}\n\nDescription: {a.get('text', '')}"
        prompt = create_prompt(ad)
        print(f"Generated prompt: {prompt}")

        response = model.generate_content(prompt)
        # print(f"{response.text}")
        
        title_match = re.search(r'Title:\s*(.*)', response.text)
        description_match = re.search(r'Description:\s*(.*)', response.text, re.DOTALL)

        title = title_match.group(1).strip() if title_match else ""
        description = description_match.group(1).strip() if description_match else ""

        rewritten.append({
            "user_query": a['user_query'],
            "title": title,
            "text": description,
            "url": a['url'],
            "seller": a['seller'],
            "brand": a['brand'],
            "source": a['source'],
            "ad_id": a['ad_id']
        })
        print(f"title: {title} \n\ndescription: {description}")

    return rewritten

def main(ads_file: str, output_file: str):
    # Load ads
    with open(ads_file, 'r', encoding='utf-8') as f:
        ads = json.load(f)

    model = initialize_gemini()
    rewritten = rewrite_ads(ads, model)

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
