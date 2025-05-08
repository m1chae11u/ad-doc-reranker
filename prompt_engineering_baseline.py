import json
import google.generativeai as genai
import os
import argparse
from typing import List, Dict
from metric_inclusion import InclusionAccuracyMetric
from metric_retrieval import RetrievalMetric


"""
prompt engineering baseline

To run:
python prompt_engineering_baseline.py --ads_file ds/faiss_index/200_sampled_ads.json --queries_file query_responses_original_200.json --classified_file classified_ads_200.json --output_file prompt_output.json
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
        ad = a.get("text")
        prompt = create_prompt(ad)
        print(f"Generated prompt: {prompt}")

        response = model.generate_content(prompt)
        print(f"Response: {response.text}")

        rewritten.append({
            "original_ad": ad,
            "rewritten_ad": response.text.strip()
        })

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
    
    #Metrics
    
    # Evaluation: Inclusion Accuracy
    inclusion_metric = InclusionAccuracyMetric(
        k=10,
        rankings_before_path='rankings_before.json',
        rankings_after_path='rankings_after.json',
        inclusions_before_path='inclusions_before.json',
        inclusions_after_path='inclusions_after.json'
    )

    # Loop through each ad ID
    with open('rankings_before.json', 'r') as f:
        rankings_before = json.load(f)
    ad_ids = list(rankings_before.keys())

    for doc_id in ad_ids:
        inclusion_result = inclusion_metric.compute_inclusion_accuracy(doc_id)
        print(f"Inclusion Accuracy Improvement for {doc_id}: {inclusion_result}")

    # Evaluation: Retrieval (MRR@K)
    with open('rewritten_rankings.json', 'r') as f:
        rewritten_rankings = json.load(f)
    with open('original_rankings.json', 'r') as f:
        original_rankings = json.load(f)
    with open('target_docs.json', 'r') as f:
        target_doc = json.load(f)  # Format: { "doc_id": ("domain", "subdomain") }

    retrieval_metric = RetrievalMetric(target_doc, original_rankings, rewritten_rankings)

    for doc_id in target_doc:
        score = retrieval_metric.evaluate_doc(doc_id)
        print(f"ΔMRR@K for {doc_id}: {score}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewrite ads to improve general quality using prompt engineering.")
    parser.add_argument("--ads_file", type=str, required=True, help="Path to the original ads JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the rewritten ads JSON output.")

    args = parser.parse_args()
    main(args.ads_file, args.output_file)
