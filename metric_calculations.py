import json
import google.generativeai as genai
import os
import argparse
from typing import List, Dict
from metric_inclusion import InclusionAccuracyMetric
from metric_retrieval import RetrievalMetric
from build_index import IndexBuilder
from rank_documents import DocumentRanker
from rag import RAGGenerator

def load_query_responses_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    formatted_data = {}
    for entry in raw_data:
        query_text = entry["query"]
        formatted_data[query_text] = entry.get("documents_in_response", [])

    return formatted_data

with open("sampled_ads.json", 'r', encoding='utf-8') as f:
    ads = json.load(f)

# create faiss index
indexer = IndexBuilder(input_path="prompt_output.json", output_dir="faiss_index_rewritten")
indexer.run()

# rank rewritten docs
with open("queries.json", "r", encoding="utf-8") as f:
    queries = json.load(f)
ranker = DocumentRanker(index_dir="faiss_index_rewritten", top_k=10)
ranker.rank_and_save(queries, "rankings_rewritten.json")

# generate rewritten responses 
generator = RAGGenerator()
generator.batch_generate(
    query_file="queries_200.json",
    index_dir="faiss_index_rewritten",
    output_file="query_responses_rewritten.json",
    top_k=10,
    use_full_docs=True
)

# Evaluation: Inclusion Accuracy
inclusion_metric = InclusionAccuracyMetric(
    k=10,
    rankings_before_path='rankings_rewritten.json',
    rankings_after_path='rankings_original.json',
    inclusions_before_path='query_responses_original.json',
    inclusions_after_path='query_responses_rewritten.json'
)

# Loop through each ad ID
inclusion_results = []
for ad in ads:
    inclusion_result = inclusion_metric.compute_inclusion_accuracy(ad['ad_id'])
    print(f"Inclusion Accuracy Improvement for {ad['ad_id']}: {inclusion_result}")
    inclusion_results.append(inclusion_result)
print (f"Average inclusion improvement: {sum(inclusion_results)/len(inclusion_results)}")

# Evaluation: Retrieval (MRR@K)
original_rankings = load_query_responses_from_json("rankings_original.json")
rewritten_rankings = load_query_responses_from_json("rankings_rewritten.json")
with open('classified_ads.json', 'r') as f:
    classified_ads = json.load(f)  
    
formatted_classified_ads = {} # Format: { "doc_id": ("domain", "subdomain") }
for ad in ads:
    ad_id = ad["id"]
    formatted_classified_ads[ad_id] = {
        "domain": ad.get("domain", "Unknown"),
        "subdomain": ad.get("subdomain", "Unknown")
    }

retrieval_metrics = []
for ad, classified in zip(ads, formatted_classified_ads):
    retrieval_metric = RetrievalMetric(classified, queries, original_rankings, rewritten_rankings)
    score = retrieval_metric.evaluate_doc(ad['ad_id'])
    print(f"ΔMRR@K for {ad["ad_id"]}: {score}")
    retrieval_metrics.append(retrieval_metric)
print (f"ΔMRR@K: {sum(retrieval_metrics)/len(retrieval_metrics)}")
