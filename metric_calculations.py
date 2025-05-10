import json
import google.generativeai as genai
import os
import argparse
from typing import List, Dict
from metric_inclusion import InclusionAccuracyMetric
from metric_retrieval import RetrievalMetric
from data_processing.build_index import IndexBuilder
from rank_documents import DocumentRanker
from rag import RAGGenerator

def load_query_responses_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    formatted_data = {}
    for entry in raw_data:
        query_text = entry["query"]["query"]
        formatted_data[query_text] = entry["ranked_ad_ids"]

    return formatted_data

with open("ds/faiss_index/200_sampled_ads.json", 'r', encoding='utf-8') as f:
    ads = json.load(f)
with open("queries_200.json", "r", encoding="utf-8") as f:
    queries = json.load(f)

# create faiss index
indexer = IndexBuilder(input_path="aprompt_output.json", output_dir="faiss_index_rewritten")
indexer.run()

# rank rewritten docs
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
    rankings_before_path='rankings_original.json',
    rankings_after_path='rankings_rewritten.json',
    inclusions_before_path='query_responses_original_200.json',
    inclusions_after_path='query_responses_rewritten.json'
)

# Loop through each ad ID
inclusion_results = []
for ad in ads:
    inclusion_result = inclusion_metric.compute_inclusion_accuracy(ad['ad_id'])
    # print(f"Inclusion Accuracy Improvement for {ad['ad_id']}: {inclusion_result}")
    inclusion_results.append(inclusion_result)
print (f"Average inclusion improvement: {sum(inclusion_results)/len(inclusion_results)}")

# Evaluation: Retrieval (MRR@K)
original_rankings = load_query_responses_from_json("rankings_original.json")
rewritten_rankings = load_query_responses_from_json("rankings_rewritten.json")
with open('classified_ads_200.json', 'r') as f:
    classified_ads = json.load(f)  
    
formatted_classified_ads = {} # Format: { "doc_id": ("domain", "subdomain") }
for ad in classified_ads:
    ad_id = ad["id"]
    formatted_classified_ads[ad_id] = {
        "domain": ad.get("domain"),
        "subdomain": ad.get("subdomain")
    }

retrieval_metrics = []
for ad in ads:
    target_doc = {ad["ad_id"]: ( formatted_classified_ads[ad['ad_id']].get("domain"), formatted_classified_ads[ad['ad_id']].get("subdomain"))}
    retrieval_metric = RetrievalMetric(target_doc, queries, original_rankings, rewritten_rankings)
    score = retrieval_metric.evaluate_doc(ad['ad_id'])
    # print(f"ΔMRR@K for {ad["ad_id"]}: {score}")
    retrieval_metrics.append(score)
print (f"ΔMRR@K: {sum(retrieval_metrics)/len(retrieval_metrics)}")
