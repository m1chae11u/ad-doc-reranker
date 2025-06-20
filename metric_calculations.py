import json
import google.generativeai as genai
import os
import asyncio
from tqdm.asyncio import tqdm_asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from metric_inclusion import InclusionAccuracyMetric
from metric_retrieval import RetrievalMetric
from retriever import AdSiteRetriever
from data_processing.build_index import IndexBuilder
from rank_documents import DocumentRanker
from rag import RAGGenerator


class MetricEvaluator:
    def __init__(
        self,
        original_ads_path: str,
        queries_path: str,
        index_input_path: str,
        index_output_dir: str,
        original_rankings_path: str,
        rewritten_rankings_path: str,
        original_responses_path: str,
        rewritten_responses_path: str,
        classified_ads_path: str,
        k: int = 10,
    ):
        self.original_ads_path = original_ads_path
        self.queries_path = queries_path
        self.index_input_path = index_input_path
        self.index_output_dir = index_output_dir
        self.original_rankings_path = original_rankings_path
        self.rewritten_rankings_path = rewritten_rankings_path
        self.original_responses_path = original_responses_path
        self.rewritten_responses_path = rewritten_responses_path
        self.classified_ads_path = classified_ads_path
        self.k = k

    @staticmethod
    def load_json(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_query_responses(path: str) -> Dict[str, List[str]]:
        raw_data = MetricEvaluator.load_json(path)
        return {entry["query"]["query"]: entry["ranked_ad_ids"] for entry in raw_data}

    def build_index(self):
        indexer = IndexBuilder(input_path=self.index_input_path, output_dir=self.index_output_dir)
        indexer.run()

    def rank_documents(self, output_file: str):
        queries = self.load_json(self.queries_path)
        ranker = DocumentRanker(index_dir=self.index_output_dir, top_k=self.k, original_file=self.index_input_path)
        ranker.rank_and_save(queries, output_file)

    def generate_responses(self):
        generator = RAGGenerator()
        generator.batch_generate(
            query_file=self.queries_path,
            index_dir=self.index_output_dir,
            output_file=self.rewritten_responses_path,
            top_k=self.k,
            use_full_docs=True,
            original_file=self.original_ads_path
        )

    def evaluate_inclusion_accuracy(self, ads: List[Dict]):
        inclusion_metric = InclusionAccuracyMetric(
            k=self.k,
            rankings_before_path=self.original_rankings_path,
            rankings_after_path=self.rewritten_rankings_path,
            inclusions_before_path=self.original_responses_path,
            inclusions_after_path=self.rewritten_responses_path
        )
        scores = [inclusion_metric.compute_inclusion_accuracy(ad["ad_id"]) for ad in ads]
        avg = sum(scores) / len(scores)
        print(f"Average inclusion improvement: {avg:.4f}")
        return avg

    def evaluate_retrieval(self, ads: List[Dict], queries: List[Dict]):
        original_rankings = self.load_query_responses(self.original_rankings_path)
        rewritten_rankings = self.load_query_responses(self.rewritten_rankings_path)
        classified_ads = self.load_json(self.classified_ads_path)

        formatted_classified_ads = {
            ad["id"]: {
                "domain": ad.get("domain"),
                "subdomain": ad.get("subdomain")
            }
            for ad in classified_ads
        }

        scores = []
        for ad in ads:
            ad_id = ad["ad_id"]
            domain_info = formatted_classified_ads.get(ad_id, {})
            target_doc = {ad_id: (domain_info.get("domain"), domain_info.get("subdomain"))}
            metric = RetrievalMetric(target_doc, queries, original_rankings, rewritten_rankings)
            score = metric.evaluate_doc(ad_id)
            scores.append(score)

        avg = sum(scores) / len(scores)
        print(f"ΔMRR@K: {avg:.4f}")
        return avg

    def run(self):
        print(f"Loading data for {self.index_input_path} {self.original_rankings_path}...")
        ads = self.load_json(self.original_ads_path)
        queries = self.load_json(self.queries_path)

        # print("Building index...")
        self.build_index()

        print("Ranking documents...")
        self.rank_documents(self.rewritten_rankings_path)

        print("Generating responses...")
        self.generate_responses()

        print("Evaluating Inclusion Accuracy...")
        self.evaluate_inclusion_accuracy(ads)

        print("Evaluating Retrieval (MRR@K)...")
        self.evaluate_retrieval(ads, queries)


# Example usage:
if __name__ == "__main__":
    # index_input_path is rewritten ads


    # evaluator = MetricEvaluator(
    #     original_ads_path="ds/test_data.json",
    #     queries_path="test_queries.json",
    #     index_input_path= "ppo452035_rewritten_ads_few.json",
    #     index_output_dir="ds/faiss_index_rewritten",
    #     original_rankings_path="prompt1_rankings_original.json",
    #     rewritten_rankings_path="rankings_rewritten.json",
    #     original_responses_path="test1_query_responses_original.json",
    #     rewritten_responses_path="query_responses_rewritten1.json",
    #     classified_ads_path="test_classified_ads.json",
    #     k=1
    # )
    # evaluator.run()
    # evaluator = MetricEvaluator(
    #     original_ads_path="ds/test_data.json",
    #     queries_path="test_queries.json",
    #     index_input_path= "ppo452035_rewritten_ads_few.json",
    #     index_output_dir="ds/faiss_index_rewritten",
    #     original_rankings_path="prompt3_rankings_original.json",
    #     rewritten_rankings_path="rankings_rewritten.json",
    #     original_responses_path="test3_query_responses_original.json",
    #     rewritten_responses_path="query_responses_rewritten3.json",
    #     classified_ads_path="test_classified_ads.json",
    #     k=3
    # )
    # evaluator.run()
    # evaluator = MetricEvaluator(
    #     original_ads_path="ds/test_data.json",
    #     queries_path="test_queries.json",
    #     index_input_path= "ppo452035_rewritten_ads_few.json",
    #     index_output_dir="ds/faiss_index_rewritten",
    #     original_rankings_path="prompt5_rankings_original.json",
    #     rewritten_rankings_path="rankings_rewritten.json",
    #     original_responses_path="test5_query_responses_original.json",
    #     rewritten_responses_path="query_responses_rewritten5.json",
    #     classified_ads_path="test_classified_ads.json",
    #     k=5
    # )
    # evaluator.run()
    # evaluator = MetricEvaluator(
    #     original_ads_path="ds/test_data.json",
    #     queries_path="test_queries.json",
    #     index_input_path= "ppo452035_rewritten_ads_few.json",
    #     index_output_dir="ds/faiss_index_rewritten",
    #     original_rankings_path="prompt10_rankings_original.json",
    #     rewritten_rankings_path="rankings_rewritten.json",
    #     original_responses_path="test10_query_responses_original.json",
    #     rewritten_responses_path="query_responses_rewritten10.json",
    #     classified_ads_path="test_classified_ads.json",
    #     k=10
    # )
    # evaluator.run()
    evaluator = MetricEvaluator(
        original_ads_path="ds/test_data.json",
        queries_path="test_queries.json",
        index_input_path= "ppo452035_rewritten_ads_few.json",
        index_output_dir="ds/faiss_index_rewritten",
        original_rankings_path="prompt20_rankings_original.json",
        rewritten_rankings_path="rankings_rewritten.json",
        original_responses_path="test20_query_responses_original.json",
        rewritten_responses_path="query_responses_rewritten20.json",
        classified_ads_path="test_classified_ads.json",
        k=20
    )
    evaluator.run()
    evaluator = MetricEvaluator(
        original_ads_path="ds/test_data.json",
        queries_path="test_queries.json",
        index_input_path= "ppo452035_rewritten_ads_few.json",
        index_output_dir="ds/faiss_index_rewritten",
        original_rankings_path="prompt30_rankings_original.json",
        rewritten_rankings_path="rankings_rewritten.json",
        original_responses_path="test30_query_responses_original.json",
        rewritten_responses_path="query_responses_rewritten30.json",
        classified_ads_path="test_classified_ads.json",
        k=30
    )
    evaluator.run()