import json
from tqdm import tqdm
from retriever import AdSiteRetriever

'''
returns query with all documents ranked per query and the top k docs for that query

usage: python rank_documents.py --query_file queries_200.json --index_dir .\ds\faiss_index\ --output_file rankings.json --top_k 10

'''

class DocumentRanker:
    def __init__(self, index_dir, top_k=200):
        self.retriever = AdSiteRetriever(index_dir=index_dir, top_k=top_k)

    def rank_and_save(self, queries, output_path):
        rankings = []
        for query in tqdm(queries, desc="Ranking documents"):
            full_docs = self.retriever.retrieve_full_documents(query['query'])

            ranked_ids = []
            for doc in full_docs:
                doc_id = doc.metadata.get("doc_id")
                if doc_id:
                    ranked_ids.append(doc_id)

            rankings.append({
                "query": query,
                "ranked_ad_ids": ranked_ids
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rankings, f, indent=2)
        print(f"Saved rankings to {output_path}")

# # Load your queries
# with open("queries_200.json", "r", encoding="utf-8") as f:
#     queries = json.load(f)

# # Create a ranker
# ranker = DocumentRanker(index_dir="data_processing/faiss_index/", top_k=10)

# # Rank and save in one call
# ranker.rank_and_save(queries, "rankings.json")
