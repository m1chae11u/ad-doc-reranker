import json
from tqdm import tqdm
from retriever import AdSiteRetriever

'''
returns query with all documents ranked per query and the top k docs for that query

usage: python rank_documents.py --query_file queries_200.json --index_dir .\ds\faiss_index\ --output_file rankings.json --top_k 10

'''

def rank_documents_using_retriever(queries, index_dir, output_path, top_k=100):
    retriever = AdSiteRetriever(index_dir=index_dir, top_k=200)

    rankings = []

    for query in tqdm(queries, desc="Ranking documents"):
        full_docs = retriever.retrieve_full_documents(query['query'])

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Use AdSiteRetriever to rank full documents for each query.")
    parser.add_argument("--query_file", type=str, required=True, help="Path to JSON file containing queries.")
    parser.add_argument("--index_dir", type=str, required=True, help="Path to FAISS index directory.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save rankings.")
    parser.add_argument("--top_k", type=int, default=100, help="Top K documents to retrieve per query.")

    args = parser.parse_args()

    with open(args.query_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    rank_documents_using_retriever(
        queries=queries,
        index_dir=args.index_dir,
        output_path=args.output_file,
        top_k=args.top_k
    )