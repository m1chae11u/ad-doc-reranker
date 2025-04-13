import json

class MRR_Evaluator:
    def __init__(self, query_results):
        """
        Initializes the evaluator with query results.
        
        Args:
            query_results: List of query results, each containing the query, ranked documents, and true relevance
        """
        self.query_results = query_results

    def evaluate_mrr(self):
        """
        Evaluates Mean Reciprocal Rank (MRR)
        
        Returns:
            Mean Reciprocal Rank (MRR)
        """
        mrr_total = 0  # Tracks the MRR across all queries
        num_queries = len(self.query_results)
        
        for result in self.query_results:
            top_k_docs = result['top_k_docs']  # List of top-k ranked documents from LLM
            relevant_docs = result['relevant_docs']  # List of relevant docs

            # 1. Calculate MRR (Mean Reciprocal Rank)
            first_relevant_rank = None
            for rank, doc in enumerate(top_k_docs, start=1):
                if doc in relevant_docs:
                    first_relevant_rank = rank
                    break
            if first_relevant_rank is not None:
                mrr_total += 1 / first_relevant_rank
        
        # Calculate the Mean Reciprocal Rank (MRR)
        mrr = mrr_total / num_queries
        return mrr

# Load the query data from the JSON file
def load_query_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

        query_results = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            print(f"Skipping non-dict entry at index {i}: {entry}")
            continue
        if "user_query" not in entry or "title" not in entry:
            print(f"Skipping incomplete entry at index {i}: {entry}")
            continue

        user_query = entry["user_query"]
        title = entry["title"]

        query_results.append({
            "query": user_query,
            "top_k_docs": [title],      # Pretend LLM returns the title
            "relevant_docs": [title]    # Pretend it’s the relevant doc too
        })


    return query_results



json_file = "/Users/diyasharma/Downloads/200_query_responses.json"  # Provide the path to your JSON file
query_results = load_query_data(json_file)

# Create an evaluator instance
evaluator = MRR_Evaluator(query_results)

# Evaluate MRR
mrr = evaluator.evaluate_mrr()


print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
