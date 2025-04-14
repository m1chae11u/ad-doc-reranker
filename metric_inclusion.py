import json

class InclusionAccuracyMetric:
    def __init__(self, k=5):
        """
        Initialize the evaluator.
        
        Args:
            k: Top-k value to consider (1, 3, 5, 10, etc.)
        """
        self.k = k

    def forward(self, dataset, doc_id):
        
        before_includes = 0
        after_includes = 0
        total_considered = 0
        contributing_query_ids = []

        for entry in dataset:
            top_k_before = entry.get("top_k_before", [])[:self.k]
            top_k_after = entry.get("top_k_after", [])[:self.k]
            response_before = entry.get("llm_response_before", "").lower()
            response_after = entry.get("llm_response_after", "").lower()

            if doc_id in top_k_before and doc_id in top_k_after:
                total_considered += 1
                contributing_query_ids.append(entry["query_id"])
                if doc_id.lower() in response_before:
                    before_includes += 1
                if doc_id.lower() in response_after:
                    after_includes += 1

        if total_considered == 0:
            return 0.0, 0.0, 0.0, []

        rate_before = before_includes / total_considered
        rate_after = after_includes / total_considered
        accuracy_gain = rate_after - rate_before

        return rate_before, rate_after, accuracy_gain, contributing_query_ids


# ===== Example usage =====
if __name__ == "__main__":
    # Load your evaluation data
    with open("reranking_eval_data.json", "r") as f:
        dataset = json.load(f)

    # Initialize the evaluator
    evaluator = InclusionAccuracyMetric(k=5)

    # Define the document of interest
    doc_id = "doc1"

    # Run the inclusion accuracy evaluation
    rate_before, rate_after, gain, used_queries = evaluator.forward(dataset, doc_id)

    # Print results
    print(f"Inclusion Rate BEFORE: {rate_before:.2%}")
    print(f"Inclusion Rate AFTER:  {rate_after:.2%}")
    print(f" Inclusion Accuracy Gain: {gain:.2%}")
    print(f" Evaluated on queries: {used_queries}")
