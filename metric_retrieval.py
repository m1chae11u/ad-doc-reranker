from collections import defaultdict

class RetrievalMetric:
    def __init__(self, target_doc, queries, ori_rankings, rewritten_rankings):
        """
        Args:
            target_doc: {doc_id: (domain, subdomain)}
            queries: list of dicts with 'query', 'domain', and 'subdomain'
        """
        self.target_doc = target_doc
        self.queries_by_domain = defaultdict(list)
        self.original_rankings = ori_rankings  # {query_str: [doc_ids]}
        self.rewritten_rankings = rewritten_rankings  # {query_str: [doc_ids]}
        self.movements = []

        # Group queries by domain/subdomain pair
        for q in queries:
            domain_pair = (q["domain"], q["subdomain"])
            self.queries_by_domain[domain_pair].append(q["query"])

    def reciprocal_rank(self, rank):
        return 1 / rank if rank > 0 else 0

    def find_rank(self, doc_id, ranked_list):
        return ranked_list.index(doc_id) + 1 if doc_id in ranked_list else -1

    def evaluate_doc(self, doc_id):
        doc_domain = self.target_doc[doc_id]
        relevant_queries = self.queries_by_domain.get(doc_domain)

        for query in relevant_queries:
            orig_ranked = self.original_rankings.get(query, [])
            rewritten_ranked = self.rewritten_rankings.get(query, [])

            orig_rank = self.find_rank(doc_id, orig_ranked)
            rewrite_rank = self.find_rank(doc_id, rewritten_ranked)
            # print (orig_rank, rewrite_rank)

            if orig_rank == -1:
                delta = self.reciprocal_rank(rewrite_rank) - 0
                self.movements.append(delta)
                continue
            if rewrite_rank == -1:
                delta = 0 - self.reciprocal_rank(orig_rank)
                self.movements.append(delta)
                continue

            delta = self.reciprocal_rank(rewrite_rank) - self.reciprocal_rank(orig_rank)
            self.movements.append(delta)

        return sum(self.movements) / len(self.movements)

# # Step 1: Define document metadata
# target_doc = {
#     "doc1": ("tech", "ai")
# }

# # Step 2: Define related queries
# queries = [
#     {"query": "artificial intelligence tools", "domain": "tech", "subdomain": "ai"},
#     {"query": "machine learning platforms", "domain": "tech", "subdomain": "ai"},
# ]

# # Step 3: Define original rankings for each query
# original_rankings = {
#     "artificial intelligence tools": ["doc3", "doc1", "doc2"],
#     "machine learning platforms": ["doc1", "doc2", "doc3"],
# }

# # Step 4: Define rewritten rankings (e.g., from AI-rewritten queries)
# rewritten_rankings = {
#     "artificial intelligence tools": ["doc1", "doc2", "doc3"],
#     "machine learning platforms": ["doc1", "doc2", "doc3"],
# }

# # Step 5: Create and use the metric
# metric = RetrievalMetric(target_doc, queries, original_rankings, rewritten_rankings)

# # Evaluate all documents
# for doc_id in target_doc:
#     print (metric.evaluate_doc(doc_id))

