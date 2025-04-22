from collections import defaultdict

class MultiQueryRetrievalMetric:
    def __init__(self, doc_metadata, queries):
        """
        Args:
            doc_metadata: dict of {doc_id: (domain, subdomain)}
            queries: list of dicts with 'query', 'domain', and 'subdomain'
        """
        self.doc_metadata = doc_metadata
        self.queries_by_domain = defaultdict(list)
        self.original_rankings = {}  # {query_str: [doc_ids]}
        self.rewritten_rankings = {}  # {query_str: [doc_ids]}
        self.movement_by_doc = defaultdict(list)

        # Group queries by domain/subdomain pair
        for q in queries:
            domain_pair = (q["domain"], q["subdomain"])
            self.queries_by_domain[domain_pair].append(q["query"])

    def set_rankings(self, original, rewritten):
        self.original_rankings = original
        self.rewritten_rankings = rewritten

    def reciprocal_rank(self, rank):
        return 1 / rank if rank > 0 else 0

    def find_rank(self, doc_id, ranked_list):
        return ranked_list.index(doc_id) + 1 if doc_id in ranked_list else -1

    def evaluate_doc(self, doc_id):
        if doc_id not in self.doc_metadata:
            return

        doc_domain = self.doc_metadata[doc_id]
        relevant_queries = self.queries_by_domain.get(doc_domain, [])

        for query in relevant_queries:
            orig_ranked = self.original_rankings.get(query, [])
            rewritten_ranked = self.rewritten_rankings.get(query, [])

            orig_rank = self.find_rank(doc_id, orig_ranked)
            rewrite_rank = self.find_rank(doc_id, rewritten_ranked)

            if orig_rank == -1 or rewrite_rank == -1:
                continue

            delta = self.reciprocal_rank(rewrite_rank) - self.reciprocal_rank(orig_rank)
            self.movement_by_doc[doc_id].append(delta)

    def summarize(self):
        return {
            doc_id: sum(moves) / len(moves)
            for doc_id, moves in self.movement_by_doc.items()
        }
