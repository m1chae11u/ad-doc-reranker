from collections import defaultdict

class MultiQueryRetrievalMetric:
    def __init__(self, doc_metadata, query_metadata):
        """
        Args:
            doc_metadata: dict of {doc_id: (domain, subdomain)}
            query_metadata: dict of {query_id: (domain, subdomain)}
        """
        self.doc_metadata = doc_metadata
        self.query_metadata = query_metadata
        self.movement_by_doc = defaultdict(list)

    def reciprocal_rank(self, rank):
        return 1 / rank if rank > 0 else 0

    def find_rank(self, doc_id, ranked_list):
        return ranked_list.index(doc_id) + 1 if doc_id in ranked_list else -1

    def process_query(self, query_id, original_ranking, rewritten_ranking):
        query_domain = self.query_metadata.get(query_id)
        if query_domain is None:
            return

        for doc_id in set(original_ranking) | set(rewritten_ranking):
            doc_domain = self.doc_metadata.get(doc_id)
            if doc_domain != query_domain:
                continue

            orig_rank = self.find_rank(doc_id, original_ranking)
            rewrite_rank = self.find_rank(doc_id, rewritten_ranking)

            if orig_rank == -1 or rewrite_rank == -1:
                continue

            rr_orig = self.reciprocal_rank(orig_rank)
            rr_rewrite = self.reciprocal_rank(rewrite_rank)
            movement = rr_rewrite - rr_orig

            self.movement_by_doc[doc_id].append(movement)

    def summarize(self):
        return {
            doc_id: sum(moves) / len(moves)
            for doc_id, moves in self.movement_by_doc.items()
        }
