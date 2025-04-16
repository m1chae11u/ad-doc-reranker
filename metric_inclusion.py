import json

class InclusionAccuracyMetric:
    def __init__(self, k, rankings_before_path, rankings_after_path, inclusions_before_path, inclusions_after_path):
        self.k = k
        self.rankings_before = self._load_json(rankings_before_path)
        self.rankings_after = self._load_json(rankings_after_path)
        self.inclusions_before = self._load_json(inclusions_before_path)
        self.inclusions_after = self._load_json(inclusions_after_path)

    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def compute_inclusion_accuracy(self, target_doc_id):
        # """Compute inclusion accuracy before and after reranking for the target_doc_id."""

        included_before = 0
        included_after = 0
        qualified_queries = 0
        
        for query_num in range(len(self.rankings_before)):
            before_topk = self.rankings_before[query_num]['ranked_ad_ids'][:self.k]
            after_topk = self.rankings_after[query_num]['ranked_ad_ids'][:self.k]
            
            if target_doc_id in before_topk and target_doc_id in after_topk:
                qualified_queries += 1
                before_docs = self.inclusions_before[query_num]["documents_in_response"]
                after_docs = self.inclusions_after[query_num]["documents_in_response"]

                if target_doc_id in before_docs:
                    included_before += 1
                if target_doc_id in after_docs:
                    included_after += 1

        total = qualified_queries
        freq_before = included_before / total
        freq_after = included_after / total
        improvement = freq_before - freq_after

        return {
            "qualified_queries": total,
            "inclusion_before": round(freq_before * 100, 2),
            "inclusion_after": round(freq_after * 100, 2),
            "inclusion_accuracy_increase": round(improvement * 100, 2)
        }

metric = InclusionAccuracyMetric(
    k=10,
    rankings_before_path='rankings_before.json',
    rankings_after_path='rankings_after.json',
    inclusions_before_path='inclusions_before.json',
    inclusions_after_path='inclusions_after.json'
)

result = metric.compute_inclusion_accuracy("a767622a-e5f8-48f9-b026-e960bb488e96")
print(result)