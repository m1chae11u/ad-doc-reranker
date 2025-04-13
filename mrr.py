import json
import os

class MRR_Evaluator:
    def __init__(self, query_results):
        self.query_results = query_results
    
    def evaluate_mrr(self, k=10):
        mrr_total = 0
        num_queries = len(self.query_results)
        
        if num_queries == 0:
            return 0.0

        for result in self.query_results:
            retrieved_docs = result.get('retrieved_docs', [])[:k]  # Top k documents
            relevant_docs = set(result.get('relevant_docs', [])) 
            
            # Find first relevant document in retrieved list
            for rank, doc in enumerate(retrieved_docs, start=1):
                if doc in relevant_docs:
                    mrr_total += 1.0 / rank
                    break

        return mrr_total / num_queries

def load_query_data(query_path, ads_path):
    # Load queries and responses
    try:
        with open(query_path, 'r') as f:
            queries = json.load(f)
    except Exception as e:
        print(f"Error loading query file: {e}")
        return []
    
    # Load ads data
    try:
        with open(ads_path, 'r') as f:
            ads_data = json.load(f)
    except Exception as e:
        print(f"Error loading ads file: {e}")
        return []
    
    # Mapping of ad IDs to their data
    ad_id_to_data = {ad['ad_id']: ad for ad in ads_data}
    
    query_results = []
    
    for query in queries:
        if not isinstance(query, dict):
            continue
            
        # Retrieved documents
        retrieved_docs = query.get('retrieved_documents', [])
        
        # Relevant Responses
        relevant_docs = []
        if 'llm_response' in query:
            relevant_docs = extract_relevant_docs_from_response(query['llm_response'])
        
        query_results.append({
            'query': query.get('user_query', ''),
            'retrieved_docs': retrieved_docs,
            'relevant_docs': relevant_docs
        })
    
    return query_results

def extract_relevant_docs_from_response(response):
    
    return []

query_path = "/Users/diyasharma/Downloads/200_query_responses.json"
ads_path = "/Users/diyasharma/Downloads/algoverse_sampled_ads.json"

query_results = load_query_data(query_path, ads_path)

if query_results:
    evaluator = MRR_Evaluator(query_results)
    mrr = evaluator.evaluate_mrr(k=10) #Assuming k =10
    print(f"Mean Reciprocal Rank: {mrr:.4f}")
else:
    print("No valid queries loaded")
