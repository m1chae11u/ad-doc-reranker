import argparse  
import torch
from torch import nn
from sentence_transformers import SentenceTransformer

class SimilarityLoss(nn.Module):
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the similarity loss with sentence transformers embeddings.
        
        Args:
            embedding_model_name: Name of the sentence transformer model
        """
        super(SimilarityLoss, self).__init__()
        self.encoder = SentenceTransformer(embedding_model_name)

    def forward(self, queries, original_doc, rewritten_doc, doc_domain, doc_subdomain):
        """
        Compute custom loss: 
        - avg(cos(matching queries, rewritten_doc)) + cos(original_doc, rewritten_doc)

        Args:
            queries: List of dicts with keys: 'domain', 'subdomain', 'query'
            original_doc: string
            rewritten_doc: string
            doc_domain: domain of doc
            doc_subdomain: subdomain of doc

        Returns:
            loss (scalar)
        """
        
        matching_queries = [q["query"] for q in queries if q["domain"] == doc_domain and q["subdomain"] == doc_subdomain]

        with torch.no_grad():
            gen_emb = self.encoder.encode([rewritten_doc], convert_to_tensor=True).squeeze(0)

            sims = []
            for q in matching_queries:
                q_emb = self.encoder.encode([q], convert_to_tensor=True).squeeze(0)
                sim = torch.nn.functional.cosine_similarity(q_emb, gen_emb, dim=0)
                sims.append(sim)
            if sims:
                avg_query_gen_sim = torch.stack(sims).mean()
                
        with torch.no_grad():
            orig_emb = self.encoder.encode([original_doc], convert_to_tensor=True).squeeze(0)
            gen_emb = self.encoder.encode([rewritten_doc], convert_to_tensor=True).squeeze(0)
            sim_orig_gen = torch.nn.functional.cosine_similarity(orig_emb, gen_emb, dim=0)

        # Final loss
        lambda_param = 1.0  # Can be adjusted based on how much you want to preserve content
        loss = -avg_query_gen_sim.mean() + lambda_param * (1 - sim_orig_gen.mean())
        return loss

# testing
queries = [
    {"domain": "electronics", "subdomain": "audio equipment", "query": "behringer juno chorus"},
    {"domain": "electronics", "subdomain": "audio equipment", "query": "best USB audio interface"},
    {"domain": "electronics", "subdomain": "monitors", "query": "studio monitors under 100"},
]

# Original and rewritten ad
original_doc = "Behringer U-Phoria UMC22 is a 2-channel USB interface with MIDAS preamp and phantom power."
rewritten_doc = "Get started recording music with Behringer U-Phoria UMC22, featuring MIDAS preamp and instrument input."
doc_domain = "electronics"
doc_subdomain = "audio equipment"

# Compute the loss
similarity_loss_fn = SimilarityLoss()
loss = similarity_loss_fn(
    queries=queries,
    original_doc=original_doc,
    rewritten_doc=rewritten_doc,
    doc_domain=doc_domain,
    doc_subdomain=doc_subdomain
)

print("Loss:", loss.item())
